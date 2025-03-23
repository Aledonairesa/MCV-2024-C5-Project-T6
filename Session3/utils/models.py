import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import ResNetModel
import torchvision.models as models
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# Baseline model
class Model(nn.Module):
    def __init__(self, device, NUM_CHAR, char2idx, TEXT_MAX_LEN=151):
        super().__init__()
        self.device = device
        self.char2idx = char2idx
        self.NUM_CHAR = NUM_CHAR
        self.TEXT_MAX_LEN = TEXT_MAX_LEN
        self.resnet = ResNetModel.from_pretrained('microsoft/resnet-18').to(device)
        self.gru = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        
        # Get image features from ResNet
        feat = self.resnet(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # 1, batch, 512
        
        # Initialize hidden state with image features
        hidden = feat
        
        # Initialize first input with zeros instead of start token
        current_input = torch.zeros(1, batch_size, 512).to(self.device)
        
        # Store all outputs
        outputs = []
        predicted_tokens = []
        
        # Generate exactly TEXT_MAX_LEN tokens
        for t in range(self.TEXT_MAX_LEN):
            # Run GRU for current step
            output, hidden = self.gru(current_input, hidden)  # output: 1, batch, 512
            
            # Project to vocabulary size
            logits = self.proj(output.squeeze(0))  # batch, NUM_CHAR
            
            # Get predicted token
            probs = F.softmax(logits, dim=1)
            predicted_token = torch.argmax(probs, dim=1)  # batch
            
            # Store outputs
            outputs.append(logits)
            predicted_tokens.append(predicted_token)
            
            # Create embedding for next input
            current_input = self.embed(predicted_token).unsqueeze(0)  # 1, batch, 512
        
        # Stack all outputs
        logits = torch.stack(outputs, dim=1)  # batch, seq_len, NUM_CHAR
        tokens = torch.stack(predicted_tokens, dim=1)  # batch, seq_len
        
        # Reshape logits to match expected output format (batch, NUM_CHAR, seq_len)
        logits = logits.permute(0, 2, 1)
        
        return logits, tokens

#Custom model
class ModelCustom(nn.Module):
    def __init__(self, device, vocab_size, TEXT_MAX_LEN=151, encoder_name='resnet18', embedding_dim=512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        self.TEXT_MAX_LEN = TEXT_MAX_LEN

        # Available encoders
        encoder_map = {
            'resnet18': (ResNetModel, 'microsoft/resnet-18', 512),
            'resnet34': (ResNetModel, 'microsoft/resnet-34', 512),
            'resnet50': (ResNetModel, 'microsoft/resnet-50', 2048),
            'vgg16':    (models.vgg16, 512*2*2), # Dim after pooling
            'vgg19':    (models.vgg19, 512*2*2), # Dim after pooling
        }

        if encoder_name not in encoder_map:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from {list(encoder_map.keys())}.")

        if "resnet" in encoder_name:
            model_class, model_str, out_dim = encoder_map[encoder_name]
            self.encoder = model_class.from_pretrained(model_str).to(device)
        else:
            model_fn, out_dim = encoder_map[encoder_name]
            self.encoder = model_fn(pretrained=True).to(device)
            self.encoder.classifier = nn.Identity() # Remove FC layers, keep only feature extractor
            self.encoder.avgpool = nn.AdaptiveAvgPool2d(2) # Reduce from 512x7x7 to 512x2x2

        # Projection layer to unify feature size
        self.feature_proj = nn.Linear(out_dim, embedding_dim)

        # GRU Decoder
        self.gru = nn.GRU(embedding_dim, embedding_dim, num_layers=1)
        self.proj = nn.Linear(embedding_dim, vocab_size)
        self.embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, img):
        batch_size = img.shape[0]

        # Forward pass through encoder
        if isinstance(self.encoder, ResNetModel):
            feat = self.encoder(img).pooler_output  # Shape: [batch_size, out_dim, 1, 1]
            feat = feat.view(batch_size, -1)  # Flatten [batch_size, 512] (RN-18,34) or [batch_size, 2048] (RN-50)
        else: # VGG Case
            feat = self.encoder.features(img)  # Extract feature maps
            feat = self.encoder.avgpool(feat)  # Adaptive pooling [batch_size, 512, 2, 2]
            feat = feat.view(batch_size, -1)  # Flatten [batch_size, 2048]

        # Project features to embedding_dim for GRU
        feat = self.feature_proj(feat)  # [batch_size, embedding_dim]
        feat = feat.unsqueeze(0)  # [1, batch_size, embedding_dim]

        # Start token embedding (assuming start token is index 0)
        start = torch.zeros(batch_size, dtype=torch.long).to(self.device)  # Start token
        start_embed = self.embed(start)  # [batch_size, embedding_dim]
        start_embeds = start_embed.unsqueeze(0)  # [1, batch_size, embedding_dim]

        inp = start_embeds
        hidden = feat  # Initialize GRU hidden state

        # Decode sequence
        outputs = [inp]
        for t in range(self.TEXT_MAX_LEN - 1):  # -1 because we've already fed the start token
            out, hidden = self.gru(inp, hidden)
            outputs.append(out)
            inp = out  # Teacher forcing disabled - use previous output as next input

        # Concatenate all outputs
        seq_output = torch.cat(outputs, dim=0)  # [seq_len, batch_size, embedding_dim]
        
        # Final projection
        res = seq_output.permute(1, 0, 2)  # [batch_size, seq_len, embedding_dim]
        res = self.proj(res)  # [batch_size, seq_len, vocab_size]
        res = res.permute(0, 2, 1)  # [batch_size, vocab_size, seq_len]

        # To get probabilities
        probs = F.softmax(res, dim=1)
        # To get token indices
        tokens = torch.argmax(probs, dim=1)  # [batch_size, seq_len]

        return res, tokens

# Additive (Bahdanau) attention
class Attention(nn.Module):
    def __init__(self, hidden_dim, encoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)    # project encoder features
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)     # project decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)                 # compute attention scores

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: [batch_size, num_regions, encoder_dim]
        # decoder_hidden: [batch_size, hidden_dim]
        att1 = self.encoder_att(encoder_out)                       # [batch, num_regions, attention_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)       # [batch, 1, attention_dim]
        att = torch.tanh(att1 + att2)                              # [batch, num_regions, attention_dim]
        e = self.full_att(att).squeeze(2)                          # [batch, num_regions]
        alpha = F.softmax(e, dim=1)                                # [batch, num_regions]
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)    # [batch, encoder_dim]
        return context, alpha



class ModelWithAttention(nn.Module):
    def __init__(self, device, tokenizer, TEXT_MAX_LEN=151, encoder_name='resnet18', decoder_name='gru'):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.TEXT_MAX_LEN=TEXT_MAX_LEN
        self.device = device
        # ------------------ Configure ENCODER ------------------
        encoder_map = {
            'resnet18': (ResNetModel, 'microsoft/resnet-18', 512),
            'resnet34': (ResNetModel, 'microsoft/resnet-34', 512),
            'resnet50': (ResNetModel, 'microsoft/resnet-50', 2048),
            'vgg16': (models.vgg16, 512),
            'vgg19': (models.vgg19, 512),
        }

        if encoder_name not in encoder_map:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from {list(encoder_map.keys())}.")

        model_class, model_str, out_dim = encoder_map[encoder_name]
        self.encoder = model_class.from_pretrained(model_str).to(device)
        self.encoder_out_dim = out_dim
        self.feature_proj = nn.Linear(out_dim, 512)  # Project encoder features

        # ------------------ Configure DECODER ------------------
        self.decoder_type = decoder_name
        if decoder_name == 'gru':
            self.decoder = nn.GRU(512, 512, num_layers=1)
        elif decoder_name == 'lstm':
            self.decoder = nn.LSTM(512, 512, num_layers=1)
        elif decoder_name == 'xlstm':
            self.decoder = nn.LSTM(512, 512, num_layers=2, dropout=0.1)
        else:
            raise ValueError(f"Unknown decoder type '{decoder_name}'.")

        # Attention module and embedding layers
        self.attention = Attention(hidden_dim=512, encoder_dim=512, attention_dim=256)
        self.attention_combine = nn.Linear(512 + 512, 512)
        self.embed = nn.Embedding(self.vocab_size, 512)
        self.proj = nn.Linear(512, self.vocab_size)

    def forward(self, img):
        batch_size = img.shape[0]

        # ------------------ Encoder ------------------
        enc_out = self.encoder(img).last_hidden_state  # [batch_size, out_dim, 7, 7]
        enc_out = enc_out.view(batch_size, self.encoder_out_dim, -1).permute(0, 2, 1)
        enc_out = self.feature_proj(enc_out)  # [batch_size, num_regions, 512]

        # ------------------ Initialize Decoder Hidden State ------------------
        mean_enc = enc_out.mean(dim=1)  # [batch_size, 512]
        if self.decoder_type in ['lstm', 'xlstm']:
            hidden = (mean_enc.unsqueeze(0).repeat(2, 1, 1), mean_enc.unsqueeze(0).repeat(2, 1, 1))
        else:
            hidden = mean_enc.unsqueeze(0)

        # Select the correct start token for different tokenization modes
        if self.tokenizer.mode == "char":
            start_token = self.tokenizer.char2idx['<SOS>']
        elif self.tokenizer.mode == "wordpiece":
            start_token = self.tokenizer.wordpiece_tokenizer.cls_token_id
        elif self.tokenizer.mode == "word":
            start_token = self.tokenizer.word2idx['<SOS>']
        else:
            raise ValueError("Unsupported tokenization mode.")

        start = torch.tensor([start_token], dtype=torch.long).to(self.device)
        start_embed = self.embed(start).repeat(batch_size, 1).unsqueeze(0)
        inp = start_embed

        outputs = [inp]

        # ------------------ Decoder with Attention ------------------
        for _ in range(self.TEXT_MAX_LEN - 1):
            hidden_for_attention = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            context, alpha = self.attention(enc_out, hidden_for_attention)

            combined = torch.cat((inp.squeeze(0), context), dim=1)
            combined = torch.tanh(self.attention_combine(combined)).unsqueeze(0)

            out, hidden = self.decoder(combined, hidden)
            outputs.append(out)
            inp = out

        # ------------------ Final Projection ------------------
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2)  # [batch_size, seq, 512]
        res = self.proj(outputs).permute(0, 2, 1)  # [batch_size, vocab_size, seq]
        tokens = torch.argmax(F.softmax(res, dim=1), dim=1)

        return res, tokens
    

class XLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(XLSTMCell, self).__init__()
        
        # Input gate
        self.W_ix = nn.Linear(input_size, hidden_size)
        self.W_ih = nn.Linear(hidden_size, hidden_size)
        self.W_im = nn.Linear(hidden_size, hidden_size)
        
        # Forget gate
        self.W_fx = nn.Linear(input_size, hidden_size)
        self.W_fh = nn.Linear(hidden_size, hidden_size)
        self.W_fm = nn.Linear(hidden_size, hidden_size)
        
        # Output gate
        self.W_ox = nn.Linear(input_size, hidden_size)
        self.W_oh = nn.Linear(hidden_size, hidden_size)
        self.W_om = nn.Linear(hidden_size, hidden_size)
        
        # Cell update
        self.W_cx = nn.Linear(input_size, hidden_size)
        self.W_ch = nn.Linear(hidden_size, hidden_size)
        
        # Extended memory gate
        self.W_ex = nn.Linear(input_size, hidden_size)
        self.W_eh = nn.Linear(hidden_size, hidden_size)
        self.W_em = nn.Linear(hidden_size, hidden_size)
        
        # Projection layer for extended memory
        self.W_pm = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization for improved training stability
        self.ln_cell = nn.LayerNorm(hidden_size)
        self.ln_hidden = nn.LayerNorm(hidden_size)
        self.ln_memory = nn.LayerNorm(hidden_size)
        
    def forward(self, x, hidden):
        # Unpack the hidden state
        h_prev, c_prev, m_prev = hidden
        
        # Input gate
        i = torch.sigmoid(self.W_ix(x) + self.W_ih(h_prev) + self.W_im(m_prev))
        
        # Forget gate
        f = torch.sigmoid(self.W_fx(x) + self.W_fh(h_prev) + self.W_fm(m_prev))
        
        # Output gate
        o = torch.sigmoid(self.W_ox(x) + self.W_oh(h_prev) + self.W_om(m_prev))
        
        # Extended memory gate
        e = torch.sigmoid(self.W_ex(x) + self.W_eh(h_prev) + self.W_em(m_prev))
        
        # Cell update
        c_tilde = torch.tanh(self.W_cx(x) + self.W_ch(h_prev))
        
        # Update cell state
        c = f * c_prev + i * c_tilde
        c = self.ln_cell(c)
        
        # Update extended memory
        m = (1 - e) * m_prev + e * self.W_pm(c)
        m = self.ln_memory(m)
        
        # Update hidden state
        h = o * torch.tanh(c + m)
        h = self.ln_hidden(h)
        
        return h, (h, c, m)


class XLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1, bidirectional=False):
        super(XLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Create XLSTMCell for each layer
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size * (2 if bidirectional else 1)
            self.cells.append(XLSTMCell(layer_input_size, hidden_size))
            
            # If bidirectional, add a second cell for backward direction
            if bidirectional:
                self.cells.append(XLSTMCell(layer_input_size, hidden_size))
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def _forward_step(self, x, hidden, layer_idx, direction=0):
        """Perform one step for one layer in one direction"""
        cell = self.cells[layer_idx * (2 if self.bidirectional else 1) + direction]
        h, new_hidden = cell(x, hidden)
        return h, new_hidden
    
    def forward(self, x, hidden=None):
        """
        Forward pass of XLSTM
        
        Args:
            x: input tensor of shape [seq_len, batch_size, input_size]
            hidden: initial hidden state tuple (h, c, m) or None
            
        Returns:
            output: output tensor of shape [seq_len, batch_size, hidden_size * (2 if bidirectional else 1)]
            hidden: final hidden state
        """
        seq_len, batch_size, _ = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            h = x.new_zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)
            c = x.new_zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)
            m = x.new_zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size, self.hidden_size)
            hidden = [(h[i], c[i], m[i]) for i in range(self.num_layers * (2 if self.bidirectional else 1))]
        
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[t]
            
            layer_outputs = []
            new_hidden = []
            
            # Process each layer
            for layer in range(self.num_layers):
                if self.bidirectional:
                    # Forward direction
                    h_forward, new_h_forward = self._forward_step(
                        x_t, hidden[layer*2], layer, 0
                    )
                    
                    # Backward direction (process sequence in reverse)
                    h_backward, new_h_backward = self._forward_step(
                        x[seq_len-1-t], hidden[layer*2+1], layer, 1
                    )
                    
                    # Concatenate outputs from both directions
                    h_combined = torch.cat([h_forward, h_backward], dim=1)
                    
                    # Update x_t for next layer
                    x_t = self.dropout_layer(h_combined) if layer < self.num_layers - 1 else h_combined
                    
                    # Save new hidden states
                    new_hidden.extend([new_h_forward, new_h_backward])
                    layer_outputs.append(h_combined)
                    
                else:
                    # Single direction
                    h, new_h = self._forward_step(x_t, hidden[layer], layer)
                    
                    # Update x_t for next layer
                    x_t = self.dropout_layer(h) if layer < self.num_layers - 1 else h
                    
                    # Save new hidden states
                    new_hidden.append(new_h)
                    layer_outputs.append(h)
            
            # Update hidden states for next time step
            hidden = new_hidden
            
            # Add output of the final layer to our outputs
            outputs.append(layer_outputs[-1])
        
        # Stack outputs along the sequence dimension
        outputs = torch.stack(outputs)
        
        return outputs, hidden
    
class Attention2(nn.Module):
    def __init__(self, encoder_dim, hidden_dim, attention_dim):
        super().__init__()
        # Project encoder features to attention space
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # Project decoder hidden state to attention space
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        # Calculate attention scores
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out shape: [batch_size, num_regions, encoder_dim]
        # decoder_hidden shape: [batch_size, hidden_dim] or [1, batch_size, hidden_dim]
        
        # Ensure decoder_hidden has the right shape [batch_size, hidden_dim]
        if decoder_hidden.dim() == 3:
            decoder_hidden = decoder_hidden.squeeze(0)
            
        att1 = self.encoder_att(encoder_out)  # [batch_size, num_regions, attention_dim]
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # [batch_size, 1, attention_dim]
        
        # Now att2 will broadcast properly across all regions
        att = self.relu(att1 + att2)  # [batch_size, num_regions, attention_dim]
        att = self.full_att(att).squeeze(2)  # [batch_size, num_regions]
        alpha = self.softmax(att)  # [batch_size, num_regions]
        
        # Weighted sum of encoder outputs
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch_size, encoder_dim]
        
        return context, alpha
    
class ModelWithAttention2(nn.Module):
    def __init__(self, device, tokenizer, TEXT_MAX_LEN=151, encoder_name='resnet18', decoder_name='gru'):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.TEXT_MAX_LEN=TEXT_MAX_LEN
        vocab_size = self.tokenizer.vocab_size

        # ------------------ Configure ENCODER ------------------
        encoder_map = {
            'resnet18': (ResNetModel, 'microsoft/resnet-18', 512),
            'resnet34': (ResNetModel, 'microsoft/resnet-34', 512),
            'resnet50': (ResNetModel, 'microsoft/resnet-50', 2048),
            'vgg16': (models.vgg16, 512),
            'vgg19': (models.vgg19, 512),
        }

        if encoder_name not in encoder_map:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from {list(encoder_map.keys())}.")

        model_class, model_str, out_dim = encoder_map[encoder_name]
        self.encoder = model_class.from_pretrained(model_str).to(self.device)
        self.encoder_out_dim = out_dim
        self.feature_proj = nn.Linear(out_dim, 512)  # Project encoder features

        # ------------------ Configure DECODER ------------------
        self.decoder_type = decoder_name
        if decoder_name == 'gru':
            self.decoder = nn.GRU(512, 512, num_layers=1)
        elif decoder_name == 'lstm':
            self.decoder = nn.LSTM(512, 512, num_layers=1)
        elif decoder_name == 'xlstm':
            self.decoder = XLSTM(512, 512, num_layers=2, dropout=0.1)
        else:
            raise ValueError(f"Unknown decoder type '{decoder_name}'.")

        # Attention module and embedding layers
        self.attention = Attention2(hidden_dim=512, encoder_dim=512, attention_dim=256)
        self.attention_combine = nn.Linear(512 + 512, 512)
        self.embed = nn.Embedding(vocab_size, 512)
        self.proj = nn.Linear(512, vocab_size)

    def forward(self, img, captions=None, teacher_forcing_ratio=0.5):
        """
        Forward pass with support for teacher forcing
        
        Args:
            img: Input images [batch_size, channels, height, width]
            captions: Target captions for teacher forcing [batch_size, seq_len] (optional)
            teacher_forcing_ratio: Probability of using teacher forcing (default: 0.5)
            
        Returns:
            res: Output logits [batch_size, vocab_size, seq_len]
            tokens: Predicted tokens [batch_size, seq_len]
        """
        batch_size = img.shape[0]
        use_teacher_forcing = captions is not None and random.random() < teacher_forcing_ratio

        # ------------------ Encoder ------------------
        enc_out = self.encoder(img).last_hidden_state  # [batch_size, out_dim, 7, 7]
        enc_out = enc_out.view(batch_size, self.encoder_out_dim, -1).permute(0, 2, 1)
        enc_out = self.feature_proj(enc_out)  # [batch_size, num_regions, 512]

        # ------------------ Initialize Decoder Hidden State ------------------
        mean_enc = enc_out.mean(dim=1)  # [batch_size, 512]
        
        # Initialize hidden states based on decoder type
        if self.decoder_type == 'xlstm':
            # For XLSTM, we need to initialize h, c, and m for each layer
            num_layers = 2  # XLSTM has 2 layers as defined in __init__
            h = mean_enc.unsqueeze(0).repeat(num_layers, 1, 1)
            c = mean_enc.unsqueeze(0).repeat(num_layers, 1, 1)
            m = mean_enc.unsqueeze(0).repeat(num_layers, 1, 1)
            
            # Create a list of tuples (h, c, m) for each layer
            hidden = [(h[i], c[i], m[i]) for i in range(num_layers)]
        elif self.decoder_type == 'lstm':
            hidden = (mean_enc.unsqueeze(0), mean_enc.unsqueeze(0))
        else:  # GRU
            hidden = mean_enc.unsqueeze(0)

        # Select the correct start token for different tokenization modes
        if self.tokenizer.mode == "char":
            start_token = self.tokenizer.char2idx['<SOS>']
        elif self.tokenizer.mode == "wordpiece":
            start_token = self.tokenizer.wordpiece_tokenizer.cls_token_id
        elif self.tokenizer.mode == "word":
            start_token = self.tokenizer.word2idx['<SOS>']
        else:
            raise ValueError("Unsupported tokenization mode.")

        start = torch.tensor([start_token], dtype=torch.long).to(self.device)
        start_embed = self.embed(start).repeat(batch_size, 1).unsqueeze(0)
        inp = start_embed

        outputs = [inp]
        tokens_list = []
        current_token = start.repeat(batch_size)
        tokens_list.append(current_token)

        # ------------------ Decoder with Attention ------------------
        for t in range(self.TEXT_MAX_LEN - 1):
            # Get the appropriate hidden state for attention
            if self.decoder_type == 'xlstm':
                # For XLSTM, use the h from the last layer's tuple
                hidden_for_attention = hidden[-1][0]  # Last layer's h
            elif self.decoder_type == 'lstm':
                hidden_for_attention = hidden[0]  # h from (h, c) tuple
            else:  # GRU
                hidden_for_attention = hidden

            # Apply attention
            context, alpha = self.attention(enc_out, hidden_for_attention)

            # Combine current input with context vector
            combined = torch.cat((inp.squeeze(0), context), dim=1)
            combined = torch.tanh(self.attention_combine(combined)).unsqueeze(0)

            # Decoder step
            out, hidden = self.decoder(combined, hidden)
            
            # Reshape out if necessary (XLSTM returns sequence, batch, features)
            if self.decoder_type == 'xlstm':
                out = out.squeeze(0)  # Remove sequence dimension

            outputs.append(out)

            # Project to vocabulary space for this step
            step_proj = self.proj(out.transpose(0, 1)).transpose(1, 2)  # [batch_size, vocab_size, 1]
            current_token = torch.argmax(F.softmax(step_proj, dim=1), dim=1).squeeze(-1)
            tokens_list.append(current_token)
            
            # Teacher forcing: decide whether to use ground truth or predicted token
            if use_teacher_forcing and t < captions.size(1) - 1:
                # Use target token as next input
                teacher_input = captions[:, t+1]
                inp = self.embed(teacher_input).unsqueeze(0)
            else:
                # Use model's own prediction as next input
                inp = out
        
        # ------------------ Final Projection ------------------
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2)  # [batch_size, seq, 512]
        res = self.proj(outputs).permute(0, 2, 1)  # [batch_size, vocab_size, seq]
        tokens = torch.stack(tokens_list, dim=1)  # [batch_size, seq_len]

        return res, tokens
    
    def inference(self, img):
        """
        Inference method for validation and testing (no teacher forcing)
        
        Args:
            img: Input images [batch_size, channels, height, width]
            
        Returns:
            res: Output logits [batch_size, vocab_size, seq_len]
            tokens: Predicted tokens [batch_size, seq_len]
        """
        with torch.no_grad():
            return self.forward(img, captions=None, teacher_forcing_ratio=0)



class ModelCustom2(nn.Module):
    def __init__(self, device, tokenizer, TEXT_MAX_LEN=151, encoder_name='resnet18', decoder_name='gru'):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
        self.TEXT_MAX_LEN = TEXT_MAX_LEN
        vocab_size = self.tokenizer.vocab_size

        # ------------------ Configure ENCODER ------------------
        encoder_map = {
            'resnet18': (ResNetModel, 'microsoft/resnet-18', 512),
            'resnet34': (ResNetModel, 'microsoft/resnet-34', 512),
            'resnet50': (ResNetModel, 'microsoft/resnet-50', 2048),
            'vgg16': (models.vgg16, 512),
            'vgg19': (models.vgg19, 512),
        }

        model_class, model_str, out_dim = encoder_map[encoder_name]
        self.encoder = model_class.from_pretrained(model_str).to(self.device)
        self.encoder_out_dim = out_dim
        
        # Project encoder output to match decoder hidden size
        self.feature_proj = nn.Linear(out_dim, 512)  

        # ------------------ Configure DECODER ------------------
        self.decoder_type = decoder_name
        if decoder_name == 'gru':
            self.decoder = nn.GRU(512, 512, num_layers=1)
        elif decoder_name == 'lstm':
            self.decoder = nn.LSTM(512, 512, num_layers=1)
        elif decoder_name == 'xlstm':
            self.decoder = XLSTM(512, 512, num_layers=2, dropout=0.1)
            # Initialize hidden state projection for XLSTM
            self.hidden_init = nn.Linear(512, 512 * 3 * 2)  # For h, c, m across 2 layers
        else:
            raise ValueError(f"Unknown decoder type '{decoder_name}'.")

        # Embedding and projection layers
        self.embed = nn.Embedding(vocab_size, 512)
        self.proj = nn.Linear(512, vocab_size)

    def forward(self, img, captions=None, teacher_forcing_ratio=0.5):
        """
        Forward pass with support for teacher forcing
        
        Args:
            img: Input images [batch_size, channels, height, width]
            captions: Target captions for teacher forcing [batch_size, seq_len] (optional)
            teacher_forcing_ratio: Probability of using teacher forcing (default: 0.5)
            
        Returns:
            res: Output logits [batch_size, vocab_size, seq_len]
            tokens: Predicted tokens [batch_size, seq_len]
        """
        batch_size = img.shape[0]
        use_teacher_forcing = True if (captions is not None and random.random() < teacher_forcing_ratio) else False

        # ------------------ Encoder ------------------
        enc_out = self.encoder(img).last_hidden_state  # [batch_size, out_dim, 7, 7]
        
        # Average pooling over spatial dimensions
        enc_out = enc_out.view(batch_size, self.encoder_out_dim, -1).mean(dim=2)  # [batch_size, out_dim]
        enc_features = self.feature_proj(enc_out)  # [batch_size, 512]

        # ------------------ Initialize Decoder Hidden State ------------------
        # Initialize hidden states based on decoder type
        if self.decoder_type == 'xlstm':
            # For XLSTM, transform image features into initial (h, c, m) states for each layer
            hidden_states = self.hidden_init(feat)  # [batch_size, 512*3*2]
            hidden_states = hidden_states.view(batch_size, 6, 512)  # [batch_size, 6, 512]
            
            # Separate into h, c, m for each layer
            h0 = hidden_states[:, 0:2, :].permute(1, 0, 2)  # [2, batch_size, 512] for 2 layers
            c0 = hidden_states[:, 2:4, :].permute(1, 0, 2)  # [2, batch_size, 512] for 2 layers
            m0 = hidden_states[:, 4:6, :].permute(1, 0, 2)  # [2, batch_size, 512] for 2 layers
            
            # Create list of tuples (h, c, m) for each layer
            hidden = [(h0[i], c0[i], m0[i]) for i in range(2)]  # 2 layers
        elif self.decoder_type == 'lstm':
            hidden = (enc_features.unsqueeze(0), enc_features.unsqueeze(0))
        else:  # GRU
            hidden = enc_features.unsqueeze(0)

        # Select the correct start token for different tokenization modes
        if self.tokenizer.mode == "char":
            start_token = self.tokenizer.char2idx['<SOS>']
        elif self.tokenizer.mode == "wordpiece":
            start_token = self.tokenizer.wordpiece_tokenizer.cls_token_id
        elif self.tokenizer.mode == "word":
            start_token = self.tokenizer.word2idx['<SOS>']
        else:
            raise ValueError("Unsupported tokenization mode.")

        start = torch.tensor([start_token], dtype=torch.long).to(self.device)
        start_embed = self.embed(start).repeat(batch_size, 1).unsqueeze(0)
        inp = start_embed

        outputs = [inp]
        tokens_list = []
        current_token = start.repeat(batch_size)
        tokens_list.append(current_token)

        # ------------------ Decoder Loop ------------------
        for t in range(self.TEXT_MAX_LEN - 1):
            # Decoder step - no attention here
            out, hidden = self.decoder(inp, hidden)
            
            # Reshape out if necessary (XLSTM returns sequence, batch, features)
            if self.decoder_type == 'xlstm':
                out = out.squeeze(0)  # Remove sequence dimension

            outputs.append(out)

            # Project to vocabulary space for this step
            step_proj = self.proj(out.transpose(0, 1)).transpose(1, 2)  # [batch_size, vocab_size, 1]
            current_token = torch.argmax(F.softmax(step_proj, dim=1), dim=1).squeeze(-1)
            tokens_list.append(current_token)
            
            # Teacher forcing: decide whether to use ground truth or predicted token
            if use_teacher_forcing and t < captions.size(1) - 1:
                # Use target token as next input
                teacher_input = captions[:, t+1]
                inp = self.embed(teacher_input).unsqueeze(0)
            else:
                # Use model's own prediction as next input
                inp = out
        
        # ------------------ Final Projection ------------------
        outputs = torch.cat(outputs, dim=0).permute(1, 0, 2)  # [batch_size, seq, 512]
        res = self.proj(outputs).permute(0, 2, 1)  # [batch_size, vocab_size, seq]
        tokens = torch.stack(tokens_list, dim=1)  # [batch_size, seq_len]

        return res, tokens
    
    def inference(self, img):
        """
        Inference method for validation and testing (no teacher forcing)
        
        Args:
            img: Input images [batch_size, channels, height, width]
            
        Returns:
            res: Output logits [batch_size, vocab_size, seq_len]
            tokens: Predicted tokens [batch_size, seq_len]
        """
        with torch.no_grad():
            return self.forward(img, captions=None, teacher_forcing_ratio=0)


class Model2(nn.Module):
    def __init__(self, device, tokenizer, TEXT_MAX_LEN=151, encoder_name='resnet18', decoder_name='gru'):
        super().__init__()
        self.device = device
        self.TEXT_MAX_LEN = TEXT_MAX_LEN
        self.tokenizer = tokenizer
        vocab_size = self.tokenizer.vocab_size
        

        # Available encoders
        encoder_map = {
            'resnet18': (ResNetModel, 'microsoft/resnet-18', 512),
            'resnet34': (ResNetModel, 'microsoft/resnet-34', 512),
            'resnet50': (ResNetModel, 'microsoft/resnet-50', 2048),
            'vgg16':    (models.vgg16, 512*2*2), # Dim after pooling
            'vgg19':    (models.vgg19, 512*2*2), # Dim after pooling
        }

        if encoder_name not in encoder_map:
            raise ValueError(f"Unknown encoder '{encoder_name}'. Choose from {list(encoder_map.keys())}.")

        if "resnet" in encoder_name:
            model_class, model_str, out_dim = encoder_map[encoder_name]
            self.encoder = model_class.from_pretrained(model_str).to(device)
        else:
            model_fn, out_dim = encoder_map[encoder_name]
            self.encoder = model_fn(pretrained=True).to(device)
            self.encoder.classifier = nn.Identity() # Remove FC layers, keep only feature extractor
            self.encoder.avgpool = nn.AdaptiveAvgPool2d(2) # Reduce from 512x7x7 to 512x2x2

        # Projection layer to unify feature size
        self.feature_proj = nn.Linear(out_dim, 512)
 
        # Choose decoder type experiments amb number of layers??
        self.decoder_type = decoder_name
        if decoder_name == 'gru':
            self.decoder = nn.GRU(512, 512, num_layers=1)
        elif decoder_name == 'lstm':
            self.decoder = nn.LSTM(512, 512, num_layers=1)
        elif decoder_name == 'xlstm':
            # Use our custom XLSTM implementation
            self.decoder = XLSTM(512, 512, num_layers=2, dropout=0.1)
            # Initialize hidden state projection for XLSTM
            self.hidden_init = nn.Linear(512, 512 * 3 * 2)  # For h, c, m across 2 layers
        elif decoder_name == 'bilstm':  # Bidirectional LSTM, he d mirar b com funciona i si té sentit
            self.decoder = nn.LSTM(512, 512, num_layers=1, bidirectional=True)
            # For bidirectional, adjust output dimension for projection
            self.bidir_proj = nn.Linear(1024, 512)
        else:
            raise ValueError(f"Unknown decoder type '{decoder_name}'. Choose from ['gru', 'lstm', 'xlstm', 'bilstm']")
            
        #self.proj = nn.Linear(512, NUM_CHAR)
        #self.embed = nn.Embedding(NUM_CHAR, 512)
        # Embedding and projection layers
        self.embed = nn.Embedding(vocab_size, 512)
        self.proj = nn.Linear(512, vocab_size)

    def forward(self, img, captions=None, teacher_forcing_ratio=0.5):
        batch_size = img.shape[0]
        device = img.device

        # Forward pass through encoder (unchanged)
        if isinstance(self.encoder, ResNetModel):
            feat = self.encoder(img).pooler_output
            feat = feat.view(batch_size, -1)
        else:  # VGG Case
            feat = self.encoder.features(img)
            feat = self.encoder.avgpool(feat)
            feat = feat.view(batch_size, -1)

        # Project features to 512 for decoder
        feat = self.feature_proj(feat)

        # Prepare initial hidden state based on decoder type
        if self.decoder_type == 'xlstm':
            # For XLSTM, transform image features into initial (h, c, m) states for each layer
            hidden_states = self.hidden_init(feat)  # [batch_size, 512*3*2]
            hidden_states = hidden_states.view(batch_size, 6, 512)  # [batch_size, 6, 512]
            
            # Separate into h, c, m for each layer
            h0 = hidden_states[:, 0:2, :].permute(1, 0, 2)  # [2, batch_size, 512] for 2 layers
            c0 = hidden_states[:, 2:4, :].permute(1, 0, 2)  # [2, batch_size, 512] for 2 layers
            m0 = hidden_states[:, 4:6, :].permute(1, 0, 2)  # [2, batch_size, 512] for 2 layers
            
            # Create list of tuples (h, c, m) for each layer
            hidden = [(h0[i], c0[i], m0[i]) for i in range(2)]  # 2 layers

        elif self.decoder_type == 'lstm' or self.decoder_type == 'bilstm':
            # For LSTM and BiLSTM, we need (h, c)
            feat = feat.unsqueeze(0)  # [1, batch_size, 512]
            hidden = (feat, feat.clone())  # Tuple of (hidden_state, cell_state)
            
        else:
            # For GRU, just use h
            feat = feat.unsqueeze(0)  # [1, batch_size, 512]
            hidden = feat

        # Start token embedding (unchanged)
        #start = torch.tensor(char2idx['< SOS >']).to(device)
        #start_token = chars[0]  # This should be '< SOS >'

        # Select the correct start token for different tokenization modes
        if self.tokenizer.mode == "char":
            start_token = self.tokenizer.char2idx['<SOS>']
        elif self.tokenizer.mode == "wordpiece":
            start_token = self.tokenizer.wordpiece_tokenizer.cls_token_id
        elif self.tokenizer.mode == "word":
            start_token = self.tokenizer.word2idx['<SOS>']
        else:
            raise ValueError("Unsupported tokenization mode.")
        
        """start = torch.tensor([start_token], dtype=torch.long).to(self.device)
        start_embed = self.embed(start)
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)
        """
        start = torch.tensor([start_token], dtype=torch.long).to(self.device)
        start_embed = self.embed(start).repeat(batch_size, 1).unsqueeze(0)
        inp = start_embed

        #inp = start_embeds 

        # Rest of the method remains unchanged
        outputs = []
        output_tokens = torch.zeros(batch_size, self.TEXT_MAX_LEN).long().to(device)
        
        # Set the first token to SOS
        output_tokens[:, 0] = start_token

        # Decode sequence with teacher forcing
        use_teacher_forcing = True if (captions is not None and random.random() < teacher_forcing_ratio) else False
        
        # Pass through decoder one step at a time
        for t in range(self.TEXT_MAX_LEN - 1):
            out, hidden = self.decoder(inp, hidden)
            
            # Handle bidirectional output if needed
            if self.decoder_type == 'bilstm':
                out = self.bidir_proj(out)
            
            projection = self.proj(out.squeeze(0))
            outputs.append(projection.unsqueeze(1))
            
            top_char = projection.argmax(dim=1)
            output_tokens[:, t+1] = top_char
            
            if use_teacher_forcing and t < captions.size(1) - 1:
                char_embed = self.embed(captions[:, t+1])
            else:
                char_embed = self.embed(top_char)
            
            inp = char_embed.unsqueeze(0)
        
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.permute(0, 2, 1)
        
        return outputs, output_tokens
    
    def inference(self, img):
        """
        Inference mode (no teacher forcing)
        """
        return self.forward(img, captions=None, teacher_forcing_ratio=0.0)

