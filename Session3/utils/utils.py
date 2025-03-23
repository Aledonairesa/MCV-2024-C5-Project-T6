import pandas as pd
import os
import torch
from transformers import AutoTokenizer
from collections import Counter
import re as re
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#-----------------------
#    METRIC SAVING
#-----------------------
def save_metrics_train(train_losses, val_losses, scores, output_dir, file_name="final_metrics.csv"):
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses) + 1)),
        "Train Loss": train_losses,
        "Validation Loss": val_losses,
        "BLEU-1 (%)": scores["bleu_1"],
        "BLEU-2 (%)":  scores["bleu_2"],
        "ROUGE-L (%)":  scores["rouge_l"],
        "METEOR (%)":  scores["meteor"]
    })
    metrics_csv_path = os.path.join(output_dir, file_name)
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_csv_path

def save_metrics_test(test_losses, times, scores, output_dir, file_name="final_metrics.csv"):
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(test_losses) + 1)),
        "Test Loss": test_losses,
        "Time per epoch": times,
        "BLEU-1 (%)": scores["bleu_1"],
        "BLEU-2 (%)":  scores["bleu_2"],
        "ROUGE-L (%)":  scores["rouge_l"],
        "METEOR (%)":  scores["meteor"]
    })
    metrics_csv_path = os.path.join(output_dir, file_name)
    metrics_df.to_csv(metrics_csv_path, index=False)

    return metrics_csv_path



#-----------------------
#    TOKENIZER
#-----------------------

class Tokenizer:
    def __init__(self, tokenization_mode="char", vocab_size=None):
        """
        Initialize tokenizer based on specified mode.
        
        Args:
            tokenization_mode (str): One of 'char', 'wordpiece', or 'word'
            vocab_size (int, optional): Max vocabulary size for word-level tokenization
        """
        self.mode = tokenization_mode
        self.vocab_size = vocab_size

        # For character-level tokenization
        if self.mode == "char":        
            # Character vocabulary and mapping (for char-level)
            self.chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', 
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 
                    'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
                    'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
                    'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
            self.char2idx = {char: idx for idx, char in enumerate(self.chars)}
            self.idx2char = {idx: char for idx, char in enumerate(self.chars)}
            self.num_char = len(self.chars)
        
        # For wordpiece tokenization
        elif self.mode == "wordpiece":
            self.wordpiece_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            # Add special tokens to the wordpiece tokenizer
            special_tokens = {'eos_token': '<EOS>', 'pad_token': '<PAD>', 'unk_token': '<UNK>'}
            self.wordpiece_tokenizer.add_special_tokens(special_tokens)
            self.vocab_size = len(self.wordpiece_tokenizer)
        
        # For word-level tokenization
        elif self.mode == "word":
            self.word2idx = None
            self.idx2word = None

        else:
            raise ValueError(f"Unknown tokenization mode: {self.mode}")
            
    def build_vocab(self, captions):
        """
        Build vocabulary for word-level tokenization.
        
        Args:
            captions (list): List of caption strings
        """
        if self.mode != "word":
            return
            
        # Tokenize captions into words
        word_counts = Counter()
        for caption in captions:
            # Simple word tokenization by splitting on whitespace and punctuation
            words = re.findall(r'\b\w+\b', caption.lower())
            word_counts.update(words)
            
        # Select top N words based on frequency
        if self.vocab_size is None:
            self.vocab_size = 4000
            
        # Keep top words plus special tokens
        common_words = [word for word, _ in word_counts.most_common(self.vocab_size - 4)]
        
        # Create word to index mapping
        self.word2idx = {
            '<SOS>': 0,
            '<EOS>': 1,
            '<PAD>': 2,
            '<UNK>': 3
        }
        
        # Add common words to vocabulary
        for idx, word in enumerate(common_words, 4):
            self.word2idx[word] = idx
            
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def tokenize(self, text, max_len):
        """
        Tokenize text based on the selected mode.
        
        Args:
            text (str): Text to tokenize
            max_len (int): Maximum sequence length (including SOS/EOS)
            
        Returns:
            torch.Tensor: Tensor of token indices
        """
        if self.mode == "char":
            # Character-level tokenization
            tokens = ['<SOS>'] + list(text) + ['<EOS>']
            tokens = [token if token in self.char2idx else '<UNK>' for token in tokens]
            
            # Truncate or pad as needed
            if len(tokens) < max_len:
                tokens.extend(['<PAD>'] * (max_len - len(tokens)))
            else:
                tokens = tokens[:max_len]
                
            # Convert to indices
            indices = [self.char2idx[token] for token in tokens]
            
        elif self.mode == "wordpiece":
            # Wordpiece tokenization using transformers
            encoding = self.wordpiece_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=max_len - 1,  # -1 to account for SOS token
                return_tensors='pt'
            )
            
            # Add SOS token at the beginning
            sos_token_id = self.wordpiece_tokenizer.cls_token_id
            indices = [sos_token_id] + encoding.input_ids[0].tolist()
            
            # Ensure we don't exceed max_len
            indices = indices[:max_len]
            
        elif self.mode == "word":
            # Word-level tokenization
            if self.word2idx is None:
                raise ValueError("Vocabulary not built. Call build_vocab first.")
                
            # Simple word tokenization
            words = re.findall(r'\b\w+\b', text.lower())
            tokens = ['<SOS>'] + words + ['<EOS>']
            
            # Replace OOV words with UNK
            tokens = [token if token in self.word2idx else '<UNK>' for token in tokens]
            
            # Truncate or pad as needed
            if len(tokens) < max_len:
                tokens.extend(['<PAD>'] * (max_len - len(tokens)))
            else:
                tokens = tokens[:max_len]
                
            # Convert to indices
            indices = [self.word2idx[token] for token in tokens]
            
        return torch.tensor(indices, dtype=torch.long)
    
    def detokenize(self, indices):
        """
        Convert token indices back to text.
        
        Args:
            indices: List or tensor of token indices
            
        Returns:
            str: Detokenized text
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        if self.mode == "char":
            # Filter out special tokens
            filtered_tokens = [self.idx2char[idx] for idx in indices 
                             if idx in self.idx2char and self.idx2char[idx] not in ['<SOS>', '<EOS>', '<PAD>']]
            # Join characters
            return ''.join(filtered_tokens)
            
        elif self.mode == "wordpiece":
            # Convert indices to tokens using the wordpiece tokenizer
            # Skip SOS token (first token)
            tokens = indices[1:]
            # Remove padding, EOS, etc.
            text = self.wordpiece_tokenizer.decode(tokens, skip_special_tokens=True)
            return text
            
        elif self.mode == "word":
            # Filter out special tokens
            filtered_tokens = [self.idx2word[idx] for idx in indices 
                             if idx in self.idx2word and self.idx2word[idx] not in ['<SOS>', '<EOS>', '<PAD>']]
            # Join words with spaces
            return ' '.join(filtered_tokens)
        
    def get_vocab_size(self):
        """
        Get the vocabulary size based on the tokenization mode.
        
        Returns:
            int: Size of the vocabulary
        """
        if self.mode == "char":
            return self.num_char
        elif self.mode == "wordpiece":
            return self.vocab_size
        elif self.mode == "word":
            return self.vocab_size
        else:
            raise ValueError(f"Unknown tokenization mode: {self.mode}")


def detokenize_caption(caption_tensor, tokenizer):
    """
    Convert a tensor of tokenized indices back to a string caption.
    
    Args:
    - caption_tensor (torch.Tensor): Tensor of token indices.
    - tokenizer (Tokenizer): Tokenizer instance.
    
    Returns:
    - str: The detokenized caption.
    """
    return tokenizer.detokenize(caption_tensor)


#-----------------------
#    DATSET LOADING
#-----------------------


class FirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, tokenizer, max_len=150+1, indices=None, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations (e.g., 'FIRD/clean_mapping.csv').
            root_dir (string): Base directory containing the folder 'food_images'.
            tokenizer (Tokenizer): Tokenizer instance for processing captions.
            max_len (int): Maximum length of the processed caption (includes <SOS> and <EOS>).
            indices (list, optional): List of indices to select a subset of the data.
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.data = pd.read_csv(csv_file)
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)
        self.root_dir = root_dir
        self.max_len = max_len
        self.tokenizer = tokenizer
        
        # Build vocabulary for word-level tokenization if needed
        if tokenizer.mode == "word" and tokenizer.word2idx is None:
            tokenizer.build_vocab(self.data['Title'].tolist())
        
        # Default transformation pipeline if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row corresponding to the current index
        row = self.data.iloc[idx]
        image_name = row['Image_Name'] + ".jpg"
        caption_text = row['Title']
        
        # Build the full image path and load the image
        img_path = os.path.join(self.root_dir, "food_images", image_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize the caption using the provided tokenizer
        caption_tensor = self.tokenizer.tokenize(caption_text, self.max_len)
        
        return image, caption_tensor, idx