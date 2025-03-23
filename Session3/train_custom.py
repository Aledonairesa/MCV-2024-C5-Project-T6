import os
import pandas as pd
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
import evaluate

from utils.visualization import *
from utils.utils import *
from utils.models import *
from utils.configuration import *


# Character vocabulary and mapping
chars = ['<SOS>', '<EOS>', '<PAD>', '<UNK>', ' ', '!', '"', '#', '&', "'", '(', ')', ',', '-', '.', 
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?', 'A', 'B', 'C', 'D', 
         'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 
         'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CHAR = len(chars)
char2idx = {char: idx for idx, char in enumerate(chars)}
idx2char = {idx: char for idx, char in enumerate(chars)}

# Initialize tokenizer
tokenizer = Tokenizer(tokenization_mode=TOKENIZATION_MODE)

# Define paths
train_split_path = os.path.join(ROOT_DIR, "clean_mapping_train.csv")
validation_split_path = os.path.join(ROOT_DIR, "clean_mapping_validation.csv")
test_split_path = os.path.join(ROOT_DIR, "clean_mapping_test.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create the FIRD Datasets
print("Loading Dataset")

train_dataset = FirdDataset(csv_file=train_split_path, root_dir=ROOT_DIR, max_len=TEXT_MAX_LEN, tokenizer=tokenizer)
validation_dataset = FirdDataset(csv_file=validation_split_path, root_dir=ROOT_DIR, max_len=TEXT_MAX_LEN, tokenizer=tokenizer)
#test_dataset = FirdDataset(csv_file=test_split_path, root_dir=ROOT_DIR, max_len=TEXT_MAX_LEN, tokenizer=tokenizer)

# Create the DataLoaders for batching and shuffling
print("Creating Dataloaders")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
#test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Set up device
print("Seting up device and loading model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate model, loss function, optimizer, and LR decay
encoder = "resnet18"
decoder = "lstm"
vocab_size = tokenizer.get_vocab_size()
model = Model2(device, tokenizer, encoder_name=encoder, decoder_name=decoder).to(device)
criterion = nn.CrossEntropyLoss()
optimizer_name = "Adam"
learning_rate = 1e-4
lr_decay = True

# Configure teacher forcing
teacher_forcing_ratio = 0.8
teacher_forcing_decay = 0.95  # Decay factor per epoch

# Fix optimizer and LR decay based on chosen parameters
if optimizer_name == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_name == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
elif optimizer_name == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
elif optimizer_name == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
else:
    raise ValueError(f"Unknown optimizer '{optimizer_name}'")

if lr_decay:
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# Print model info
print(f"Using {encoder} encoder")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")
print(f"Initial teacher forcing ratio: {teacher_forcing_ratio:.2f}")

num_epochs = 15
patience = 3

best_val_loss = float('inf')
early_stop_counter = 0

train_losses = []
val_losses = []

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")

scores = {
    "bleu_1": [],
    "bleu_2": [],
    "rouge_l": [],
    "meteor": []
}

print("Starting training")
for epoch in range(num_epochs):
    # --------------------------------
    # TRAINING STEP
    # --------------------------------
    model.train()
    running_train_loss = 0.0

    print(f"Epoch {epoch+1}/{num_epochs} - Teacher forcing ratio: {teacher_forcing_ratio:.2f}")

    for images, captions, idxs in train_dataloader:
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        outputs, tokens = model(images, captions=captions, teacher_forcing_ratio=teacher_forcing_ratio)

        # Calculate CrossEntropyLoss
        loss = criterion(outputs, captions[:, 1:])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_dataloader)
    train_losses.append(epoch_train_loss)

    # Decay teacher forcing ratio each epoch
    teacher_forcing_ratio *= teacher_forcing_decay

    # --------------------------------
    # VALIDATION STEP
    # --------------------------------
    model.eval()
    running_val_loss = 0.0
    all_predictions = []
    all_references = []
    with torch.no_grad():
        for batch_idx, (images, captions, idxs) in enumerate(validation_dataloader):
            images = images.to(device)
            captions = captions.to(device)

            outputs, tokens =  model.inference(images)
            
            if batch_idx % 50 == 0:
                visualize_results(images, tokens, idxs, validation_dataset, tokenizer, epoch, OUTPUT_DIR)
                
            loss = criterion(outputs, captions[:, 1:])
            running_val_loss += loss.item()

            # Accumulate predictions and corresponding ground truth captions
            for pred, idx in zip(tokens, idxs):
                pred_str = detokenize_caption(pred, tokenizer)
                ref_str = validation_dataset.data.iloc[int(idx)]['Title']
                all_predictions.append(pred_str)
                all_references.append([ref_str]) # Wrap reference in list (required by evaluate library)

    epoch_val_loss = running_val_loss / len(validation_dataloader)
    val_losses.append(epoch_val_loss)

    # ---------------------------
    # COMPUTE METRICS ON VALIDATION SET
    # ---------------------------
    bleu1_result = bleu_metric.compute(predictions=all_predictions, references=all_references, max_order=1)
    bleu2_result = bleu_metric.compute(predictions=all_predictions, references=all_references, max_order=2)
    rouge_result = rouge_metric.compute(predictions=all_predictions, references=all_references)
    meteor_result = meteor_metric.compute(predictions=all_predictions, references=all_references)
    
    epoch_bleu1 = bleu1_result['bleu'] * 100
    epoch_bleu2 = bleu2_result['bleu'] * 100
    epoch_rouge_l = rouge_result['rougeL'] * 100
    epoch_meteor = meteor_result['meteor'] * 100
    
    scores["bleu_1"].append(epoch_bleu1)
    scores["bleu_2"].append(epoch_bleu2)
    scores["rouge_l"].append(epoch_rouge_l)
    scores["meteor"].append(epoch_meteor)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_train_loss:.4f} "
          f"Val Loss: {epoch_val_loss:.4f} "
          f"BLEU-1: {epoch_bleu1:.1f}%, "
          f"BLEU-2: {epoch_bleu2:.1f}%, "
          f"ROUGE-L: {epoch_rouge_l:.1f}%, "
          f"METEOR: {epoch_meteor:.1f}%")

    # --------------------------------
    # CHECK FOR BEST MODEL & EARLY STOPPING
    # --------------------------------
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        early_stop_counter = 0  # Reset counter if improvement
        print(f"New best model found! Saving model with Val Loss: {epoch_val_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))  # Save best model
    else:
        early_stop_counter += 1
        print(f"No improvement ({early_stop_counter}/{patience})")
    
    # --------------------------------
    # LR DECAY
    # --------------------------------
    if lr_decay:
        scheduler.step(epoch_val_loss)  # Reduce LR if no improvement
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr:.7f}")

    # Stop if no improvement for 'patience' epochs
    if early_stop_counter >= patience:
        print(f"Early stopping triggered after {patience} epochs without improvement!")
        break  # Exit training loop

# --------------------------------
# PLOT THE LOSSES
# --------------------------------
loss_plot(train_losses, val_losses, OUTPUT_DIR)

# ---------------------------
# PLOT METRIC EVOLUTION
# ---------------------------
scores_plot(scores, OUTPUT_DIR)

# ---------------------------
# SAVE FINAL METRICS TO CSV
# ---------------------------
metrics_csv_path = save_metrics_train(train_losses, val_losses, scores, OUTPUT_DIR)
print(f"Final metrics saved to {metrics_csv_path}")