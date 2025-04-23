import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    ViTModel,
)
from peft import LoraConfig, get_peft_model
import argparse
import datetime

# -------------------------------
# 1. Custom Dataset using Llama Tokenizer
# -------------------------------
class FirdDataset(Dataset):
    def __init__(self, csv_file, root_dir, food_images_dir, tokenizer, max_len=64, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations (e.g., 'FIRD/clean_mapping_train.csv').
            root_dir (string): Base directory containing the folder 'food_images'.
            tokenizer: Pretrained Llama tokenizer.
            max_len (int): Maximum sequence length for tokenized captions.
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.food_images_dir = food_images_dir
        self.tokenizer = tokenizer
        self.max_len = max_len

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
        # Read CSV row
        row = self.data.iloc[idx]
        image_name_base = row['Image_Name']

        # Choose file extension based on substring match
        extension = ".png" if "prompt" in image_name_base else ".jpg"
        image_name = image_name_base + extension

        caption_text = row['Title']

        # Load image from the "food_images" folder inside root_dir
        img_path = os.path.join(self.root_dir, self.food_images_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Tokenize the caption using the Llama tokenizer.
        tokenized = self.tokenizer(
            caption_text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        # Squeeze out the extra batch dimension.
        caption_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return image, caption_ids, attention_mask

# -------------------------------
# 2. Main Fine-Tuning Script
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune a ViT-Llama model on a custom dataset.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA low-rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj", help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--vit_model_path", type=str, default="google/vit-base-patch16-224", help="Path or name of the ViT model")
    parser.add_argument("--llama_model_path", type=str, default="meta-llama/Llama-3.2-3B", help="Path or name of the Llama model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"], help="Optimizer to use: adamw or adam")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_text_len", type=int, default=64, help="Max token length for captions")
    parser.add_argument("--root_dir", type=str, default="/ghome/c5mcv06/FIRD", help="Root directory containing the dataset")
    parser.add_argument("--train_csv", type=str, default="clean_mapping_train.csv", help="Relative path to training CSV file from root_dir")
    parser.add_argument("--validation_csv", type=str, default="clean_mapping_validation.csv", help="Relative path to validation CSV file from root_dir")
    parser.add_argument("--food_images_dir", type=str, default="food_images", help="Relative path to food images directory from root_dir")
    args = parser.parse_args()

    # Parse LoRA target modules from comma-separated string to list
    lora_target_modules = [mod.strip() for mod in args.lora_target_modules.split(",")]

    # -------- Configuration --------
    print("Setting configuration values")
    root_dir = args.root_dir
    train_csv = os.path.join(root_dir, args.train_csv)
    validation_csv = os.path.join(root_dir, args.validation_csv)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr
    max_text_len = args.max_text_len
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the Llama tokenizer")
    # -------- Load the Llama Tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token  # llama does not have default padding token defined

    print("Creating the Dataset & DataLoader")
    # -------- Create Dataset & DataLoader --------
    train_dataset = FirdDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        food_images_dir=args.food_images_dir,
        tokenizer=tokenizer,
        max_len=max_text_len
    )
    validation_dataset = FirdDataset(
        csv_file=validation_csv,
        root_dir=root_dir,
        food_images_dir=args.food_images_dir,
        tokenizer=tokenizer,
        max_len=max_text_len
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Loading the pre-trained ViT")
    # -------- Load the Pretrained ViT --------
    vit_model = ViTModel.from_pretrained(args.vit_model_path)
    vit_model.to(device)
    vit_model.eval()  # freeze ViT parameters
    for param in vit_model.parameters():
        param.requires_grad = False

    print("Loading the pre-trained Llama Model")
    # -------- Load the Pretrained Llama Model --------
    llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model_path)
    llama_model.to(device)
    # Freeze Llama base weights
    for param in llama_model.parameters():
        param.requires_grad = False

    print("Applying LoRA to Llama")
    # -------- Apply LoRA to Llama --------
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=lora_target_modules
    )
    lora_model = get_peft_model(llama_model, lora_config)
    lora_model.to(device)

    print("Creating a projection layer")
    # -------- Trainable Projection Layer --------
    # Programmatically obtain the hidden sizes
    vit_hidden_size = vit_model.config.hidden_size
    llama_hidden_size = llama_model.config.hidden_size
    image_projector = nn.Linear(vit_hidden_size, llama_hidden_size, bias=False).to(device)

    # Print informative details about parameters.
    total_vit_params = sum(p.numel() for p in vit_model.parameters())
    total_llama_params = sum(p.numel() for p in llama_model.parameters())
    lora_trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    projector_params = sum(p.numel() for p in image_projector.parameters() if p.requires_grad)
    total_trainable = lora_trainable_params + projector_params
    print(f"ViT Model Parameters: {total_vit_params}")
    print(f"Llama Model Parameters: {total_llama_params}")
    print(f"LoRA Trainable Parameters: {lora_trainable_params}")
    print(f"Image Projector Trainable Parameters: {projector_params}")
    print(f"Total Trainable Parameters (LoRA + Projector): {total_trainable}")

    print("Setting up the optimizer")
    # -------- Setup Optimizer --------
    trainable_params = list(lora_model.parameters()) + list(image_projector.parameters())
    if args.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    else:  # args.optimizer.lower() == "adam"
        optimizer = optim.Adam(trainable_params, lr=learning_rate)

    print("Starting training")
    train_losses = []
    val_losses = []

    # Track best validation loss
    best_val_loss = float("inf")
    best_epoch = -1

    lora_model.train()
    image_projector.train()

    # Prepare directory for saving final and best checkpoints
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./llama_finetuned_lora_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    best_ckpt_dir = os.path.join(save_dir, "best_checkpoint")
    os.makedirs(best_ckpt_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # --- Training ---
        epoch_loss = 0.0
        for batch in train_loader:
            images, caption_ids, caption_attention = batch
            images = images.to(device)
            caption_ids = caption_ids.to(device)
            caption_attention = caption_attention.to(device)

            optimizer.zero_grad()

            # Step 1: Extract image features with ViT
            with torch.no_grad():
                vit_outputs = vit_model(pixel_values=images)
                vit_feats = vit_outputs.last_hidden_state

            # Step 2: Project ViT features to Llama hidden space
            image_embeds = image_projector(vit_feats)

            # Step 3: Get text embeddings from Llama's embedding layer
            embedding_layer = lora_model.get_input_embeddings()
            text_embeds = embedding_layer(caption_ids)

            # Step 4: Concatenate image and text embeddings
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

            # Step 5: Create attention mask for the combined sequence
            batch_size_cur = images.size(0)
            img_seq_len = image_embeds.size(1)
            image_attention = torch.ones((batch_size_cur, img_seq_len),
                                         dtype=caption_attention.dtype,
                                         device=device)
            combined_attention_mask = torch.cat([image_attention, caption_attention], dim=1)

            # Step 6: Create labels for language modeling loss
            labels = torch.cat(
                [torch.full((batch_size_cur, img_seq_len), -100,
                            dtype=caption_ids.dtype, device=device),
                 caption_ids],
                dim=1
            )

            # Step 7: Forward pass
            outputs = lora_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                labels=labels
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation ---
        lora_model.eval()
        image_projector.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, caption_ids, caption_attention = batch
                images = images.to(device)
                caption_ids = caption_ids.to(device)
                caption_attention = caption_attention.to(device)

                # Extract image features
                vit_outputs = vit_model(pixel_values=images)
                vit_feats = vit_outputs.last_hidden_state

                # Project ViT features
                image_embeds = image_projector(vit_feats)

                # Get text embeddings
                embedding_layer = lora_model.get_input_embeddings()
                text_embeds = embedding_layer(caption_ids)

                # Concatenate
                combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)

                # Attention mask
                batch_size_cur = images.size(0)
                img_seq_len = image_embeds.size(1)
                image_attention = torch.ones((batch_size_cur, img_seq_len),
                                             dtype=caption_attention.dtype,
                                             device=device)
                combined_attention_mask = torch.cat([image_attention, caption_attention], dim=1)

                # Labels
                labels = torch.cat(
                    [torch.full((batch_size_cur, img_seq_len), -100,
                                dtype=caption_ids.dtype, device=device),
                     caption_ids],
                    dim=1
                )

                # Forward pass
                outputs = lora_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Check if this is the best model so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            # Save current best model
            print(f"New best model found at epoch {epoch+1}, saving to {best_ckpt_dir} ...")
            lora_model.save_pretrained(best_ckpt_dir)
            torch.save(image_projector.state_dict(), os.path.join(best_ckpt_dir, "image_projector.pt"))

        # Switch back to train mode
        lora_model.train()
        image_projector.train()

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # -------- Final Saving --------
    print(f"Training complete. Best model was from epoch {best_epoch+1} "
          f"with validation loss = {best_val_loss:.4f}.")
    
    # Save the final model (if you want both final and best)
    lora_model.save_pretrained(save_dir)
    torch.save(image_projector.state_dict(), os.path.join(save_dir, "image_projector.pt"))
    print("Final model saved.")

    # Save the arguments used
    args_file = os.path.join(save_dir, "args.txt")
    with open(args_file, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Save params info
    params_file = os.path.join(save_dir, "params.txt")
    with open(params_file, "w") as f:
        f.write(f"ViT Model Parameters: {total_vit_params}\n")
        f.write(f"Llama Model Parameters: {total_llama_params}\n")
        f.write(f"LoRA Trainable Parameters: {lora_trainable_params}\n")
        f.write(f"Image Projector Trainable Parameters: {projector_params}\n")
        f.write(f"Total Trainable Parameters (LoRA + Projector): {total_trainable}\n")

    # Plot the losses
    epochs_range = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Evolution')
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

if __name__ == "__main__":
    main()
