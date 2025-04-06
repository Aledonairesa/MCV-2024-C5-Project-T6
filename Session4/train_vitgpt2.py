import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    VisionEncoderDecoderModel, 
    AutoImageProcessor, 
    AutoTokenizer,
    AdamW, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import evaluate
import torchvision.transforms as transforms
from PIL import Image

from dataloader import VitGPT2Dataset 

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

def evaluate_metrics(predictions, references):
    bleu1 = bleu.compute(predictions=predictions, references=references, max_order=1)["bleu"]
    bleu2 = bleu.compute(predictions=predictions, references=references, max_order=2)["bleu"]
    rouge_l = rouge.compute(predictions=predictions, references=references)["rougeL"]
    meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
    
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'METEOR': meteor_score,
        'ROUGE-L': rouge_l
    }

def save_sample_images(dataset, predictions, references, output_dir, processor, num_samples=5):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(num_samples, len(dataset))):
        image_path = dataset[i]['image_path']
        image = Image.open(image_path).convert("RGB")
        
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"GT: {references[i]}\nPred: {predictions[i]}")
        plt.savefig(os.path.join(output_dir, f"sample_{i}.png"))
        plt.close()

def plot_metrics(train_losses, val_losses, metrics_history, output_dir):
    epochs = range(1, len(train_losses) + 1)
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot train and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()
    
    # Plot metrics
    plt.figure(figsize=(10, 5))
    for metric in metrics_history[0].keys():
        plt.plot(epochs, [m[metric] for m in metrics_history], label=metric)
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Evaluation Metrics')
    plt.savefig(os.path.join(output_dir, "metrics_curve.png"))
    plt.close()

def train(
    dataset_train,
    dataset_val,
    model_name='nlpconnect/vit-gpt2-image-captioning',
    output_dir='models',
    freeze_encoder=False,
    freeze_decoder=False,
    batch_size=16,
    epochs=25,
    lr=5e-5,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    os.makedirs(output_dir, exist_ok=True)
    
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
    if freeze_decoder:
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, collate_fn=dataset_val.collate_fn
    )
    
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = []
    print(f"==============\nTraining parameters:\nepochs: {epochs}\nbatch size: {batch_size}\nlearning rate: {lr}\nfreeze encoder: {freeze_encoder}\nfreeze decoder: {freeze_decoder}\n==============")    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            pixel_values, labels = batch['pixel_values'].to(device), batch['input_ids'].to(device)
            loss = model(pixel_values=pixel_values, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        predictions, references = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                pixel_values, labels = batch['pixel_values'].to(device), batch['input_ids'].to(device)
                outputs = model(pixel_values=pixel_values, labels=labels)
                val_loss += outputs.loss.item()
                
                pred_tokens = model.generate(pixel_values)
                preds = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
                refs = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                predictions.extend(preds)
                references.extend(refs)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch}: Val Loss = {avg_val_loss:.4f}")
        metrics = evaluate_metrics(predictions, references)
        metrics_history.append(metrics)
        print(f"Metrics: {metrics}")
        
        save_sample_images(dataset_val, predictions, references, os.path.join(output_dir, "samples", "epoch"+str(epoch)), processor)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.encoder.save_pretrained(os.path.join(output_dir, "best_model_encoder"))
            model.decoder.save_pretrained(os.path.join(output_dir, "best_model_decoder"))
            model.save_pretrained(os.path.join(output_dir, "best_model"))
            processor.save_pretrained(os.path.join(output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))
    
    plot_metrics(train_losses, val_losses, metrics_history, output_dir)
    model.encoder.save_pretrained(os.path.join(output_dir, "final_model_encoder"))
    model.decoder.save_pretrained(os.path.join(output_dir, "final_model_decoder"))
    model.save_pretrained(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

def main():
    csv_train = '/ghome/c5mcv06/FIRD/clean_mapping_train.csv'
    csv_val = '/ghome/c5mcv06/FIRD/clean_mapping_validation.csv'
    root_dir = '/ghome/c5mcv06/FIRD'
    
    train_data = VitGPT2Dataset(csv_file=csv_train, root_dir=root_dir, max_len=150)
    val_data = VitGPT2Dataset(csv_file=csv_val, root_dir=root_dir, max_len=150)
    
    train(dataset_train=train_data, dataset_val=val_data, output_dir='./models/encoder_only', freeze_decoder=True)
    train(dataset_train=train_data, dataset_val=val_data, output_dir='./models/decoder_only', freeze_encoder=True)
    train(dataset_train=train_data, dataset_val=val_data, output_dir='./models/both')

if __name__ == "__main__":
    main()
