import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForCausalLM, ViTModel
from peft import PeftModel
import evaluate
import matplotlib.pyplot as plt
import argparse

# -------------------------------
# Helper: Save a Batch of Images & Captions
# -------------------------------
def save_batch_visuals(images, gt_captions, pred_captions, out_dir, batch_idx):
    """
    Saves each image in the batch using matplotlib, embedding ground truth and predicted captions below each image.

    Args:
        images (Tensor): shape [B, 3, H, W] - batch of images.
        gt_captions (list[str]): ground truth captions (length = B).
        pred_captions (list[str]): predicted captions (length = B).
        out_dir (str): output directory for saving.
        batch_idx (int): current batch index.
    """
    os.makedirs(out_dir, exist_ok=True)

    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    batch_size = images.size(0)

    for i in range(batch_size):
        img_i = inv_normalize(images[i])
        img_i = torch.clamp(img_i, 0.0, 1.0).cpu().permute(1, 2, 0).numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_i)
        ax.axis('off')

        gt_text = f"True: {gt_captions[i]}"
        pred_text = f"Pred: {pred_captions[i]}"

        plt.figtext(0.5, 0.01, f"{gt_text}\n{pred_text}", wrap=True, horizontalalignment='center', fontsize=12)

        img_filename = f"batch_{batch_idx}_sample_{i}.png"
        plt.savefig(os.path.join(out_dir, img_filename), bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

# -------------------------------
# 1. Custom Dataset for Test Data
# -------------------------------
class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file (e.g., 'clean_mapping_test.csv').
            root_dir (str): Base directory containing the 'food_images' folder.
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir

        # Default transformation pipeline (same as during training)
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
        row = self.data.iloc[idx]
        image_name = row['Image_Name'] + ".jpg"
        caption = row['Title']

        img_path = os.path.join(self.root_dir, "food_images", image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, caption

# -------------------------------
# 2. Evaluation Script
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA-fine-tuned ViT-Llama model on a test set.")
    parser.add_argument("--test_csv", type=str, default="/ghome/c5mcv06/FIRD/clean_mapping_test.csv",
                        help="Path to the test CSV file.")
    parser.add_argument("--root_dir", type=str, default="/ghome/c5mcv06/FIRD",
                        help="Directory containing the 'food_images' folder.")
    parser.add_argument("--save_dir", type=str, default="./llama_finetuned_lora",
                        help="Directory containing the fine-tuned LoRA adapter and projector.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for evaluation.")
    parser.add_argument("--max_text_len", type=int, default=64,
                        help="Max new tokens to generate.")
    parser.add_argument("--save_visuals_interval", type=int, default=50,
                        help="Save visualization every N batches.")
    parser.add_argument("--llama_model_name", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Name or path of the base Llama model.")
    parser.add_argument("--vit_model_name", type=str, default="google/vit-base-patch16-224",
                        help="Name or path of the base ViT model.")
    args = parser.parse_args()

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test dataset and dataloader
    test_dataset = TestDataset(csv_file=args.test_csv, root_dir=args.root_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Visuals will be saved under save_dir/visual_samples
    visual_samples_dir = os.path.join(args.save_dir, "visual_samples")
    os.makedirs(visual_samples_dir, exist_ok=True)

    # Load tokenizer (same one as used for training)
    tokenizer = AutoTokenizer.from_pretrained(args.llama_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the pre-trained ViT model
    vit_model = ViTModel.from_pretrained(args.vit_model_name)
    vit_model.to(device)
    vit_model.eval()  # freeze ViT

    # Load the frozen base Llama model
    llama_model = AutoModelForCausalLM.from_pretrained(args.llama_model_name)
    for param in llama_model.parameters():
        param.requires_grad = False
    llama_model.to(device)

    # Load the LoRA adapter from the saved directory, merging with the Llama model
    lora_model = PeftModel.from_pretrained(llama_model, args.save_dir)
    lora_model.to(device)
    lora_model.eval()

    # Dynamically get the hidden sizes
    vit_hidden_size = vit_model.config.hidden_size
    llama_hidden_size = llama_model.config.hidden_size

    # Create and load the image projector
    image_projector = nn.Linear(vit_hidden_size, llama_hidden_size, bias=False).to(device)
    projector_path = os.path.join(args.save_dir, "image_projector.pt")
    image_projector.load_state_dict(torch.load(projector_path, map_location=device))
    image_projector.eval()

    # Initialize metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    all_predictions = []
    all_references = []

    print("Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(test_loader):
            images = images.to(device)

            # 1. Extract image features
            vit_outputs = vit_model(pixel_values=images)
            vit_feats = vit_outputs.last_hidden_state  # [batch_size, vit_seq_len, vit_hidden_size]

            # 2. Project ViT features to Llama hidden space
            image_embeds = image_projector(vit_feats)  # [batch_size, vit_seq_len, llama_hidden_size]

            # 3. Generate captions using LoRA model
            generated_ids = lora_model.generate(
                inputs_embeds=image_embeds,
                attention_mask=torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device),
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=args.max_text_len,
                do_sample=False
            )

            generated_texts = [tokenizer.decode(g, skip_special_tokens=True).strip() for g in generated_ids]

            all_predictions.extend(generated_texts)
            all_references.extend(captions)

            # Save a batch of visuals every N batches
            if batch_idx % args.save_visuals_interval == 0:
                save_batch_visuals(
                    images,
                    captions,
                    generated_texts,
                    out_dir=visual_samples_dir,
                    batch_idx=batch_idx
                )

    # Compute metrics
    bleu1_result = bleu_metric.compute(predictions=all_predictions, references=all_references, max_order=1)
    bleu2_result = bleu_metric.compute(predictions=all_predictions, references=all_references, max_order=2)
    rouge_result = rouge_metric.compute(predictions=all_predictions, references=all_references)
    meteor_result = meteor_metric.compute(predictions=all_predictions, references=all_references)

    epoch_bleu1 = bleu1_result['bleu'] * 100
    epoch_bleu2 = bleu2_result['bleu'] * 100
    epoch_rouge_l = rouge_result['rougeL'] * 100
    epoch_meteor = meteor_result['meteor'] * 100

    print("\nEvaluation Results:")
    print(f"BLEU-1:   {epoch_bleu1:.2f}")
    print(f"BLEU-2:   {epoch_bleu2:.2f}")
    print(f"ROUGE-L:  {epoch_rouge_l:.2f}")
    print(f"METEOR:   {epoch_meteor:.2f}")

    # Save metrics to metrics.txt inside save_dir
    metrics_file = os.path.join(args.save_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"BLEU-1:   {epoch_bleu1:.2f}\n")
        f.write(f"BLEU-2:   {epoch_bleu2:.2f}\n")
        f.write(f"ROUGE-L:  {epoch_rouge_l:.2f}\n")
        f.write(f"METEOR:   {epoch_meteor:.2f}\n")

    print(f"Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
