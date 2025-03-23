import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
from .utils import detokenize_caption

def visualize_results(images, predicted_captions, idxs, dataset, tokenizer, epoch, output_dir, starting_epoch = 0, phase="val", extra_info=""):
    if epoch >= starting_epoch:
        print(f"Saving {phase} images")

        image_saving_dir = os.path.join(output_dir ,f"{phase}_images_{extra_info}")
        if not os.path.exists(image_saving_dir):
            os.makedirs(image_saving_dir)

        image_saving_dir = os.path.join(image_saving_dir,f"epoch_{epoch}")
        if not os.path.exists(image_saving_dir):
            os.makedirs(image_saving_dir)

        for image, prediction, idx in zip(images, predicted_captions, idxs):
            row = dataset.data.iloc[int(idx)]
            image_name = row['Image_Name']
            caption_text = row['Title']
            caption_prediction = detokenize_caption(prediction, tokenizer)
            img_path = os.path.join(dataset.root_dir, "food_images", image_name+".jpg")
            original_image = Image.open(img_path).convert("RGB")
            image_saving_path = os.path.join(image_saving_dir, f"{image_name}.jpg")
            plot_images(original_image, image, caption_text, caption_prediction, image_saving_path)
    else:
        return None

def plot_images(original_image, transformed_image, original_caption, predicted_caption, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Ensure transformed image is in PIL format
    if isinstance(transformed_image, torch.Tensor):
        transform_to_pil = transforms.ToPILImage()
        transformed_image = transform_to_pil(transformed_image)

    # Plot original image (already PIL)
    axes[0].imshow(original_image)
    axes[0].axis("off")
    axes[0].set_title("Original Image")

    # Plot transformed image
    axes[1].imshow(transformed_image)
    axes[1].axis("off")
    axes[1].set_title("Transformed Image")

    # Add captions
    plt.figtext(0.5, 0.01, f"Original caption: {original_caption}\nPredicted caption: {predicted_caption}",
                wrap=True, horizontalalignment='center', fontsize=12)

    plt.savefig(save_path)

    plt.close(fig)

def loss_plot(train_losses, val_losses, output_dir):
    plt.figure(figsize=(6, 3))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))

def scores_plot(scores, output_dir):
    plt.figure(figsize=(6, 3))
    plt.plot(scores["bleu_1"], label="BLEU-1")
    plt.plot(scores["bleu_2"], label="BLEU-2")
    plt.plot(scores["rouge_l"], label="ROUGE-L")
    plt.plot(scores["meteor"], label="METEOR")
    plt.xlabel("Epoch")
    plt.ylabel("Score (%)")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_evolution.png"))