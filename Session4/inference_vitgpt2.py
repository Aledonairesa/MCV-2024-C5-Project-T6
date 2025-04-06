import os
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import evaluate
import matplotlib.pyplot as plt

from transformers import VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer
from torch.utils.data import DataLoader
from dataloader import VitGPT2Dataset

def compute_metrics(references, predictions):
    """
    Compute multiple evaluation metrics using Hugging Face evaluate library
    
    Args:
        references (list): List of ground truth captions
        predictions (list): List of generated captions
    
    Returns:
        dict: Dictionary of metric scores
    """
    # Load metrics
    bleu_1 = evaluate.load("bleu")
    bleu_2 = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    # Compute BLEU-1 (unigram)
    bleu_1_results = bleu_1.compute(
        predictions=predictions, 
        references=[[ref] for ref in references], 
        max_order=1
    )

    # Compute BLEU-2 (bigram)
    bleu_2_results = bleu_2.compute(
        predictions=predictions, 
        references=[[ref] for ref in references], 
        max_order=2
    )

    # Compute ROUGE
    rouge_results = rouge.compute(
        predictions=predictions, 
        references=references,
        use_stemmer=True
    )

    # Compute METEOR
    meteor_results = meteor.compute(
        predictions=predictions, 
        references=references
    )
    
    return {
        'bleu_1': bleu_1_results['bleu'],
        'bleu_2': bleu_2_results['bleu'],
        'rouge_l': rouge_results['rougeL'],
        'meteor': meteor_results['meteor']
    }

def save_captioned_images(
    dataset, 
    generated_captions, 
    ground_truth_captions, 
    output_dir='captioned_images'
):
    """
    Save images with their ground truth and predicted captions
    
    Args:
        dataset (VitGPT2Dataset): Original dataset
        generated_captions (list): List of generated captions
        ground_truth_captions (list): List of ground truth captions
        output_dir (str): Directory to save captioned images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through images
    for idx in range(len(dataset)):
        # Get the original image from the dataset
        dataset_item = dataset[idx]
        original_image = dataset_item['original_image']
        image_name = dataset_item['image_name']
        
        # Get captions
        gt_caption = ground_truth_captions[idx]
        pred_caption = generated_captions[idx]
        
        # Create figure
        plt.figure(figsize=(10, 5))
        plt.imshow(original_image)
        plt.axis('off')
        
        # Add captions as text
        plt.title(f"Ground Truth: {gt_caption}\nPredicted: {pred_caption}", 
                  wrap=True, fontsize=10)
        
        # Save figure
        output_path = os.path.join(output_dir, f'image_{image_name}')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()

def save_results(results, output_dir):
    """
    Save evaluation results in multiple formats
    
    Args:
        results (dict): Evaluation results dictionary
        output_dir (str): Directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as CSV
    metrics_df = pd.DataFrame([results])
    metrics_csv_path = os.path.join(output_dir, 'evaluation_metrics.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # Save full results as pickle
    results_pickle_path = os.path.join(output_dir, 'evaluation_results.pkl')
    with open(results_pickle_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full results saved to {results_pickle_path}")
    
    # Save captions as separate CSV
    captions_df = pd.DataFrame({
        'Ground Truth Captions': results['ground_truth_captions'],
        'Generated Captions': results['generated_captions']
    })
    captions_csv_path = os.path.join(output_dir, 'image_captions.csv')
    captions_df.to_csv(captions_csv_path, index=False)
    print(f"Captions saved to {captions_csv_path}")

def evaluate_image_captioning(
    dataset, 
    model_name='nlpconnect/vit-gpt2-image-captioning', 
    batch_size=16, 
    max_length=150,
    device=None,
    save_images=True,
    output_dir='evaluation_results'
):
    """
    Perform comprehensive evaluation of image captioning model
    
    Args:
        dataset (VitGPT2Dataset): Dataset to evaluate
        model_name (str): Hugging Face model name
        batch_size (int): Batch size for evaluation
        max_length (int): Maximum length of generated captions
        device (torch.device, optional): Compute device 
        save_images (bool): Whether to save images with captions
        output_dir (str): Directory to save evaluation results
    
    Returns:
        dict: Evaluation metrics and captions
    """
    # Set up device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pre-trained model and processors
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    feature_extractor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Configure tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=dataset.collate_fn
    )
    
    # Lists to store results
    generated_captions = []
    ground_truth_captions = []
    
    # Evaluation loop
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            pixel_values = batch['pixel_values'].to(device)
            
            # Extract ground truth captions for reference
            gt_captions = [
                dataset.tokenizer.decode(caption, skip_special_tokens=True) 
                for caption in batch['input_ids']
            ]
            ground_truth_captions.extend(gt_captions)
            
            # Generate captions
            outputs = model.generate(
                pixel_values, 
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode generated captions
            batch_captions = [
                tokenizer.decode(g, skip_special_tokens=True) 
                for g in outputs
            ]
            generated_captions.extend(batch_captions)
    
    # Calculate metrics
    metrics = compute_metrics(ground_truth_captions, generated_captions)
    
    # Save images with captions if requested
    if save_images:
        save_captioned_images(
            dataset, 
            generated_captions, 
            ground_truth_captions, 
            os.path.join(output_dir, 'captioned_images')
        )
    
    # Add additional information to results
    results = {
        **metrics,
        'total_samples': len(dataset),
        'generated_captions': generated_captions,
        'ground_truth_captions': ground_truth_captions
    }
    
    # Save results to files
    save_results(results, output_dir)
    
    # Print sample results
    print("\nSample Evaluation Results:")
    for i in range(min(5, len(generated_captions))):
        print(f"Image {i+1}:")
        print(f"  Ground Truth: {ground_truth_captions[i]}")
        print(f"  Generated:    {generated_captions[i]}\n")
    
    # Print overall metrics
    print("\nOverall Metrics:")
    for metric, score in metrics.items():
        print(f"{metric.upper()}: {score:.4f}")
    
    return results

def main():
    # Example usage
    csv_file = '/ghome/c5mcv06/FIRD/clean_mapping_train.csv'
    root_dir = '/ghome/c5mcv06/FIRD'
    output_dir = './prova_train'
    
    # First, install required libraries
    # !pip install evaluate pandas
    
    # Create dataset
    dataset = VitGPT2Dataset(
        csv_file=csv_file, 
        root_dir=root_dir,
        max_len=150
    )
    
    # Perform evaluation
    results = evaluate_image_captioning(dataset, output_dir=output_dir)

if __name__ == "__main__":
    main()