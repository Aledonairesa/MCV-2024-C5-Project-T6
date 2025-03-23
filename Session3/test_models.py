import os
import time
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from utils.visualization import *
from utils.utils import *
from utils.models import *
from utils.configuration import *

def test_model(model, model_path, test_dataloader, test_dataset, device, tokenizer, output_dir="./test_outputs"):
    """
    Test a trained image captioning model and generate visualizations and metrics.
    
    Args:
        model_path (str): Path to the saved model weights
        test_dataloader (DataLoader): DataLoader for the test dataset
        test_dataset (FirdDataset): The test dataset
        device (torch.device): Device to run the model on
        idx2char (dict): Dictionary mapping indices to characters
        output_dir (str): Directory to save outputs
    
    Returns:
        dict: Dictionary containing test metrics
    """

    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the model
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize metrics
    criterion = torch.nn.CrossEntropyLoss()
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    
    # Lists to store results
    all_predictions = []
    all_references = []
    all_losses = []
    time_list= []
    # For tracking metrics over testing
    test_loss_list = []
    scores = {
    "bleu1": [],
    "bleu2": [],
    "rouge_l": [],
    "meteor": []
    }
    
    # Start testing
    print("Starting model evaluation...")
    
    
    with torch.no_grad():
        for batch_idx, (images, captions, idxs) in enumerate(tqdm(test_dataloader, desc="Testing")):
            start_time = time.time()
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            outputs, tokens = model(images)
            
            # Calculate loss
            loss = criterion(outputs, captions)
            all_losses.append(loss.item())
            
            # Visualize results for every 20th batch
            if batch_idx % 20 == 0:
                visualize_results(
                    images=images.cpu(), 
                    predicted_captions=tokens.cpu(), 
                    idxs=idxs, 
                    dataset=test_dataset,
                    tokenizer=tokenizer, 
                    epoch=0,  # Use 0 for test
                    phase="test"
                )
            
            # Collect predictions and references for metric calculation
            for i, (pred, idx) in enumerate(zip(tokens, idxs)):
                pred_str = detokenize_caption(pred, tokenizer)
                ref_str = test_dataset.data.iloc[int(idx)]['Title']
                all_predictions.append(pred_str)
                all_references.append([ref_str])  # Wrap in list for evaluate library

            time_list.append(time.time()-start_time)
    
    # Calculate test loss
    test_loss = sum(all_losses) / len(all_losses)
    test_loss_list.append(test_loss)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Calculate metrics
    print("Calculating metrics...")
    bleu1_result = bleu_metric.compute(predictions=all_predictions, references=all_references, max_order=1)
    bleu2_result = bleu_metric.compute(predictions=all_predictions, references=all_references, max_order=2)
    rouge_result = rouge_metric.compute(predictions=all_predictions, references=all_references)
    meteor_result = meteor_metric.compute(predictions=all_predictions, references=all_references)
    
    # Extract metric values
    epoch_bleu1 = bleu1_result['bleu'] * 100
    epoch_bleu2 = bleu2_result['bleu'] * 100
    epoch_rouge_l = rouge_result['rougeL'] * 100
    epoch_meteor = meteor_result['meteor'] * 100
    
    # Append metrics to lists
    scores["bleu1"].append(epoch_bleu1)
    scores["bleu2"].append(epoch_bleu2)
    scores["rouge_l"].append(epoch_rouge_l)
    scores["meteor"].append(epoch_meteor)


    # ---------------------------
    # PLOT METRIC EVOLUTION
    # ---------------------------
    scores_plot(scores, output_dir)

    # ---------------------------
    # SAVE FINAL METRICS TO CSV
    # ---------------------------
    metrics_csv_path = save_metrics_test(test_loss_list, time_list, scores, output_dir)
    print(f"Final metrics saved to {metrics_csv_path}")

# Example usage:
tokenizer = Tokenizer(tokenization_mode=TOKENIZATION_MODE)

test_split_path = os.path.join(ROOT_DIR, "clean_mapping_test.csv")
test_dataset = FirdDataset(csv_file=test_split_path, root_dir=ROOT_DIR, max_len=TEXT_MAX_LEN, tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# Load your model and datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ModelWithAttention(device=device, tokenizer=tokenizer, decoder_name="lstm").to(device)

test_metrics = test_model(
    model=model,
    model_path="/ghome/c5mcv06/biel_working_dir/exps_S3/res_finals/word_tf_lstm_resnet18_att/best_model.pth",
    test_dataloader=test_dataloader,
    test_dataset=test_dataset,
    device=device,
    tokenizer=tokenizer,
    output_dir="./test_results"
)