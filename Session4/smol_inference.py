import os
import csv
import argparse
import torch
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

def process_images(csv_file, image_dir, output_csv, output_image_dir, system_prompt, user_prompt, visualize=True):
    """
    Process images in the CSV file with SmolVLM-256M-Instruct for image captioning.
    
    Args:
        csv_file (str): Path to the input CSV file.
        image_dir (str): Directory containing the images.
        output_csv (str): Path to save the output CSV file.
        output_image_dir (str): Directory to save the visualized images.
        system_prompt (str): System prompt for the model.
        user_prompt (str): User prompt for the model.
        visualize (bool): Whether to create visualization images.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_image_dir, exist_ok=True)
    
    # Load CSV data
    df = pd.read_csv(csv_file)
    
    # Initialize model and processor
    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Use device_map="auto" for multi-GPU support and avoid specifying flash attention
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    
    # Initialize output CSV with header
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Image_Name', 'Original_Title', 'Predicted_Caption'])
    
    # Process images
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_name = row['Image_Name'] + ".jpg"
        original_title = row['Title']
        
        # Check if image exists
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found. Skipping...")
            continue
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create messages for SmolVLM
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Prepare inputs
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=[image], return_tensors="pt")
            inputs = inputs.to(model.device)
            
            # Generate caption
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            # Decode the generated text
            predicted_caption = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Extract the assistant's response (removing the prompt parts)
            if "Assistant:" in predicted_caption:
                predicted_caption = predicted_caption.split("Assistant:")[1].strip()
            
            # Write prediction to CSV
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([image_name, original_title, predicted_caption])
            
            # Create visualization
            if visualize:
                visualize_prediction(image, original_title, predicted_caption, 
                                os.path.join(output_image_dir, f"viz_{image_name}"))
            
        except Exception as e:
            print(f"Error processing image {image_name}: {e}")
    
    print(f"Processing complete! Results saved to {output_csv} and visualizations in {output_image_dir}")

def visualize_prediction(image, original_title, predicted_caption, output_path):
    """
    Create a visualization with the original image and both captions.
    
    Args:
        image (PIL.Image): The original image.
        original_title (str): The original title/caption.
        predicted_caption (str): The predicted caption.
        output_path (str): Path to save the visualization.
    """
    # Calculate new image dimensions with frame
    img_width, img_height = image.size
    frame_width = img_width + 200  # Add padding for text
    frame_height = img_height + 250  # Add space for text below
    
    # Create new image with white background
    new_img = Image.new('RGB', (frame_width, frame_height), color='white')
    
    # Paste original image
    new_img.paste(image, (100, 100))
    
    # Add text
    draw = ImageDraw.Draw(new_img)
    
    # Try to use a standard font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial", 20)
    except IOError:
        font = ImageFont.load_default()
    
    # Add original and predicted captions
    draw.text((20, img_height + 120), f"Original: {original_title}", fill='black', font=font)
    draw.text((20, img_height + 160), f"Predicted: {predicted_caption}", fill='black', font=font)
    
    # Save the new image
    new_img.save(output_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Caption images using SmolVLM-256M-Instruct')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_csv', type=str, default='predicted_captions.csv', help='Path to output CSV file')
    parser.add_argument('--output_image_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--system_prompt', type=str, default='You are a helpful assistant that captions food images accurately and concisely.', help='System prompt for the model')
    parser.add_argument('--user_prompt', type=str, default='Give the recipe name for this food dish in one concise sentence. Do not include any other information.', help='User prompt for the model')
    parser.add_argument('--visualize', type=bool, default=True, help='Whether to visualize predictions')
    
    args = parser.parse_args()
    
    process_images(
        args.csv_file,
        args.image_dir,
        args.output_csv,
        args.output_image_dir,
        args.system_prompt,
        args.user_prompt,
        args.visualize
    )