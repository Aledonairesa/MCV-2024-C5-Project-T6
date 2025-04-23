import os
import torch
import time
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import random
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from huggingface_hub import login
from utils.interactive_html import create_interactive_viewer

# Food prompts to test
FOOD_PROMPTS = [
    "Apple Pie",
    "Bacon oatmeal and raisin cookies",
    "Crispy waffles with salted caramel coulis",
    "Egg in a hole sandwich with-bacon and cheddar",
    "Frozen negroni sluhy cocktail parsons",
    "Triple beef cheeseburgers with spiced ketchup and red vinegar pickles",
    "Roasted beet soup with creme fraiche"
]

# Model options
MODEL_OPTIONS = {
    "sd35_medium": {
        "name": "Stable Diffusion 3.5 Medium",
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "variant": "fp16"
    },
    "sdxl_base": {
        "name": "Stable Diffusion XL Base",
        "repo_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "variant": "fp16"
    }
}

# *** SELECT YOUR MODEL HERE ***
# Options: "sd35_medium" or "sdxl_base"
SELECTED_MODEL = "sdxl_base"  # Change this value to switch models

# Authenticate with Hugging Face Hub
login(token="hf_token")

def create_parameter_grid(prompts, cfg_scales, steps_list, samplers, neg_prompt_options):
    """Create grid of all parameter combinations to test"""
    parameter_grid = []
    
    for prompt_idx, prompt in enumerate(prompts):
        for cfg in cfg_scales:
            for steps in steps_list:
                for sampler in samplers:
                    for use_neg_prompt in neg_prompt_options:
                        parameter_grid.append({
                            "prompt_idx": prompt_idx,
                            "prompt": prompt,
                            "cfg_scale": cfg,
                            "steps": steps, 
                            "sampler": sampler,
                            "negative_prompt": use_neg_prompt
                        })
    
    return parameter_grid
    
def generate_images(pipe, parameter_grid, output_dir, model_key, seed=42):
    """Generate images for all parameter combinations and track timing"""
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Create images directory
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Save prompts to JSON
    prompts = []
    for params in parameter_grid:
        if params["prompt_idx"] >= len(prompts):
            prompts.append(params["prompt"])
    
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    
    # Process each parameter combination
    for params in tqdm(parameter_grid, desc="Generating images"):
        # Create a fresh generator with the same seed for each image
        generator = torch.Generator("cuda").manual_seed(seed)

        # Set up parameters
        prompt_idx = params["prompt_idx"]
        prompt = params["prompt"]
        cfg = params["cfg_scale"]
        steps = params["steps"]
        sampler = params["sampler"]
        use_neg_prompt = params["negative_prompt"]
        neg_prompt = "low quality, blurry, distorted food, ugly" if use_neg_prompt else ""
        
        # Set sampler
        if sampler == "DDIM":
            pipe.scheduler_type = "ddim"
        elif sampler == "DDPM":
            pipe.scheduler_type = "ddpm"
        
        # Measure generation time
        start_time = time.time()
        
        # Generate image
        image = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator
        ).images[0]
        
        # Calculate generation time
        generation_time = time.time() - start_time

        # Create a copy with parameter info
        labeled_image = image.copy()
        draw = ImageDraw.Draw(labeled_image)
        font_size = 10
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # Try system font
        except:
            # Default font if arial not available
            font = ImageFont.load_default()
        
        # Save individual image
        filename = f"prompt{prompt_idx:02d}_cfg{cfg}_steps{steps}_{sampler}_neg{use_neg_prompt}.png"
        image_path = os.path.join(images_dir, filename)
        labeled_image.save(image_path)
        
        # Add to results, including generation time
        results.append({
            "prompt_idx": prompt_idx,
            "prompt": prompt,
            "cfg_scale": cfg,
            "steps": steps,
            "sampler": sampler,
            "negative_prompt": use_neg_prompt,
            "image_path": os.path.join("images", filename),
            "generation_time": generation_time
        })
    
    # Save results to JSON for the interactive viewer
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2) 
    
    # Save results CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "parameter_results.csv"), index=False)
    
    # Generate timing analysis
    analyze_generation_times(results_df, output_dir, model_key)
    
    return results_df

def analyze_generation_times(results_df, output_dir, model_key):
    """Analyze and visualize the impact of parameters on generation time"""
    timing_dir = os.path.join(output_dir, "timing_analysis")
    os.makedirs(timing_dir, exist_ok=True)
    
    # Calculate average generation time per parameter
    avg_by_cfg = results_df.groupby('cfg_scale')['generation_time'].mean().reset_index()
    avg_by_steps = results_df.groupby('steps')['generation_time'].mean().reset_index()
    avg_by_sampler = results_df.groupby('sampler')['generation_time'].mean().reset_index()
    avg_by_neg = results_df.groupby('negative_prompt')['generation_time'].mean().reset_index()
    
    # Create timing summary CSV
    timing_summary = pd.DataFrame({
        'parameter': ['Model'] + 
                     ['CFG=' + str(cfg) for cfg in avg_by_cfg['cfg_scale']] + 
                     ['Steps=' + str(steps) for steps in avg_by_steps['steps']] + 
                     ['Sampler=' + sampler for sampler in avg_by_sampler['sampler']] + 
                     ['NegPrompt=' + str(neg) for neg in avg_by_neg['negative_prompt']],
        'value': [MODEL_OPTIONS[model_key]['name']] + 
                 avg_by_cfg['generation_time'].tolist() + 
                 avg_by_steps['generation_time'].tolist() + 
                 avg_by_sampler['generation_time'].tolist() + 
                 avg_by_neg['generation_time'].tolist(),
    })
    timing_summary.to_csv(os.path.join(timing_dir, "timing_summary.csv"), index=False)
    
    # Generate plots
    # Plot timing by CFG scale
    plt.figure(figsize=(10, 6))
    plt.bar(avg_by_cfg['cfg_scale'].astype(str), avg_by_cfg['generation_time'])
    plt.xlabel('CFG Scale')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title(f'Generation Time by CFG Scale - {MODEL_OPTIONS[model_key]["name"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(timing_dir, "timing_by_cfg.png"))
    plt.close()
    
    # Plot timing by steps
    plt.figure(figsize=(10, 6))
    plt.bar(avg_by_steps['steps'].astype(str), avg_by_steps['generation_time'])
    plt.xlabel('Number of Steps')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title(f'Generation Time by Steps - {MODEL_OPTIONS[model_key]["name"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(timing_dir, "timing_by_steps.png"))
    plt.close()
    
    # Plot timing by sampler
    plt.figure(figsize=(10, 6))
    plt.bar(avg_by_sampler['sampler'], avg_by_sampler['generation_time'])
    plt.xlabel('Sampler')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title(f'Generation Time by Sampler - {MODEL_OPTIONS[model_key]["name"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(timing_dir, "timing_by_sampler.png"))
    plt.close()
    
    # Plot timing by negative prompt usage
    plt.figure(figsize=(10, 6))
    labels = ['Without Negative Prompt', 'With Negative Prompt']
    values = avg_by_neg.sort_values('negative_prompt')['generation_time'].tolist()
    plt.bar([0, 1], values)
    plt.xticks([0, 1], labels)
    plt.xlabel('Negative Prompt Usage')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title(f'Generation Time by Negative Prompt Usage - {MODEL_OPTIONS[model_key]["name"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(timing_dir, "timing_by_neg_prompt.png"))
    plt.close()
    
    # Create summary plot with all parameters
    plt.figure(figsize=(12, 8))
    # Convert parameter and value columns to appropriate types
    timing_summary_plot = timing_summary.copy()
    # Ensure we're plotting with string types on x-axis
    plt.bar(timing_summary_plot['parameter'].astype(str), timing_summary_plot['value'].astype(float))
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Parameter')
    plt.ylabel('Average Generation Time (seconds)')
    plt.title(f'Generation Time by Parameter - {MODEL_OPTIONS[model_key]["name"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(timing_dir, "timing_summary.png"))
    plt.close()

def main():
    # Use the hardcoded model selection
    model_key = SELECTED_MODEL
    model_info = MODEL_OPTIONS[model_key]
    
    # Parameters to test
    cfg_scales = [5, 7.5, 10, 12]
    denoising_steps = [20, 50, 100]
    sampler_types = ["DDIM", "DDPM"]
    negative_prompt_options = [True, False]  # True = enabled, False = disabled
    
    prompts = FOOD_PROMPTS
    seed = 42
    
    # Set up directories with model name
    output_dir = f"param_opti_{model_key}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using model: {model_info['name']}")
    print(f"Will test {len(cfg_scales)} CFG scales × {len(denoising_steps)} step counts × "
          f"{len(sampler_types)} samplers × {len(negative_prompt_options)} negative prompt options "
          f"on {len(prompts)} prompts = {len(cfg_scales) * len(denoising_steps) * len(sampler_types) * len(negative_prompt_options) * len(prompts)} total images")
    
    # Load the selected model
    print(f"Loading {model_info['name']}...")
    pipe = DiffusionPipeline.from_pretrained(
        model_info['repo_id'], 
        torch_dtype=torch.float16,
        variant=model_info['variant']
    )
    pipe = pipe.to("cuda")
    
    # Create parameter grid
    parameter_grid = create_parameter_grid(
        prompts,
        cfg_scales,
        denoising_steps,
        sampler_types,
        negative_prompt_options
    )
    
    # Generate images and track timing
    print(f"Generating {len(parameter_grid)} images with {model_info['name']}...")
    results_df = generate_images(pipe, parameter_grid, output_dir, model_key, seed)
    
    # Create interactive viewer
    print("Creating interactive viewer...")
    create_interactive_viewer(
        output_dir,
        cfg_scales,
        denoising_steps,
        sampler_types,
        negative_prompt_options
    )
    
    print("\nProcess complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Open {os.path.join(output_dir, 'interactive_viewer.html')} in your browser to compare results")

if __name__ == "__main__":
    main()