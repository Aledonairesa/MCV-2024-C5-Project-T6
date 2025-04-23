import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
import psutil
import gc
import json
import pandas as pd
from huggingface_hub import login
from tqdm import tqdm

# Create output directory
output_dir = "diffusion_model_comparison_3.5Medium"
os.makedirs(output_dir, exist_ok=True)

# Authenticate with Hugging Face Hub
login(token="hf_token")

# Models to compare - revised to avoid incompatible refiners
model_configs = [
    #{"name": "SD 1.5", "model_id": "runwayml/stable-diffusion-v1-5", "type": "sd", "has_refiner": False, "use_cpu_offload": False},
    #{"name": "SD 2.1", "model_id": "stabilityai/stable-diffusion-2-1-base", "type": "sd", "has_refiner": False, "use_cpu_offload": False},
    #{"name": "SDXL Base", "model_id": "stabilityai/stable-diffusion-xl-base-1.0", "type": "sdxl", "has_refiner": False, "use_cpu_offload": False},
    #{"name": "SDXL Base + Refiner", "model_id": "stabilityai/stable-diffusion-xl-base-1.0", 
    # "refiner_id": "stabilityai/stable-diffusion-xl-refiner-1.0", "type": "sdxl", "has_refiner": True, "use_cpu_offload": False},
    #{"name": "SDXL Turbo", "model_id": "stabilityai/sdxl-turbo", "type": "sdxl", "has_refiner": False, "use_cpu_offload": False},
    {"name": "SD 3.5 Medium", "model_id": "stabilityai/stable-diffusion-3.5-medium", "type": "3.5", "has_refiner": False, "use_cpu_offload": False},
]

# Test prompts (food-related)
prompts = [
    "Baked sardines in pepperonata",
    "Chocolate brownie cookies",
    "Cilantro garlic yogurt sauce",
    "Indian spiced chicken eggplant and tomato skewers",
    "22 minute pad thai",
    "Best ever barbecue ribs",
    "Best of both worlds lobster",
    "Chinese orange chicken",
    "Frozen negroni sluhy cocktail parsons",
    "Little lemony ricotta cheesecake",
    "Apple Pie",
    "Chicken fajitas",
    "A green peas soup without meat",
    "Triple beef cheeseburgers with spiced ketchup and red vinegar pickles",
    "Asian turkey noodle soup with ginger and chiles",
    "Prosciutto watercress and fontina toasties",
    "Halibut confit with leeks coriander and lemon",
    "Roasted beet soup with creme fraiche"
]
num_variations = 4
fixed_seed = 42

# Refinement parameters
refiner_steps = 10
high_noise_frac = 0.8

# Data collection structures
performance_data = {
    "prompt": [],
    "model": [],
    "generation_time_seconds": [],
    "vram_usage_gb": [],
    "ram_usage_gb": [],
    "cpu_usage_percent": [],
    "image_width": [],
    "image_height": [],
    "variation": [],
    "has_refiner": []
}

# Function to get GPU memory usage
def get_gpu_memory_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # in GB
    return 0

# Function to clear memory (more thorough for large models)
def clear_memory(thorough=False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if thorough:
            torch.cuda.synchronize()
    gc.collect()

# Function to load model pipeline with adaptive memory management
def load_model(config):
    model_id = config["model_id"]
    model_type = config["type"]
    has_refiner = config.get("has_refiner", False)
    use_cpu_offload = config.get("use_cpu_offload", False)
    
    print(f"Loading {model_id}...")
    clear_memory(thorough=True)  # Clear memory before loading large models
    
    try:
        # Load base model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set memory efficient settings
        torch.set_grad_enabled(False)  # Disable gradient tracking
        
        if model_type == "sdxl":
            # SDXL models
            base_pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        elif model_type == "sd":
            # Standard SD models
            base_pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        else:
            # Generic pipeline as fallback (for SD 3.5)
            base_pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        
        # Enable attention slicing for all models
        base_pipeline.enable_attention_slicing()
        
        # Conditionally enable CPU offload for larger models
        if use_cpu_offload:
            print(f"Enabling CPU offload for {model_id} to manage memory")
            base_pipeline.enable_model_cpu_offload()
        else:
            # If not using CPU offload, move to GPU
            base_pipeline = base_pipeline.to(device)
        
        # Load refiner if specified
        refiner_pipeline = None
        if has_refiner:
            refiner_id = config.get("refiner_id")
            if model_type == "sdxl":
                print(f"Loading refiner {refiner_id}...")
                refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    refiner_id,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16"
                )
                
                refiner_pipeline.enable_attention_slicing()
                
                # Apply same CPU offload strategy as base model
                if use_cpu_offload:
                    refiner_pipeline.enable_model_cpu_offload()
                else:
                    refiner_pipeline = refiner_pipeline.to(device)
        
        return (base_pipeline, refiner_pipeline)
        
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        clear_memory(thorough=True)  # Clear memory after failure
        return (None, None)

# Function to generate images with smart resource monitoring
def generate_images(pipeline_tuple, prompt, num_images, prompt_index, config, seed=None):
    base_pipeline, refiner_pipeline = pipeline_tuple
    images = []
    
    if base_pipeline is None:
        return [None] * num_images
    
    model_name = config["name"]
    has_refiner = config.get("has_refiner", False)
    model_type = config["type"]
    use_cpu_offload = config.get("use_cpu_offload", False)
    
    for i in range(num_images):
        try:
            # Use fixed seed for the last image if provided
            if seed is not None and i == num_images - 1:
                # For CPU-offloaded models, we can't access device directly
                if use_cpu_offload:
                    generator = torch.Generator().manual_seed(seed)
                else:
                    generator = torch.Generator(device=base_pipeline.device).manual_seed(seed)
                variation_name = "fixed"
            else:
                if use_cpu_offload:
                    generator = torch.Generator().manual_seed(torch.random.seed())
                else:
                    generator = torch.Generator(device=base_pipeline.device).manual_seed(torch.random.seed())
                variation_name = f"var_{i+1}"
            
            # Start measuring
            process = psutil.Process(os.getpid())
            start_ram = process.memory_info().rss / (1024 * 1024 * 1024)  # in GB
            start_vram = get_gpu_memory_usage()
            start_cpu = psutil.cpu_percent(interval=None)
            start_time = time.time()
            
            # Generate image
            print(f"  Generating image {i+1}/{num_images} for prompt {prompt_index+1}...")
            
            if has_refiner and refiner_pipeline is not None and model_type == "sdxl":
                # Two-pass generation with base model and refiner for SDXL
                # Calculate steps for base and refiner
                total_steps = 30  # Total steps across both pipelines
                base_steps = int(total_steps * high_noise_frac)
                
                # First pass with base model
                first_pass = base_pipeline(
                    prompt,
                    num_inference_steps=base_steps,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                    generator=generator
                )
                
                # Store latents
                latents = first_pass.images
                
                # Second pass with refiner
                image = refiner_pipeline(
                    prompt,
                    num_inference_steps=total_steps,
                    denoising_start=high_noise_frac,
                    image=latents,
                    generator=generator
                ).images[0]
            else:
                # Single pass generation for models without refiners
                image = base_pipeline(
                    prompt, 
                    num_inference_steps=30,
                    generator=generator
                ).images[0]
            
            # End measuring
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            end_vram = get_gpu_memory_usage()
            end_ram = process.memory_info().rss / (1024 * 1024 * 1024)  # in GB
            
            # Calculate and store metrics
            generation_time = end_time - start_time
            vram_usage = max(0, end_vram - start_vram)  # In case of measurement noise
            ram_usage = max(0, end_ram - start_ram)
            cpu_usage = (start_cpu + end_cpu) / 2  # Average CPU usage
            
            # Store performance data
            performance_data["prompt"].append(prompt_index)
            performance_data["model"].append(model_name)
            performance_data["generation_time_seconds"].append(generation_time)
            performance_data["vram_usage_gb"].append(vram_usage)
            performance_data["ram_usage_gb"].append(ram_usage)
            performance_data["cpu_usage_percent"].append(cpu_usage)
            performance_data["image_width"].append(image.width)
            performance_data["image_height"].append(image.height)
            performance_data["variation"].append(variation_name)
            performance_data["has_refiner"].append(has_refiner)
            
            images.append(image)
            
            # Clear some memory between iterations
            clear_memory(thorough=True)
            
        except Exception as e:
            print(f"  Error generating image: {e}")
            images.append(None)
    
    return images

# Function to reorder models to optimize memory usage
def reorder_models_for_memory_efficiency(models):
    # Put largest models last to avoid memory fragmentation
    model_sizes = {
        "SD 1.5": 1,
        "SD 2.1": 2, 
        "SDXL Base": 3,
        "SDXL Turbo": 3,
        "SDXL Base + Refiner": 4,
        "SD 3.5 Medium": 5
    }
    
    return sorted(models, key=lambda x: model_sizes.get(x["name"], 0))

# Reorder models to process smaller ones first
model_configs = reorder_models_for_memory_efficiency(model_configs)

# Process each prompt
for prompt_index, prompt in enumerate(prompts):
    # Skip the ones already processed
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_index+1}")
    if os.path.exists(prompt_dir) and len(os.listdir(prompt_dir)) > 1:
        print(f"\nSkipping already processed prompt {prompt_index+1}: {prompt}")
        continue
    
    print(f"\n{'='*40}\nProcessing prompt {prompt_index+1}: {prompt}\n{'='*40}")
    
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Save the prompt text for reference
    with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    
    # Generate images for each model
    all_model_images = []
    
    for current_model_index, config in enumerate(model_configs):
        print(f"\nProcessing model: {config['name']}")
        try:
            # Load model
            pipelines = load_model(config)
            
            if pipelines[0] is not None:  # Check if base pipeline loaded successfully
                # Generate random seed images
                random_seed_images = generate_images(pipelines, prompt, num_variations, prompt_index, config)
                
                # Generate fixed seed image
                fixed_seed_image = generate_images(pipelines, prompt, 1, prompt_index, config, fixed_seed)[0]
                
                # Save images to disk
                for i, img in enumerate(random_seed_images):
                    if img is not None:
                        img_path = os.path.join(prompt_dir, f"{config['name'].replace(' ', '_')}_random_{i}.png")
                        img.save(img_path)
                
                if fixed_seed_image is not None:
                    fixed_seed_image.save(os.path.join(prompt_dir, f"{config['name'].replace(' ', '_')}_fixed.png"))
                
                # Store for grid creation
                all_model_images.append((config["name"], random_seed_images, fixed_seed_image))
                
                # Free GPU memory
                base_pipeline, refiner_pipeline = pipelines
                del base_pipeline
                if refiner_pipeline is not None:
                    del refiner_pipeline
                
                # Do a thorough cleanup after each model
                clear_memory(thorough=True)
            else:
                print(f"Skipping {config['name']} due to loading error")
                all_model_images.append((config["name"], None, None))
            
        except Exception as e:
            print(f"Error with model {config['name']}: {e}")
            all_model_images.append((config["name"], None, None))
            clear_memory(thorough=True)
    
    # Create comparison grids for this prompt
    def create_grid(images, titles, filepath, figsize=(20, 15)):
        """Create grid of images with titles"""
        rows = len(images)
        cols = len(images[0])
        
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        
        for i in range(rows):
            for j in range(cols):
                if cols == 1 and rows == 1:
                    ax = axs
                elif rows == 1:
                    ax = axs[j]
                elif cols == 1:
                    ax = axs[i]
                else:
                    ax = axs[i, j]
                
                if j < len(images[i]) and images[i][j] is not None:
                    ax.imshow(np.array(images[i][j]))
                    ax.set_title(titles[i][j])
                else:
                    ax.text(0.5, 0.5, "Generation failed", 
                            horizontalalignment='center', verticalalignment='center')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 1. Create individual model grids (4 images per model)
    for name, random_images, _ in all_model_images:
        if random_images and any(img is not None for img in random_images):
            images = [random_images]
            titles = [[f"{name} - Var {i+1}" for i in range(len(random_images))]]
            create_grid(images, titles, os.path.join(prompt_dir, f"{name.replace(' ', '_')}_variations.png"), figsize=(20, 5))
    
    # 2. Create comparison grid (models in columns, variations in rows)
    if all_model_images:
        valid_models = [m for m in all_model_images if m[1] is not None and any(img is not None for img in m[1])]
        if valid_models:
            # Extract model names
            model_names = [m[0] for m in valid_models]
            
            # Prepare comparison grid for random seed images
            random_comparison_images = []
            random_comparison_titles = []
            
            for i in range(num_variations):
                row_images = [m[1][i] if i < len(m[1]) else None for m in valid_models]
                row_titles = [f"{name} - Var {i+1}" for name in model_names]
                random_comparison_images.append(row_images)
                random_comparison_titles.append(row_titles)
            
            # Add fixed seed row
            fixed_comparison_images = [[m[2] for m in valid_models]]
            fixed_comparison_titles = [[f"{name} - Fixed Seed" for name in model_names]]
            
            # Create random seed comparison grid
            if random_comparison_images:
                create_grid(random_comparison_images, random_comparison_titles, 
                            os.path.join(prompt_dir, "model_comparison_random.png"), 
                            figsize=(len(valid_models) * 5, num_variations * 5))
            
            # Create fixed seed comparison grid
            if all(img is not None for img in fixed_comparison_images[0]):
                create_grid(fixed_comparison_images, fixed_comparison_titles, 
                            os.path.join(prompt_dir, "model_comparison_fixed_seed.png"), 
                            figsize=(len(valid_models) * 5, 5))
            
            # Combined grid (random + fixed)
            combined_images = random_comparison_images + fixed_comparison_images
            combined_titles = random_comparison_titles + fixed_comparison_titles
            create_grid(combined_images, combined_titles, 
                        os.path.join(prompt_dir, "model_comparison_combined.png"), 
                        figsize=(len(valid_models) * 5, (num_variations + 1) * 5))

# Save performance data
print("\nSaving performance data...")

# Convert to pandas DataFrame
df = pd.DataFrame(performance_data)

# Save as CSV
df.to_csv(os.path.join(output_dir, "performance_data.csv"), index=False)

# Save as JSON (for more flexibility)
with open(os.path.join(output_dir, "performance_data.json"), 'w') as f:
    json.dump(performance_data, f, indent=2)

# Create performance visualization
print("Creating performance visualizations...")

# Set up plotting style
plt.style.use('ggplot')

# Average generation time by model
plt.figure(figsize=(15, 8))
avg_time = df.groupby('model')['generation_time_seconds'].mean().sort_values(ascending=False)
avg_time.plot(kind='bar')
plt.title('Average Generation Time by Model')
plt.ylabel('Time (seconds)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "avg_generation_time.png"), dpi=300)
plt.close()

# Average VRAM usage by model
plt.figure(figsize=(15, 8))
avg_vram = df.groupby('model')['vram_usage_gb'].mean().sort_values(ascending=False)
avg_vram.plot(kind='bar')
plt.title('Average VRAM Usage by Model')
plt.ylabel('VRAM (GB)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "avg_vram_usage.png"), dpi=300)
plt.close()

# Image resolutions by model
plt.figure(figsize=(15, 8))
df['resolution'] = df['image_width'].astype(str) + 'x' + df['image_height'].astype(str)
resolution_counts = df.groupby(['model', 'resolution']).size().unstack(fill_value=0)
resolution_counts.plot(kind='bar', stacked=True)
plt.title('Image Resolutions by Model')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "image_resolutions.png"), dpi=300)
plt.close()

# Generation time box plot
plt.figure(figsize=(15, 10))
df_box = df.pivot(columns='model', values='generation_time_seconds')
df_box.boxplot(vert=False, figsize=(15, 10))
plt.title('Generation Time Distribution by Model')
plt.xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "generation_time_boxplot.png"), dpi=300)
plt.close()

# Compare models with and without refiners
plt.figure(figsize=(15, 8))
refiner_comparison = df.groupby(['model', 'has_refiner'])['generation_time_seconds'].mean().unstack(fill_value=0)
if not refiner_comparison.empty and 1.0 in refiner_comparison.columns:
    # Only plot if we have models with refiners
    refiner_comparison.plot(kind='bar')
    plt.title('Impact of Refiners on Generation Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(['Without Refiner', 'With Refiner'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "refiner_impact.png"), dpi=300)
    plt.close()

print(f"\nAll data and visualizations saved to {output_dir} directory")
