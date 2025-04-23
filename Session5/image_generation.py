import os
import torch
import argparse
import re
import random
from diffusers import DiffusionPipeline
from tqdm import tqdm

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Generate images from prompts in a text file using Stable Diffusion')
    parser.add_argument('--prompt_file', type=str, required=True, help='Path to text file with prompts')
    parser.add_argument('--output_dir', type=str, default='generated_images2', help='Directory to save generated images')
    parser.add_argument('--model', type=str, default='sd35', 
                        help='Model path or identifier (e.g., local path or "stabilityai/stable-diffusion-3.5-medium")')
    parser.add_argument('--cfg', type=float, default=5.0, help='Classifier Free Guidance scale')
    parser.add_argument('--steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--scheduler', type=str, choices=['ddim', 'ddpm'], default='ddpm', 
                        help='Scheduler to use: ddim or ddpm')
    parser.add_argument('--negative_prompt', type=str, default='', help='Negative prompt (leave empty for none)')
    parser.add_argument('--base_seed', type=int, default=42, help='Base random seed for generation (ignored when using random seeds)')
    parser.add_argument('--num_images', type=int, default=3, help='Number of images to generate per prompt')
    parser.add_argument('--use_local', action='store_true', help='Use a locally downloaded model')
    parser.add_argument('--local_path', type=str, default=None, help='Path to locally downloaded model')
    parser.add_argument('--random_seeds', action='store_true', default=True, help='Use completely random seeds instead of sequential')
    parser.add_argument('--resume', action='store_true', help='Resume from where processing left off')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prompts from file
    with open(args.prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip()+', food photography' for line in f.readlines() if line.strip()]

    
    print(f"Found {len(prompts)} prompts in file")
    
    # Check which prompts have already been processed if resuming
    if args.resume:
        completed_prompt_indices = find_completed_prompts(args.output_dir, prompts)
        if completed_prompt_indices:
            print(f"Found {len(completed_prompt_indices)} already processed prompts. Will skip these.")
    else:
        completed_prompt_indices = []
    
    # Load the model
    print(f"Loading model...")
    try:
        if args.use_local and args.local_path:
            print(f"Using locally downloaded model from: {args.local_path}")
            pipe = load_local_model(args.local_path)
        else:
            print(f"Using model: {args.model}")
            pipe = load_model(args.model)
        
        # Set scheduler
        print(f"Setting scheduler to: {args.scheduler}")
        update_scheduler(pipe, args.scheduler)
        
        # Generate images
        print(f"Generating {args.num_images} images per prompt ({len(prompts)} prompts) with:")
        print(f"  CFG: {args.cfg}, Steps: {args.steps}, Scheduler: {args.scheduler}")
        print(f"  Using completely random seeds for generation")
        if args.negative_prompt:
            print(f"  Negative prompt: {args.negative_prompt}")
        
        generate_images(
            pipe=pipe,
            prompts=prompts,
            output_dir=args.output_dir,
            cfg_scale=args.cfg,
            steps=args.steps,
            negative_prompt=args.negative_prompt,
            num_images=args.num_images,
            completed_prompt_indices=completed_prompt_indices
        )
        
        print(f"\nAll done! Images saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

def find_completed_prompts(output_dir, prompts):
    """Find which prompts have already been processed by scanning the output directory"""
    completed_prompt_indices = set()
    
    # Get all image filenames in the output directory
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    # Extract prompt numbers from filenames
    pattern = r'_prompt(\d+)_seed\d+\.png$'
    for filename in image_files:
        match = re.search(pattern, filename)
        if match:
            prompt_index = int(match.group(1)) - 1  # Convert to 0-based index
            completed_prompt_indices.add(prompt_index)
    
    return completed_prompt_indices
        
def load_model(model_id):
    """Load a model from Hugging Face Hub or use a custom identifier"""
    # Map common model shortcuts to their paths
    model_map = {
        'sd35': "stabilityai/stable-diffusion-3.5-medium",
        'sdxl': "stabilityai/stable-diffusion-xl-base-1.0",
        'sd21': "stabilityai/stable-diffusion-2-1"
    }
    
    # Get the actual model path
    model_path = model_map.get(model_id, model_id)
    
    # Load the model
    pipe = DiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe = pipe.to("cuda")
    
    return pipe

def load_local_model(local_path):
    """Load a model from a local directory"""
    pipe = DiffusionPipeline.from_pretrained(
        local_path,
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True
    )
    pipe = pipe.to("cuda")
    
    return pipe

def update_scheduler(pipe, scheduler_type):
    """Update the pipeline's scheduler"""
    pipe.scheduler_type = scheduler_type

def create_filename_from_prompt(prompt, max_length=50):
    """Create a filename-safe version of the prompt"""
    # Remove special characters and replace spaces with underscores
    filename = re.sub(r'[^\w\s]', '', prompt).strip().replace(' ', '_')
    
    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]
    
    # Ensure filename is not empty
    if not filename:
        filename = "image"
    
    return filename

def generate_images(pipe, prompts, output_dir, cfg_scale, steps, negative_prompt, num_images, completed_prompt_indices=None):
    """Generate multiple images for all prompts with completely random seeds"""
    
    if completed_prompt_indices is None:
        completed_prompt_indices = set()
    
    # Save all prompts to a reference file (only if not resuming or the file doesn't exist)
    prompts_file_path = os.path.join(output_dir, "all_prompts.txt")
    if not os.path.exists(prompts_file_path):
        with open(prompts_file_path, "w", encoding="utf-8") as f:
            for i, prompt in enumerate(prompts):
                f.write(f"Prompt {i+1}: {prompt}\n\n")
    
    # Process only the prompts that haven't been completed yet
    remaining_prompts = [(i, prompt) for i, prompt in enumerate(prompts) if i not in completed_prompt_indices]
    
    for i, prompt in tqdm(remaining_prompts, desc="Processing prompts"):
        # Create a base filename from the prompt
        base_filename = create_filename_from_prompt(prompt)
        
        # Generate multiple images with different seeds
        for j in range(num_images):
            # Generate a completely random seed between 0 and 2^32-1 (torch generator range)
            random_seed = random.randint(0, 2**32 - 1)
            generator = torch.Generator("cuda").manual_seed(random_seed)
            
            # Generate the image
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator
            ).images[0]
            
            # Save the image with prompt-based filename and seed
            filename = f"{base_filename}_prompt{i+1}_seed{random_seed}.png"
            image_path = os.path.join(output_dir, filename)
            image.save(image_path)
            
            # Print info about the current generation
            prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
            print(f"Generated {filename} for prompt {i+1}/{len(prompts)}: {prompt_short}")

if __name__ == "__main__":
    main()