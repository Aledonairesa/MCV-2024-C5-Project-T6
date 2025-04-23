import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

# Load CSV
df = pd.read_csv('/ghome/c5mcv06/FIRD/clean_mapping_validation.csv')
titles = df["Title"].tolist()

# Define categories
categories = [
    "Meat & Poultry", "Seafood Specialties", "Vegetarian & Plant-Based",
    "Baked Goods & Breads", "Desserts & Sweet Treats", "Appetizers & Small Plates",
    "Soups & Stews", "Salads & Fresh Preparations", "Breakfast & Brunch",
    "Beverages & Drinks", "Condiments, Sauces & Preserves"
]

# Check for GPU
device = 0 if torch.cuda.is_available() else -1

# Initialize zero-shot classification pipeline with GPU
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device
)

# Batch processing function
def classify_batch(texts, candidate_labels):
    results = classifier(texts, candidate_labels, multi_label=False)
    # If only one result wrap in a list for consistency
    if isinstance(results, dict):
        results = [results]
    return [res["labels"][0] for res in results]

# Process in batches
batch_size = 32
predicted_categories = []

for i in tqdm(range(0, len(titles), batch_size), desc="Classifying"):
    batch = titles[i:i+batch_size]
    preds = classify_batch(batch, categories)
    predicted_categories.extend(preds)

# Add predictions to DataFrame
df["Predicted_Category"] = predicted_categories

# Save to CSV
df.to_csv("categorized_captions_validation.csv", index=False)
