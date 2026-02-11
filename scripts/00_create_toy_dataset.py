import json
import random
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from logo_similarity.config import paths
from logo_similarity.utils.logging import logger

def create_toy_dataset():
    """
    Creates a toy dataset metadata file by sampling images per Vienna class.
    Saves to data/toy_results.json.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of samples per class")
    args = parser.parse_args()
    
    logger.info(f"Creating toy dataset with {args.samples} samples per class...")
    
    # 1. Load full metadata
    if not paths.DATASET_METADATA.exists():
        logger.error(f"Full metadata not found at {paths.DATASET_METADATA}")
        return

    logger.info("Loading full metadata...")
    with open(paths.DATASET_METADATA, "r") as f:
        full_data = json.load(f)
    
    logger.info(f"Loaded {len(full_data)} images.")

    # 2. Group by Vienna Code
    # An image can have multiple codes. We'll assign it to its first code for sampling purposes,
    # or we can try to balance more carefully. For a toy dataset, first code is fine.
    vienna_groups = defaultdict(list)
    no_code_count = 0
    
    for item in tqdm(full_data, desc="Grouping by Vienna Code"):
        codes = item.get('vienna_codes', [])
        if not codes:
            no_code_count += 1
            continue
            
        # Strategy: Use the first code to categorize
        primary_code = codes[0]
        vienna_groups[primary_code].append(item)

    logger.info(f"Found {len(vienna_groups)} unique Vienna codes.")
    logger.info(f"Skipped {no_code_count} images with no Vienna codes.")

    # 3. Sample 100 per class
    toy_dataset = []
    toy_dataset = []
    samples_per_class = args.samples
    
    # Track selected file IDs to avoid duplicates if we change strategy later
    selected_ids = set()

    for code, items in tqdm(vienna_groups.items(), desc="Sampling"):
        if len(items) <= samples_per_class:
            selected = items
        else:
            selected = random.sample(items, samples_per_class)
            
        for item in selected:
            # Fix extension mismatch: metadata has .JPG but files are .jpg
            if item['file'].endswith('.JPG'):
                item['file'] = item['file'].replace('.JPG', '.jpg')
                
            if item['file'] not in selected_ids:
                toy_dataset.append(item)
                selected_ids.add(item['file'])

    # 4. Save Toy Metadata
    logger.info(f"Toy dataset created with {len(toy_dataset)} images.")
    with open(paths.TOY_DATASET_METADATA, "w") as f:
        json.dump(toy_dataset, f, indent=2)
        
    logger.info(f"Saved toy metadata to {paths.TOY_DATASET_METADATA}")

if __name__ == "__main__":
    create_toy_dataset()
