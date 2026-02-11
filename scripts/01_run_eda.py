#!/usr/bin/env python3
"""
Phase 0: Data Analysis & Preparation (EDA)

Analyzes dataset characteristics and prepares train/val/test splits.
- Dataset statistics with incremental checkpointing
- Train/val/test splits stratified by Vienna codes
- Known-similar pairs for validation

Usage:
    python scripts/01_run_eda.py [--resume]
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logo_similarity.config.paths import (
    RAW_DATASET_DIR,
    SPLITS_DIR,
    TOY_SPLITS_DIR,
    VALIDATION_DIR,
    TOY_VALIDATION_DIR,
    DATA_DIR,
    DATASET_METADATA,
    TOY_DATASET_METADATA
)
from logo_similarity.utils.logging import setup_logging

logger = setup_logging()

# Constants
CHECKPOINT_INTERVAL = 10000  # Save stats every N images
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_all_metadata(dataset_dir: Path) -> List[Dict]:
    """Load metadata from all yearly JSON files."""
    if dataset_dir.suffix == '.json':
        # Single file provided (e.g. toy dataset metadata)
        logger.info(f"Loading single metadata file: {dataset_dir}")
        with open(dataset_dir, "r", encoding="utf-8") as f:
            all_entries = json.load(f)
    else:
        # Directory - check for results.json or legacy output_*.json
        if (dataset_dir / "results.json").exists():
            logger.info(f"Loading results.json from {dataset_dir}")
            with open(dataset_dir / "results.json", "r", encoding="utf-8") as f:
                all_entries = json.load(f)
        else:
            json_files = sorted(dataset_dir.glob("output_*.json"))
            logger.info(f"Found {len(json_files)} legacy metadata files in {dataset_dir}")
            all_entries = []
            for json_file in json_files:
                logger.info(f"Loading {json_file.name}...")
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Normalize vienna_codes to always be a list
                    for entry in data:
                        codes = entry.get("vienna_codes", [])
                        if isinstance(codes, str):
                            entry["vienna_codes"] = [codes]
                        elif codes is None:
                            entry["vienna_codes"] = []
                    all_entries.extend(data)
    
    logger.info(f"Total entries loaded: {len(all_entries)}")
    return all_entries


def analyze_dataset(
    entries: List[Dict],
    images_dir: Path,
    checkpoint_path: Optional[Path] = None,
    resume: bool = False,
) -> Dict:
    """
    Analyze dataset with incremental checkpointing.
    
    Returns statistics dict with:
    - image_sizes: list of (width, height) tuples
    - aspect_ratios: list of aspect ratios
    - color_modes: count of RGB vs grayscale
    - vienna_code_distribution: Counter of level-2 codes
    - year_distribution: Counter of years
    - corrupted_images: list of corrupted image filenames
    - text_stats: % of images with text
    """
    stats = {
        "total_images": 0,
        "image_sizes": [],
        "aspect_ratios": [],
        "color_modes": Counter(),
        "vienna_code_distribution": Counter(),
        "year_distribution": Counter(),
        "corrupted_images": [],
        "with_text": 0,
        "without_text": 0,
    }
    
    # Load checkpoint if resuming
    processed_files = set()
    if resume and checkpoint_path and checkpoint_path.exists():
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path, "r") as f:
            checkpoint_stats = json.load(f)
            stats = checkpoint_stats["stats"]
            processed_files = set(checkpoint_stats["processed_files"])
            # Convert lists back from JSON
            stats["image_sizes"] = [tuple(s) for s in stats["image_sizes"]]
            stats["color_modes"] = Counter(stats["color_modes"])
            stats["vienna_code_distribution"] = Counter(stats["vienna_code_distribution"])
            stats["year_distribution"] = Counter(stats["year_distribution"])
            stats["corrupted_images"] = list(stats["corrupted_images"])
        logger.info(f"Resumed with {len(processed_files)} already processed")
    
    entries_to_process = [e for e in entries if e["file"] not in processed_files]
    
    logger.info(f"Analyzing {len(entries_to_process)} images...")
    
    for i, entry in enumerate(tqdm(entries_to_process, desc="Analyzing images")):
        filename = entry["file"]
        # Image files have .jpg extension (converted from .TIF/.JPG)
        image_name = Path(filename).stem + ".jpg"
        image_path = images_dir / image_name
        
        # Update metadata stats
        stats["total_images"] += 1
        
        # Text stats
        if entry.get("text"):
            stats["with_text"] += 1
        else:
            stats["without_text"] += 1
        
        # Vienna codes (use level 2: first two components)
        for code in entry.get("vienna_codes", []):
            level2 = ".".join(code.split(".")[:2])
            stats["vienna_code_distribution"][level2] += 1
        
        # Year distribution
        year = entry.get("year")
        if year:
            stats["year_distribution"][str(year)] += 1
        
        # Image analysis (if exists)
        if image_path.exists():
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    stats["image_sizes"].append((width, height))
                    stats["aspect_ratios"].append(width / height if height > 0 else 0)
                    
                    # Color mode
                    if img.mode == "L" or img.mode == "1":
                        stats["color_modes"]["grayscale"] += 1
                    else:
                        stats["color_modes"]["rgb"] += 1
                        
            except Exception as e:
                logger.warning(f"Corrupted image: {image_path} - {e}")
                stats["corrupted_images"].append(str(image_path))
        else:
            logger.debug(f"Image not found: {image_path}")
        
        processed_files.add(filename)
        
        # Save checkpoint periodically
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            logger.info(f"Saving checkpoint after {i + 1} images...")
            _save_checkpoint(checkpoint_path, stats, processed_files)
    
    # Final checkpoint
    if checkpoint_path:
        _save_checkpoint(checkpoint_path, stats, processed_files)
    
    return stats


def _save_checkpoint(checkpoint_path: Path, stats: Dict, processed_files: set):
    """Save checkpoint with atomic write."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Counter to dict for JSON serialization
    stats_copy = stats.copy()
    stats_copy["image_sizes"] = [list(s) for s in stats["image_sizes"]]
    stats_copy["color_modes"] = dict(stats["color_modes"])
    stats_copy["vienna_code_distribution"] = dict(stats["vienna_code_distribution"])
    stats_copy["year_distribution"] = dict(stats["year_distribution"])
    
    checkpoint = {
        "stats": stats_copy,
        "processed_files": list(processed_files),
    }
    
    # Atomic write
    tmp_path = checkpoint_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    tmp_path.rename(checkpoint_path)


def print_statistics(stats: Dict):
    """Print dataset statistics."""
    logger.info("=" * 60)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 60)
    
    logger.info(f"Total images: {stats['total_images']}")
    
    # Text statistics
    total_with_metadata = stats["with_text"] + stats["without_text"]
    if total_with_metadata > 0:
        text_pct = 100 * stats["with_text"] / total_with_metadata
        logger.info(f"Images with text: {stats['with_text']} ({text_pct:.1f}%)")
        logger.info(f"Images without text: {stats['without_text']} ({100-text_pct:.1f}%)")
    
    # Image size statistics
    if stats["image_sizes"]:
        widths = [s[0] for s in stats["image_sizes"]]
        heights = [s[1] for s in stats["image_sizes"]]
        logger.info(f"Image width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        logger.info(f"Image height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
    
    # Color mode
    if stats["color_modes"]:
        logger.info(f"Color modes: {dict(stats['color_modes'])}")
    
    # Vienna code distribution (top 10)
    logger.info("Top 10 Vienna codes (level 2):")
    for code, count in stats["vienna_code_distribution"].most_common(10):
        logger.info(f"  {code}: {count}")
    
    # Year distribution
    logger.info("Year distribution:")
    for year, count in sorted(stats["year_distribution"].items()):
        logger.info(f"  {year}: {count}")
    
    # Corrupted images
    if stats["corrupted_images"]:
        logger.warning(f"Corrupted images found: {len(stats['corrupted_images'])}")
    
    logger.info("=" * 60)


def create_stratified_splits(
    entries: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/val/test splits stratified by Vienna codes.
    
    Strategy: Group by primary Vienna code, then split each group.
    """
    logger.info("Creating stratified train/val/test splits...")
    
    random.seed(RANDOM_SEED)
    
    # Group entries by primary Vienna code (first code)
    groups = defaultdict(list)
    for entry in entries:
        codes = entry.get("vienna_codes", [])
        primary_code = codes[0] if codes else "unknown"
        level2 = ".".join(primary_code.split(".")[:2])
        groups[level2].append(entry)
    
    train_entries = []
    val_entries = []
    test_entries = []
    
    for code, group_entries in groups.items():
        # Shuffle within group
        random.shuffle(group_entries)
        
        n = len(group_entries)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Test gets remainder to ensure all images are used
        
        train_entries.extend(group_entries[:n_train])
        val_entries.extend(group_entries[n_train:n_train + n_val])
        test_entries.extend(group_entries[n_train + n_val:])
    
    # Shuffle again to mix codes
    random.shuffle(train_entries)
    random.shuffle(val_entries)
    random.shuffle(test_entries)
    
    logger.info(f"Train: {len(train_entries)} images")
    logger.info(f"Validation: {len(val_entries)} images")
    logger.info(f"Test: {len(test_entries)} images")
    
    return train_entries, val_entries, test_entries


def create_similarity_pairs(
    entries: List[Dict],
    n_pairs: int = 1000,
) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Create known-similar and known-dissimilar pairs for validation.
    
    Similar pairs: Same Vienna code
    Dissimilar pairs: Different Vienna codes
    """
    logger.info(f"Creating similarity pairs (n={n_pairs})...")
    
    random.seed(RANDOM_SEED)
    
    # Group by Vienna codes
    by_code = defaultdict(list)
    for entry in entries:
        for code in entry.get("vienna_codes", []):
            level2 = ".".join(code.split(".")[:2])
            by_code[level2].append(entry)
    
    # Filter codes with at least 2 images
    valid_codes = {code: entries for code, entries in by_code.items() if len(entries) >= 2}
    
    similar_pairs = []
    dissimilar_pairs = []
    
    # Create similar pairs
    for _ in range(n_pairs // 2):
        code = random.choice(list(valid_codes.keys()))
        group = valid_codes[code]
        if len(group) >= 2:
            a, b = random.sample(group, 2)
            similar_pairs.append((a["file"], b["file"], code))
    
    # Create dissimilar pairs
    all_codes = list(valid_codes.keys())
    for _ in range(n_pairs // 2):
        code1, code2 = random.sample(all_codes, 2)
        a = random.choice(valid_codes[code1])
        b = random.choice(valid_codes[code2])
        dissimilar_pairs.append((a["file"], b["file"], f"{code1}_vs_{code2}"))
    
    logger.info(f"Created {len(similar_pairs)} similar pairs")
    logger.info(f"Created {len(dissimilar_pairs)} dissimilar pairs")
    
    return similar_pairs, dissimilar_pairs


def save_splits(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
    output_dir: Path,
):
    """Save train/val/test splits to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train.json": train,
        "val.json": val,
        "test.json": test,
    }
    
    for filename, entries in splits.items():
        path = output_dir / filename
        # Atomic write
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(entries, f, indent=2)
        tmp_path.rename(path)
        logger.info(f"Saved {len(entries)} entries to {path}")


def save_validation_pairs(
    similar_pairs: List[Tuple],
    dissimilar_pairs: List[Tuple],
    output_dir: Path,
):
    """Save similarity pairs for validation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tuples to dicts for JSON
    similar = [
        {"image1": p[0], "image2": p[1], "vienna_code": p[2]}
        for p in similar_pairs
    ]
    dissimilar = [
        {"image1": p[0], "image2": p[1], "codes": p[2]}
        for p in dissimilar_pairs
    ]
    
    # Similar pairs
    path = output_dir / "similar_pairs.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(similar, f, indent=2)
    tmp_path.rename(path)
    logger.info(f"Saved {len(similar)} similar pairs to {path}")
    
    # Dissimilar pairs
    path = output_dir / "dissimilar_pairs.json"
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(dissimilar, f, indent=2)
    tmp_path.rename(path)
    logger.info(f"Saved {len(dissimilar)} dissimilar pairs to {path}")


def main():
    parser = argparse.ArgumentParser(description="Run EDA and create dataset splits")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip image analysis (use existing checkpoint)")
    parser.add_argument("--n-pairs", type=int, default=1000, help="Number of validation pairs to create")
    parser.add_argument("--toy", action="store_true", help="Use toy dataset")
    args = parser.parse_args()
    
    logger.info("Starting Phase 0: Data Analysis & Preparation")
    
    # Paths
    # Paths
    if args.toy:
        logger.info("Running in TOY mode")
        dataset_meta = TOY_DATASET_METADATA
        output_splits_dir = TOY_SPLITS_DIR
        output_val_dir = TOY_VALIDATION_DIR
        checkpoint_path = DATA_DIR / "toy_stats_checkpoint.json"
    else:
        dataset_meta = RAW_DATASET_DIR
        output_splits_dir = SPLITS_DIR
        output_val_dir = VALIDATION_DIR
        checkpoint_path = DATA_DIR / "stats_checkpoint.json"

    images_dir = RAW_DATASET_DIR / "images"

    # Check disk space
    if not args.toy: # Skip disk check for toy
        import shutil
        total, used, free = shutil.disk_usage(DATA_DIR)
        logger.info(f"Disk space: {free // (2**30)} GB free")
        if free < 10 * (2**30):  # Less than 10GB
            logger.warning("Low disk space! Consider freeing up space.")
    
    # Load metadata
    entries = load_all_metadata(dataset_meta)
    
    # Analyze dataset
    if not args.skip_analysis:
        stats = analyze_dataset(
            entries,
            images_dir,
            checkpoint_path=checkpoint_path,
            resume=args.resume,
        )
        print_statistics(stats)
    else:
        logger.info("Skipping analysis (using existing checkpoint)")
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
            stats = checkpoint["stats"]
        print_statistics(stats)
    
    # Create stratified splits
    # Create stratified splits
    train, val, test = create_stratified_splits(entries)
    save_splits(train, val, test, output_splits_dir)
    
    # Create validation pairs
    similar_pairs, dissimilar_pairs = create_similarity_pairs(val, n_pairs=args.n_pairs)
    save_validation_pairs(similar_pairs, dissimilar_pairs, output_val_dir)
    
    logger.info("Phase 0 complete!")


if __name__ == "__main__":
    main()
