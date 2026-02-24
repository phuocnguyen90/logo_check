
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json
import random
from typing import List, Dict, Any, Tuple
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from pathlib import Path

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.training.moco_trainer import MoCo
from logo_similarity.preprocessing.pipeline import PreprocessingPipeline
from logo_similarity.training.dataset import get_moco_augmentations

# --- 1. Enhanced Dataset for Semantic Alignment ---
class SemanticAlignmentDataset(torch.utils.data.Dataset):
    """
    Dataset for Phase 4: Semantic Alignment.
    
    Sampling Strategy for Positive Pairs:
    1. 40% Chance: Self-Augmentation (Instance Discrimination) - prevents catastrophic forgetting.
    2. 40% Chance: Same Text Label (Brand Name) - strongest semantic signal.
    3. 20% Chance: Same Vienna Code - weak semantic signal (fallback).
    """
    
    import cv2
    from collections import defaultdict

    def __init__(self, metadata, transform=None):
        self.metadata = metadata
        self.transform = transform
        
        # Pre-compute indices for fast lookup
        self.text_to_indices = defaultdict(list)
        self.vienna_to_indices = defaultdict(list)
        
        logger.info("Indexing dataset for semantic sampling...")
        for idx, item in tqdm(enumerate(metadata), total=len(metadata)):
            # Index by Text
            text = item.get('text')
            if text and len(text) > 2:
                norm_text = text.lower().strip()
                self.text_to_indices[norm_text].append(idx)
            
            # Index by Vienna Code
            for code in item.get('vienna_codes', []):
                self.vienna_to_indices[code].append(idx)
                
        # Filter for valid groups
        self.valid_text_groups = [k for k, v in self.text_to_indices.items() if len(v) > 1]
        self.valid_vienna_groups = [k for k, v in self.vienna_to_indices.items() if len(v) > 1]
        
        logger.info(f"Unique Brands: {len(self.valid_text_groups)}, Unique Vienna: {len(self.valid_vienna_groups)}")

    def __len__(self):
        return len(self.metadata)

    def _load_and_preprocess(self, idx):
        """Fast path loader optimized for training throughput."""
        item = self.metadata[idx]
        original_name = item['file']
        
        # 1. Resolve Path (Fastest checks first)
        base_dir = paths.RAW_DATASET_DIR
        # Check standard path
        path = base_dir / "images" / original_name
        
        # Fallback 1: Flat directory
        if not path.exists():
             path = base_dir / original_name
             
        # Fallback 2: Extension swap
        if not path.exists():
            stem = os.path.splitext(original_name)[0]
            path = base_dir / "images" / f"{stem}.jpg"
            if not path.exists():
                path = base_dir / f"{stem}.jpg"
        
        if not path.exists():
            return None

        try:
            # 2. Load with OpenCV
            img = cv2.imread(str(path))
            if img is None: return None
            
            # 3. Normalize (Resize + Pad)
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w = img.shape[:2]
            target_size = settings.IMG_SIZE
            scale = target_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            if scale != 1.0:
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas
            canvas = np.full((target_size, target_size, 3), 255, dtype=np.uint8)
            y_off = (target_size - new_h) // 2
            x_off = (target_size - new_w) // 2
            canvas[y_off:y_off+new_h, x_off:x_off+new_w] = img
            
            # 4. To PIL
            return Image.fromarray(canvas)
            
        except Exception:
            return None

    def __getitem__(self, idx):
        # 1. Anchor Image
        img_q = self._load_and_preprocess(idx)
        
        # Retry logic
        attempts = 0
        while img_q is None and attempts < 5:
            idx = random.randint(0, len(self.metadata) - 1)
            img_q = self._load_and_preprocess(idx)
            attempts += 1
            
        if img_q is None: # Ultimate fallback
             img_q = Image.new('RGB', (settings.IMG_SIZE, settings.IMG_SIZE), (255, 255, 255))
        
        item = self.metadata[idx]
        
        # 2. Pick Positive Strategy
        choice = random.random()
        pos_idx = idx 
        
        # Strategy A: Text Match (25%)
        # Reduced from 40% to prevent OCR overfitting
        if choice < 0.25:
            raw_text = item.get('text')
            txt = raw_text.lower().strip() if raw_text else ""
            if txt and txt in self.text_to_indices:
                candidates = self.text_to_indices[txt]
                if len(candidates) > 1:
                    pos_idx = random.choice(candidates)
                    
        # Strategy B: Vienna Match (45%)
        # Increased to emphasize shape/concept
        elif choice < 0.70:
            codes = item.get('vienna_codes', [])
            valid_codes = [c for c in codes if c in self.vienna_to_indices and len(self.vienna_to_indices[c]) > 1]
            if valid_codes:
                code = random.choice(valid_codes)
                candidates = self.vienna_to_indices[code]
                pos_idx = random.choice(candidates)

        # Strategy C: Self-Augmentation (20% or fallback)
        # pos_idx stays as idx
        
        # 3. Load Positive
        if pos_idx == idx:
            img_k = img_q 
        else:
            img_k = self._load_and_preprocess(pos_idx)
            if img_k is None:
                img_k = img_q # Fallback
        
        # 4. Augment
        if self.transform:
            q = self.transform(img_q)
            k = self.transform(img_k)
            return q, k
            
        # Fallback transforms if none provided
        ts = transforms.ToTensor()
        return ts(img_q), ts(img_k)


# --- 2. Training Script ---
import argparse

def train_semantic_alignment():
    parser = argparse.ArgumentParser(description="Phase 4b: Semantic Alignment Training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--name", type=str, default="v1", help="Experiment name/version tag (e.g. v1, v2)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (optional)")
    args = parser.parse_args()

    logger.info(f"Initializing Phase 4b: Semantic Alignment ({args.name})")
    logger.info(f"Config: Epochs={args.epochs}, LR={args.lr}, Batch={args.batch_size}")
    
    # 1. Setup Data
    train_path = paths.SPLITS_DIR / "train.json"
    if not train_path.exists():
        if (paths.TOY_SPLITS_DIR / "train.json").exists():
            train_path = paths.TOY_SPLITS_DIR / "train.json"
        else:
            logger.error("No training split found.")
            return

    with open(train_path, 'r') as f:
        metadata = json.load(f)
        
    logger.info(f"Loaded {len(metadata)} images.")
    
    transform = get_moco_augmentations(settings.IMG_SIZE)
    dataset = SemanticAlignmentDataset(metadata, transform=transform)
    
    
    # Optimize Data Loading Hardware Usage
    num_cpus = os.cpu_count() or 8
    num_workers = min(32, num_cpus) # Cap at 32
    logger.info(f"Using {num_workers} DataLoader workers (detected {num_cpus} CPUs)")

    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True, 
        drop_last=True,
        persistent_workers=True, # Keep workers alive between epochs
        prefetch_factor=2        # Buffer batches
    )
    
    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_encoder = EfficientNetEmbedder()
    
    model = MoCo(
        base_encoder,
        dim=settings.EMBEDDING_DIM,
        K=settings.QUEUE_SIZE, 
        m=settings.MOMENTUM,
        T=settings.TEMPERATURE
    ).to(device)
    
    # Load Weights
    # Priority: Explicit Resume -> Semantic Latest -> Pre-trained Best -> Pre-trained Latest
    ckpt_path = None
    if args.resume:
        ckpt_path = Path(args.resume)
    elif (paths.CHECKPOINTS_DIR / f"semantic_{args.name}_latest.pth").exists():
        ckpt_path = paths.CHECKPOINTS_DIR / f"semantic_{args.name}_latest.pth"
        logger.info("Found existing semantic checkpoint for this version. Resuming.")
    elif (paths.CHECKPOINTS_DIR / "best_model.pth").exists():
        ckpt_path = paths.CHECKPOINTS_DIR / "best_model.pth"
        logger.info("Starting fresh semantic alignment from MOCO BEST model.")
    else:
        ckpt_path = paths.CHECKPOINTS_DIR / "latest.pth"
        
    start_epoch = 1
    if ckpt_path and ckpt_path.exists():
        logger.info(f"Loading weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        
        # Handle state dict keys
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        
        # Restore Queue if available
        if 'queue' in checkpoint:
            model.queue = checkpoint['queue'].to(device)
            model.queue_ptr = checkpoint['queue_ptr'].to(device)
            logger.info("Restored MoCo negative queue state.")
            
        # If resuming same run, load epoch
        if args.resume or "semantic" in ckpt_path.name:
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Resuming from Epoch {start_epoch}")
    else:
        logger.warning("No checkpoint found! Training from random initialization (NOT RECOMMENDED).")

    # 3. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=settings.WEIGHT_DECAY
    )
    
    scaler = GradScaler(enabled=settings.USE_AMP)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Training Loop
    model.train()
    
    for epoch in range(start_epoch, args.epochs + 1):
        loss_epoch = []
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} [{args.name}]")
        
        for im_q, im_k in pbar:
            im_q, im_k = im_q.to(device, non_blocking=True), im_k.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(enabled=settings.USE_AMP):
                logits, labels = model(im_q, im_k)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loss_epoch.append(loss.item())
            pbar.set_postfix(loss=np.mean(loss_epoch))
            
        avg_loss = np.mean(loss_epoch)
        logger.info(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}")
        
        # Metadata for Version Control
        training_metadata = {
            'version': args.name,
            'method': 'semantic_alignment_phase4b',
            'base_lr': args.lr,
            'sampling_strategy': '40% Text, 40% Vienna, 20% Instance',
            'parent_checkpoint': str(ckpt_path) if ckpt_path else 'none'
        }

        # Save Checkpoint
        save_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'queue': model.queue,
            'queue_ptr': model.queue_ptr,
            'training_metadata': training_metadata
        }
        
        # Save "latest" for this version
        torch.save(save_dict, paths.CHECKPOINTS_DIR / f"semantic_{args.name}_latest.pth")
        
        # Save epoch snapshot
        if epoch % 5 == 0 or epoch == args.epochs:
             torch.save(save_dict, paths.CHECKPOINTS_DIR / f"semantic_{args.name}_epoch_{epoch}.pth")

if __name__ == "__main__":
    # Fix torch serialization for numpy scalars
    try:
        import torch.serialization
        torch.serialization.add_safe_globals([
            np.float64, np.int64,
            np.core.multiarray.scalar
        ])
    except:
        pass
        
    train_semantic_alignment()
