
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
    
    def __init__(self, metadata: List[Dict[str, Any]], transform=None):
        self.metadata = metadata
        self.transform = transform
        
        # Training config: Skip text removal to speed up training
        self.pipeline = PreprocessingPipeline(config={'skip_text_removal': True})
        
        # Indexing
        self.text_to_indices = {}
        self.vienna_to_indices = {}
        
        logger.info("Indexing dataset for semantic sampling...")
        for idx, item in tqdm(enumerate(metadata), total=len(metadata)):
            # Index by Text (Brand Name)
            text = item.get('text')
            if text and len(text) > 2: # Ignore tiny/empty strings
                norm_text = text.lower().strip()
                if norm_text not in self.text_to_indices:
                    self.text_to_indices[norm_text] = []
                self.text_to_indices[norm_text].append(idx)
            
            # Index by Vienna Code
            codes = item.get('vienna_codes', [])
            for code in codes:
                if code not in self.vienna_to_indices:
                    self.vienna_to_indices[code] = []
                self.vienna_to_indices[code].append(idx)
                
        # Filter for valid groups (must have > 1 image to sample a pair)
        self.valid_text_groups = [k for k, v in self.text_to_indices.items() if len(v) > 1]
        self.valid_vienna_groups = [k for k, v in self.vienna_to_indices.items() if len(v) > 1]
        
        logger.info(f"Found {len(self.valid_text_groups)} unique brand names with multiple logo variants.")
        logger.info(f"Found {len(self.valid_vienna_groups)} vienna codes with multiple variants.")

    def __len__(self):
        return len(self.metadata)

    def _load_image(self, idx):
        item = self.metadata[idx]
        original_name = item['file']
        
        # Robust loading: The JSON might have .JPG/.TIF but disk might be all .jpg
        candidates = [original_name]
        
        # Add fallback to .jpg (lowercase)
        stem = os.path.splitext(original_name)[0]
        candidates.append(f"{stem}.jpg")
        candidates.append(f"{stem}.JPG")
        
        # Deduplicate while preserving order
        candidates = list(dict.fromkeys(candidates))
        
        for name in candidates:
            path1 = paths.RAW_DATASET_DIR / "images" / name
            if path1.exists():
                return str(path1)
            path2 = paths.RAW_DATASET_DIR / name
            if path2.exists():
                return str(path2)
        return None

    def __getitem__(self, idx):
        # Retry logic if image load fails
        for _ in range(5):
            try:
                return self._getitem_safe(idx)
            except Exception:
                idx = random.randint(0, len(self.metadata) - 1)
        return self._getitem_safe(0) # Ultimate fallback

    def _getitem_safe(self, idx):
        # 1. Anchor Image
        img_path = self._load_image(idx)
        if not img_path: raise FileNotFoundError(f"Image not found for idx {idx}")
        
        prep1 = self.pipeline.process_on_the_fly(img_path)
        if not prep1: raise ValueError("Preprocessing failed")
        img1 = Image.fromarray(prep1.normalized)

        # 2. Pick Positive Strategy
        strategy = random.random()
        item = self.metadata[idx]
        pos_idx = idx # Default to self
        
        # Strategy A: Text Match (40%)
        found_match = False
        if strategy < 0.4:
            text = item.get('text', '')
            if text and len(text) > 2:
                norm_text = text.lower().strip()
                candidates = self.text_to_indices.get(norm_text, [])
                if len(candidates) > 1:
                    pos_idx = random.choice(candidates)
                    # Don't pick self if possible
                    if pos_idx == idx and len(candidates) > 1:
                        while pos_idx == idx:
                            pos_idx = random.choice(candidates)
                    found_match = True

        # Strategy B: Vienna Match (40% -> if in 0.4-0.8 range)
        if not found_match and strategy < 0.8:
            codes = item.get('vienna_codes', [])
            # Filter to codes that actually have matches
            valid_codes = [c for c in codes if c in self.vienna_to_indices and len(self.vienna_to_indices[c]) > 1]
            if valid_codes:
                code = random.choice(valid_codes)
                candidates = self.vienna_to_indices[code]
                pos_idx = random.choice(candidates)
                if pos_idx == idx and len(candidates) > 1:
                    while pos_idx == idx:
                        pos_idx = random.choice(candidates)
                found_match = True
        
        # Strategy C: Self-Augmentation (20% or fallback)
        # pos_idx stays as idx
        
        # 3. Load Positive
        if pos_idx == idx:
            img2 = img1.copy()
        else:
            pos_path = self._load_image(pos_idx)
            if pos_path:
                prep2 = self.pipeline.process_on_the_fly(pos_path)
                if prep2:
                    img2 = Image.fromarray(prep2.normalized)
                else:
                    img2 = img1.copy() # Fallback
            else:
                img2 = img1.copy() # Fallback

        # 4. Augment
        if self.transform:
            view1 = self.transform(img1)
            view2 = self.transform(img2)
        else:
            ts = transforms.ToTensor()
            view1 = ts(img1)
            view2 = ts(img2)
            
        return view1, view2


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
    
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True, 
        drop_last=True
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
