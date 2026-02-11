import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import json

from logo_similarity.config import settings, paths
from logo_similarity.utils.logging import logger
from logo_similarity.embeddings.efficientnet import EfficientNetEmbedder
from logo_similarity.training.dataset import TrademarkDataset, get_moco_augmentations
from logo_similarity.training.moco_trainer import MoCo
import argparse

def load_split(split_path):
    with open(split_path, "r") as f:
        return json.load(f)

def train():
    logger.info("Starting MoCo v3 Training...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", action="store_true", help="Run on toy dataset")
    args = parser.parse_args()

    # 1. Setup Data
    if args.toy:
        logger.info("Training on TOY dataset")
        train_path = paths.TOY_SPLITS_DIR / "train.json"
        val_path = paths.TOY_SPLITS_DIR / "val.json"
    else:
        train_path = paths.SPLITS_DIR / "train.json"
        val_path = paths.SPLITS_DIR / "val.json"

    if not train_path.exists():
        logger.error(f"Split file not found: {train_path}. Run scripts/01_run_eda.py first.")
        return

    train_metadata = load_split(train_path)
    # val_metadata = load_split(val_path) # Not used in MoCo loop currently but good to have

    # Subsampling removed for "mini-toy" representative test
    # if args.toy:
    #     logger.info("Toy mode: Subsampling training data to 2000 images for speed.")
    #     train_metadata = train_metadata[:2000]

    logger.info(f"Loaded {len(train_metadata)} training images")
    
    train_transform = get_moco_augmentations(settings.IMG_SIZE)
    train_dataset = TrademarkDataset(train_metadata, transform=train_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        drop_last=True
    )
    
    # 2. Setup Model
    base_encoder = EfficientNetEmbedder()
    model = MoCo(
        base_encoder, 
        dim=settings.EMBEDDING_DIM, 
        K=settings.QUEUE_SIZE, 
        m=settings.MOMENTUM, 
        T=settings.TEMPERATURE
    ).cuda()
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=settings.LEARNING_RATE, 
        weight_decay=settings.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss().cuda()
    scaler = GradScaler(enabled=settings.USE_AMP)
    
    # 3. Resumability
    checkpoint_path = paths.CHECKPOINTS_DIR / "latest.pth"
    start_epoch = 0
    best_loss = float('inf')
    
    if checkpoint_path.exists():
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            logger.info(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")

    # 4. Training Loop
    target_epochs = 2 if args.toy else settings.EPOCHS
    if args.toy:
        logger.info(f"Toy mode: limiting training to {target_epochs} epochs")
        
    for epoch in range(start_epoch, target_epochs):
        model.train()
        train_loss = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for im_q, im_k in pbar:
            im_q, im_k = im_q.cuda(non_blocking=True), im_k.cuda(non_blocking=True)
            
            optimizer.zero_grad()
            
            with autocast(enabled=settings.USE_AMP):
                logits, labels = model(im_q, im_k)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(loss.item())
            pbar.set_postfix(loss=np.mean(train_loss))
            
        avg_loss = np.mean(train_loss)
        logger.info(f"Epoch {epoch} complete. Avg Loss: {avg_loss:.4f}")
        
        # 5. Atomic Checkpointing
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_loss': best_loss,
            'queue': model.queue,
            'queue_ptr': model.queue_ptr
        }
        
        temp_ckpt = paths.CHECKPOINTS_DIR / "latest.tmp.pth"
        torch.save(checkpoint, temp_ckpt)
        temp_ckpt.replace(checkpoint_path)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(checkpoint, paths.CHECKPOINTS_DIR / "best_model.pth")
            logger.info(f"New best model saved with loss {best_loss:.4f}")

    logger.info("Training complete!")

if __name__ == "__main__":
    train()
