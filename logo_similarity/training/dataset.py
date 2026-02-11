import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Any, Tuple
from ..utils.logging import logger
from ..config import paths, settings
from ..preprocessing.pipeline import PreprocessingPipeline

class TrademarkDataset(Dataset):
    """
    Dataset for MoCo v3 training.
    Uses Vienna codes as weak similarity signal for positive pair sampling.
    """
    
    def __init__(
        self, 
        metadata: List[Dict[str, Any]], 
        transform: transforms.Compose = None,
        is_training: bool = True
    ):
        self.metadata = metadata
        self.transform = transform
        self.is_training = is_training
        self.pipeline = PreprocessingPipeline()
        
        # Build index of images by Vienna code for positive sampling
        self.vienna_to_indices = {}
        for idx, item in enumerate(metadata):
            codes = item.get('vienna_codes', [])
            for code in codes:
                if code not in self.vienna_to_indices:
                    self.vienna_to_indices[code] = []
                self.vienna_to_indices[code].append(idx)
        
        # Filter out codes with only one image (can't sample positive pair)
        self.active_codes = [c for c, indices in self.vienna_to_indices.items() if len(indices) > 1]
        
        logger.info(f"Initialized TrademarkDataset with {len(metadata)} images.")
        logger.info(f"Found {len(self.active_codes)} Vienna codes suitable for positive sampling.")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two views of 'similar' images for MoCo v3.
        View 1: Current image (idx)
        View 2: Another image sharing at least one Vienna code, or a second view of the same image.
        """
        item = self.metadata[idx]
        img_path = paths.RAW_DATASET_DIR / "images" / item['file']
        
        # 1. Load and process first image on-the-fly
        prep1 = self.pipeline.process_on_the_fly(str(img_path))
        if prep1 is None:
            # Fallback to random image if processing fails
            return self.__getitem__(random.randint(0, len(self.metadata) - 1))
        
        img1 = Image.fromarray(prep1.normalized)
        
        # 2. Sample positive pair
        codes = item.get('vienna_codes', [])
        valid_codes = [c for c in codes if c in self.vienna_to_indices and len(self.vienna_to_indices[c]) > 1]
        
        if self.is_training and valid_codes:
            # Sample another image sharing a Vienna code
            target_code = random.choice(valid_codes)
            pos_idx = random.choice(self.vienna_to_indices[target_code])
            # Ensure we don't pick the same index optionally, 
            # though two views of same image is also valid contrastive learning.
            
            pos_item = self.metadata[pos_idx]
            pos_path = paths.RAW_DATASET_DIR / "images" / pos_item['file']
            prep2 = self.pipeline.process_on_the_fly(str(pos_path))
            
            if prep2 is None:
                img2 = img1.copy() # Fallback to second view of same image
            else:
                img2 = Image.fromarray(prep2.normalized)
        else:
            # For validation or if no Vienna codes, use two views of the same image
            img2 = img1.copy()

        # 3. Apply augmentations
        if self.transform:
            view1 = self.transform(img1)
            view2 = self.transform(img2)
        else:
            to_tensor = transforms.ToTensor()
            view1 = to_tensor(img1)
            view2 = to_tensor(img2)
            
        return view1, view2

def get_moco_augmentations(img_size: int = 224) -> transforms.Compose:
    """Strong augmentations for contrastive learning."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
