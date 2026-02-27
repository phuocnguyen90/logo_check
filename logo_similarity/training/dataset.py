import os
import json
import random
import torch
import sqlite3
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from ..utils.logging import logger
from ..config import paths, settings
from ..preprocessing.pipeline import PreprocessingPipeline

class TrademarkDataset(Dataset):
    """
    Dataset for MoCo v3 training.
    Uses Vienna codes as weak similarity signal for positive pair sampling.
    Supports both JSON list (legacy) and SQLite backend (scalable).
    """
    
    def __init__(
        self, 
        metadata: Optional[List[Dict[str, Any]]] = None, 
        transform: transforms.Compose = None,
        is_training: bool = True,
        use_instance_discrimination: bool = True,
        db_path: Optional[str] = None,
        split: Optional[str] = None
    ):
        self.metadata = metadata
        self.transform = transform
        self.is_training = is_training
        self.use_instance_discrimination = use_instance_discrimination
        self.db_path = db_path
        self.split = split
        
        # Training throughput optimization: skip compute-heavy text removal on-the-fly
        pipeline_config = {'skip_text_removal': is_training}
        self.pipeline = PreprocessingPipeline(config=pipeline_config)
        
        # 1. Initialize from Database (Preferred for scale)
        if self.db_path and self.split:
            logger.info(f"💾 Initializing TrademarkDataset from SQLite: {self.db_path} [split={self.split}]")
            self._init_db()
        # 2. Initialize from JSON List (Legacy)
        elif self.metadata is not None:
            logger.info(f"Initialized TrademarkDataset with {len(self.metadata)} images from memory list.")
            self._init_list()
        else:
            raise ValueError("Either metadata list or (db_path AND split) must be provided.")

    def _init_db(self):
        """Pre-fetch IDs and potentially Vienna code mappings from DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all IDs for this split
        cursor.execute("SELECT id FROM trademarks WHERE split = ?", (self.split,))
        self.ids = [row[0] for row in cursor.fetchall()]
        self.length = len(self.ids)
        
        # Build index of images by Vienna code for positive sampling (if weak supervision enabled)
        self.vienna_to_ids = {}
        if not self.use_instance_discrimination:
            logger.info("Building Vienna code index from DB (Weak Supervision Mode)...")
            cursor.execute("""
                SELECT tv.trademark_id, vc.code 
                FROM trademark_vienna tv
                JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
                JOIN trademarks t ON t.id = tv.trademark_id
                WHERE t.split = ?
            """, (self.split,))
            for tm_id, code in cursor.fetchall():
                if code not in self.vienna_to_ids:
                    self.vienna_to_ids[code] = []
                self.vienna_to_ids[code].append(tm_id)
            
            # Filter out codes with only one image
            self.active_codes = [c for c, ids in self.vienna_to_ids.items() if len(ids) > 1]
            logger.info(f"Found {len(self.active_codes)} active Vienna codes.")
            
        conn.close()

    def _init_list(self):
        """Legacy initialization from in-memory list."""
        self.ids = list(range(len(self.metadata)))
        self.length = len(self.metadata)
        
        self.vienna_to_indices = {}
        if not self.use_instance_discrimination:
            for idx, item in enumerate(self.metadata):
                codes = item.get('vienna_codes', [])
                for code in codes:
                    if code not in self.vienna_to_indices:
                        self.vienna_to_indices[code] = []
                    self.vienna_to_indices[code].append(idx)
            
            self.active_codes = [c for c, indices in self.vienna_to_indices.items() if len(indices) > 1]

    def __len__(self) -> int:
        return self.length

    def _get_item_from_db(self, tm_id: int) -> Dict[str, Any]:
        """Fetch a single trademark's data from DB."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT filename, tm_text, year FROM trademarks WHERE id = ?", (tm_id,))
        res = cursor.fetchone()
        
        codes = []
        if not self.use_instance_discrimination:
            cursor.execute("""
                SELECT vc.code FROM trademark_vienna tv
                JOIN vienna_codes vc ON vc.id = tv.vienna_code_id
                WHERE tv.trademark_id = ?
            """, (tm_id,))
            codes = [row[0] for row in cursor.fetchall()]
            
        conn.close()
        
        if not res: return None
        return {
            "file": res[0],
            "text": res[1],
            "year": res[2],
            "vienna_codes": codes
        }

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns two views of 'similar' images for MoCo v3.
        """
        # 1. Get metadata for this item
        if self.db_path:
            tm_id = self.ids[idx]
            item = self._get_item_from_db(tm_id)
        else:
            item = self.metadata[idx]
            
        if item is None:
            return self.__getitem__(random.randint(0, self.length - 1))
            
        img_path = paths.RAW_DATASET_DIR / "images" / item['file']
        
        # Case correction (metadata might have .JPG, disk has .jpg)
        if not img_path.exists():
            img_path = paths.RAW_DATASET_DIR / "images" / (Path(item['file']).stem + ".jpg")

        # 2. Load and process first image on-the-fly
        prep1 = self.pipeline.process_on_the_fly(str(img_path))
        if prep1 is None:
            return self.__getitem__(random.randint(0, self.length - 1))
        
        img1 = Image.fromarray(prep1.normalized)
        
        # 3. Sample positive pair
        img2 = None
        
        if self.is_training and not self.use_instance_discrimination:
            codes = item.get('vienna_codes', [])
            if self.db_path:
                valid_codes = [c for c in codes if c in self.vienna_to_ids and len(self.vienna_to_ids[c]) > 1]
            else:
                valid_codes = [c for c in codes if c in self.vienna_to_indices and len(self.vienna_to_indices[c]) > 1]
            
            if valid_codes:
                target_code = random.choice(valid_codes)
                if self.db_path:
                    pos_tm_id = random.choice(self.vienna_to_ids[target_code])
                    pos_item = self._get_item_from_db(pos_tm_id)
                else:
                    pos_idx = random.choice(self.vienna_to_indices[target_code])
                    pos_item = self.metadata[pos_idx]
                
                if pos_item:
                    pos_path = paths.RAW_DATASET_DIR / "images" / pos_item['file']
                    if not pos_path.exists():
                        pos_path = paths.RAW_DATASET_DIR / "images" / (Path(pos_item['file']).stem + ".jpg")
                    
                    prep2 = self.pipeline.process_on_the_fly(str(pos_path))
                    if prep2:
                        img2 = Image.fromarray(prep2.normalized)

        # Fallback to SSL (same image)
        if img2 is None:
            img2 = img1.copy()

        # 4. Apply augmentations
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
