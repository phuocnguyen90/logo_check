import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDEXES_DIR = BASE_DIR / "indexes"

# Dataset Location (from plan)
RAW_DATASET_DIR = Path("/home/phuoc/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/")
DATASET_METADATA = RAW_DATASET_DIR / "results.json"

# Processed Data Paths
SPLITS_DIR = DATA_DIR / "splits"
VALIDATION_DIR = DATA_DIR / "validation"
LOGS_DIR = BASE_DIR / "logs"

# Model Paths
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_DIR = MODELS_DIR / "onnx"

# Index Paths
EMBEDDING_INDEX_DIR = INDEXES_DIR / "embeddings"

# Ensure directories exist
for path in [DATA_DIR, MODELS_DIR, INDEXES_DIR, SPLITS_DIR, VALIDATION_DIR, LOGS_DIR, CHECKPOINTS_DIR, ONNX_DIR, EMBEDDING_INDEX_DIR]:
    path.mkdir(parents=True, exist_ok=True)
