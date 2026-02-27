import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
INDEXES_DIR = BASE_DIR / "indexes"

# Dataset Location
RAW_DATASET_DIR = Path(os.getenv("RAW_DATASET_DIR", DATA_DIR / "raw"))
DATASET_METADATA = RAW_DATASET_DIR / "results.json"
TOY_DATASET_METADATA = DATA_DIR / "toy_results.json"
MASTER_METADATA_DB = DATA_DIR / "metadata_v2.db" # <--- Normalized master DB

# Processed Data Paths
SPLITS_DIR = DATA_DIR / "splits"
TOY_SPLITS_DIR = DATA_DIR / "toy_splits"
VALIDATION_DIR = DATA_DIR / "validation"
TOY_VALIDATION_DIR = DATA_DIR / "toy_validation" # <--- New
LOGS_DIR = BASE_DIR / "logs"

# Model Paths
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
ONNX_DIR = MODELS_DIR / "onnx"

# Index Paths
EMBEDDING_INDEX_DIR = INDEXES_DIR / "embeddings"

# Ensure directories exist
for path in [DATA_DIR, MODELS_DIR, INDEXES_DIR, SPLITS_DIR, VALIDATION_DIR, LOGS_DIR, CHECKPOINTS_DIR, ONNX_DIR, EMBEDDING_INDEX_DIR]:
    path.mkdir(parents=True, exist_ok=True)
