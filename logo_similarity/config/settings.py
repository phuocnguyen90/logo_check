from .paths import *

# Preprocessing Settings
IMG_SIZE = 224
IMG_RESIZE = 256
TEXT_DETECTION_METHOD = "tesseract"  # Options: tesseract, craft, east
INPAINT_METHOD = "telea"            # Options: telea, ns
LRU_CACHE_SIZE = 1000

# Embedding Settings
MODEL_NAME = "efficientnet-b0"
EMBEDDING_DIM = 1280
REDUCED_DIM = 512
USE_PCA = False  # Set to False to use raw embeddings (1280 dim)

# Search Settings
TOP_K_GLOBAL = 1000
TOP_K_RERANK = 100

# MoCo v3 Hyperparameters (Optimized for High-VRAM GPUs)
BATCH_SIZE = 128
QUEUE_SIZE = 65536
MOMENTUM = 0.99
TEMPERATURE = 0.1
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 30
WARMUP_EPOCHS = 2
USE_AMP = True

# Composite Scoring Weights
WEIGHT_GLOBAL = 0.50
WEIGHT_SPATIAL = 0.35
WEIGHT_TEXT = 0.10
WEIGHT_COLOR = 0.05
