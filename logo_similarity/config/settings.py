import os
from .paths import *

# Environment-aware mode
IS_PRODUCTION = os.getenv("DEPLOY_ENV") == "production"

# Preprocessing Settings
IMG_SIZE = 224
IMG_RESIZE = 256
TEXT_DETECTION_METHOD = "tesseract"
INPAINT_METHOD = "telea"
LRU_CACHE_SIZE = 100

# Embedding Settings
MODEL_NAME = "efficientnet-b0"
EMBEDDING_DIM = 1280

# PCA Settings - Auto-detection is preferred, but these are defaults
REDUCED_DIM = 256
USE_PCA = True

# Search Settings
TOP_K_GLOBAL = 1000
TOP_K_RERANK = 100

# Composite Scoring Weights
WEIGHT_GLOBAL = 0.50
WEIGHT_SPATIAL = 0.35
WEIGHT_TEXT = 0.10
WEIGHT_COLOR = 0.05
