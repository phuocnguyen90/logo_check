# Logo Similarity Detection — Detailed Implementation Plan

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Project Context](#project-context)
3. [Technical Architecture](#technical-architecture)
4. [Environment Setup](#environment-setup)
5. [Phase 0: Data Analysis & Preparation](#phase-0-data-analysis--preparation)
6. [Phase 1: Preprocessing Pipeline](#phase-1-preprocessing-pipeline)
7. [Phase 2: Stage 1 - Global Embedding Retrieval](#phase-2-stage-1---global-embedding-retrieval)
8. [Phase 3: Stage 2 - Re-ranking with Local Features](#phase-3-stage-2---re-ranking-with-local-features)
9. [Phase 4: Fine-tuning with Contrastive Learning](#phase-4-fine-tuning-with-contrastive-learning)
10. [Phase 5: Composite Mark Integration](#phase-5-composite-mark-integration)
11. [Phase 6: Production Deployment](#phase-6-production-deployment)
12. [Testing & Validation](#testing--validation)
13. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

**Goal**: Build a production-ready logo similarity search system for trademark images.

**Timeline**: 8-10 weeks

**Key Deliverables**:
- Working logo search with deep embeddings (Week 2)
- Re-ranking with spatial features (Week 3.5)
- Fine-tuned model with improved accuracy (Week 6.5)
- Full composite mark support (Week 7.5)
- Production-ready API (Week 8.5)

---

## Project Context

### Dataset Information
- **Location**: `/home/phuoc/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/`
- **Total Images**: 769,674 trademark logos
- **Format**: JPG images with JSON metadata
- **Metadata Structure**:
  ```json
  {
    "file": "uuid.jpg",
    "text": "BRAND_NAME" | null,  // 90.1% have text
    "vienna_codes": ["27.05.01"],
    "year": 2016
  }
  ```

### Environment
- **Python Environment**: `torch12` (conda/mamba env)
- **CUDA**: Version 12 preinstalled
- **GPU**: RTX 3060 (12GB VRAM) - **Critical constraint for training**
- **System RAM**: 32GB
- **Working Directory**: `/home/phuoc/git/l3d/tm-dataset/`

### Hardware Considerations

#### GPU Memory Constraints (RTX 3060 - 12GB VRAM)
The primary bottleneck is **Phase 4 (Fine-tuning)** with contrastive learning:

| Configuration | VRAM Usage | Fits on 3060? |
|---------------|------------|---------------|
| InfoNCE, batch_size=512 | ~8-10GB | **Yes, tight** |
| InfoNCE, batch_size=256 | ~4-5GB | Yes |
| InfoNCE, batch_size=64 | ~1.5GB | Yes |

**Why batch size matters for InfoNCE**: InfoNCE quality degrades with small batches because you get fewer in-batch negatives. The loss expects each anchor to compare against many other examples in the batch.

#### Strategies for Limited VRAM

**Option 1: Gradient Accumulation (Recommended)**
```python
# Effective batch 512 = 8 accumulation steps × 64 actual batch
accumulation_steps = 8
actual_batch_size = 64  # Fits in ~1.5GB VRAM

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Option 2: MoCo v3 (Alternative to InfoNCE)**
MoCo (Momentum Contrast) maintains a momentum-updated queue of negatives (e.g., 65536 samples), allowing:
- Small actual batch size (64)
- Massive effective negative pool
- Designed specifically for limited VRAM scenarios

```python
# MoCo v3 setup
queue_size = 65536  # Huge negative pool
batch_size = 64     # Fits comfortably
momentum = 0.999    # For momentum encoder
```

#### Batch Indexing Performance Estimates

Full TMView dataset indexing (~770K images):

| Metric | Value |
|--------|-------|
| Per-image inference time | ~50ms (RTX 3060) |
| Total time for 770K images | ~14 hours |
| Optimal batch size | 32-64 (maximize GPU utilization) |
| Incremental saving | Required (resume capability) |

**Indexing strategy**:
```python
# Process in chunks, save incrementally
chunk_size = 10000
for i, chunk in enumerate(chunks):
    embeddings = process_batch(chunk, batch_size=64)
    save_embeddings(f'embeddings_chunk_{i}.npz')
    # If crash at 500K, resume from checkpoint
```

#### Vector Index Memory Requirements

| Dataset Size | Vectors | Dim | Raw Data | HNSW Overhead | Total RAM Needed | Fits in 32GB? |
|--------------|---------|-----|----------|---------------|------------------|---------------|
| Current | 770K | 512 | ~1.5GB | 3-4x | ~6GB | **Yes** |
| 2M | 2M | 512 | ~4GB | 3-4x | ~16GB | **Yes** |
| 3M+ | 3M+ | 512 | ~6GB+ | 3-4x | ~24GB+ | Borderline |

**If exceeding 2M vectors**: Switch to IVFFlat index (less RAM during build, slightly worse recall).

#### FP16 Inference Optimization

**Critical for production**: RTX 3060 supports native FP16 (half precision):
- **Memory**: Halves VRAM usage (512-dim float32 → 256-dim float16 equivalent)
- **Speed**: Actually faster due to Tensor Core acceleration
- **Accuracy**: No loss for inference (only gradients matter for training)

```python
# FP16 ONNX export
def export_to_onnx_fp16(model, output_path):
    from onnxconverter_common import float16
    # Export in FP16 for production inference
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, output_path)
```

**ONNX Runtime with FP16**:
```python
import onnxruntime as ort

session = ort.InferenceSession(
    'model_fp16.onnx',
    providers=['CUDAExecutionProvider'],
    provider_options=[{'device_id': 0}]
)
# Automatic FP16 inference on RTX 3060
```

---

## Technical Architecture

### System Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                          Logo Similarity System                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Input      │───▶│ Preprocess   │───▶│  Stage 1:    │           │
│  │   Logo       │    │  (OCR+Mask)  │    │  ANN Search  │           │
│  └──────────────┘    └──────────────┘    └──────┬───────┘           │
│                                                  │                   │
│                                                  │ Top-1000          │
│                                                  ▼                   │
│                                         ┌──────────────┐            │
│                                         │  Stage 2:    │            │
│                                         │  Re-ranking  │            │
│                                         └──────┬───────┘            │
│                                                │                     │
│                                                ▼                     │
│                                         ┌──────────────┐            │
│                                         │  Composite   │            │
│                                         │   Scoring    │            │
│                                         └──────┬───────┘            │
│                                                │                     │
│                                                ▼                     │
│                                         ┌──────────────┐            │
│                                         │   Results    │            │
│                                         └──────────────┘            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Directory Structure (To Be Created)
```
/home/phuoc/git/l3d/tm-dataset/
├── logo_similarity/              # Main package
│   ├── __init__.py
│   ├── config/                   # Configuration files
│   │   ├── __init__.py
│   │   ├── settings.py           # Main settings
│   │   └── paths.py              # Path configurations
│   ├── preprocessing/            # Stage 0: Preprocessing
│   │   ├── __init__.py
│   │   ├── text_detector.py      # OCR text detection
│   │   ├── text_masker.py        # Text masking/inpainting
│   │   ├── image_normalizer.py   # Resize, normalize
│   │   └── pipeline.py           # Preprocessing pipeline
│   ├── embeddings/               # Stage 1: Embeddings
│   │   ├── __init__.py
│   │   ├── efficientnet.py       # EfficientNet model wrapper
│   │   ├── onnx_exporter.py      # Export to ONNX
│   │   ├── embedder.py           # Embedding generation
│   │   └── pca_reducer.py        # Dimensionality reduction
│   ├── retrieval/                # Stage 1: ANN Search
│   │   ├── __init__.py
│   │   ├── vector_store.py       # pgvector wrapper
│   │   ├── index_builder.py      # Build ANN index
│   │   └── searcher.py           # Query interface
│   ├── reranking/                # Stage 2: Re-ranking
│   │   ├── __init__.py
│   │   ├── spatial_matcher.py    # Spatial feature matching
│   │   ├── scoring.py            # Composite scoring
│   │   └── reranker.py           # Re-ranking pipeline
│   ├── training/                 # Offline training
│   │   ├── __init__.py
│   │   ├── dataset.py            # Training dataset
│   │   ├── contrastive.py        # InfoNCE contrastive loss
│   │   ├── trainer.py            # Training loop
│   │   └── finetune_efficientnet.py  # Fine-tuning script
│   ├── text/                     # Text similarity
│   │   ├── __init__.py
│   │   ├── phonetic.py           # Phonetic similarity
│   │   └── string_matcher.py     # String similarity
│   ├── utils/                    # Utilities
│   │   ├── __init__.py
│   │   ├── image_utils.py        # Image utilities
│   │   ├── data_loader.py        # Data loading
│   │   └── metrics.py            # Evaluation metrics
│   └── api/                      # Production API (optional)
│       ├── __init__.py
│       └── routes.py             # API endpoints
├── scripts/                      # Standalone scripts
│   ├── 01_analyze_dataset.py
│   ├── 02_preprocess_all.py
│   ├── 03_build_index.py
│   ├── 04_train_model.py
│   └── 05_export_onnx.py
├── tests/                        # Tests
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_retrieval.py
│   └── test_reranking.py
├── notebooks/                    # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_model_training.ipynb
├── data/                         # Local data (symlinks to cache)
│   └── dataset -> ~/.cache/kagglehub/...
├── models/                       # Saved models
│   ├── checkpoints/              # Training checkpoints
│   └── onnx/                     # ONNX exports
└── indexes/                      # Vector indexes
    └── embeddings/
```

---

## Environment Setup

### Step 1: Activate Environment
```bash
mamba activate torch12
```

### Step 2: Create Directory Structure
```bash
cd /home/phuoc/git/l3d/tm-dataset/
mkdir -p logo_similarity/{config,preprocessing,embeddings,retrieval,reranking,training,text,utils,api}
mkdir -p scripts tests notebooks models/{checkpoints,onnx} indexes/embeddings
touch logo_similarity/__init__.py
touch logo_similarity/{config,preprocessing,embeddings,retrieval,reranking,training,text,utils,api}/__init__.py
```

### Step 3: Install Dependencies
```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install efficientnet-pytorch
pip install onnx onnxruntime
pip install pillow opencv-python-headless
pip install pytesseract
pip install numpy pandas scipy
pip install scikit-learn
pip install tqdm
pip install matplotlib seaborn
pip install jupyter

# OCR dependencies
sudo apt-get install tesseract-ocr
pip install pytesseract

# For database (later)
# pip install psycopg2-binary pgvector
```

### Step 4: Verify Dataset Access
```bash
python3 -c "
import json
data = json.load(open('/home/phuoc/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/results.json'))
print(f'Dataset loaded: {len(data):,} images')
"
```

---

## Phase 0: Data Analysis & Preparation

**Duration**: 2-3 days
**Goal**: Understand dataset characteristics and prepare train/val/test splits

### Tasks

#### Task 0.1: Dataset Statistics Script
**File**: `scripts/01_analyze_dataset.py`

**Objective**: Generate comprehensive dataset statistics

**Output**:
- Image size distribution (min, max, mean, median)
- Aspect ratio distribution
- Color vs grayscale distribution
- Text presence statistics (already known: 90.1%)
- Vienna code distribution
- Year distribution
- Sample images visualization

**Key Questions to Answer**:
1. What percentage of images are smaller than 224x224? (will need upscaling)
2. What is the average aspect ratio? (affects preprocessing strategy)
3. How many unique Vienna codes exist? (for weak supervision)
4. Are there corrupted images?

#### Task 0.2: Create Train/Val/Test Splits
**File**: `logo_similarity/utils/splits.py`

**Strategy**:
- **Train**: 70% (538,771 images)
- **Validation**: 15% (115,451 images)
- **Test**: 15% (115,452 images)

**Split Method**:
- Stratified by Vienna codes (ensure distribution similarity)
- Year-based split: Train on 1996-2017, Val/Test on 2018-2020 (temporal split)

**Output**: `data/splits/train.json`, `data/splits/val.json`, `data/splits/test.json`

#### Task 0.3: Create Known-Similar Pairs for Validation
**File**: `logo_similarity/utils/val_pairs.py`

**Strategy**:
- Use Vienna codes as weak similarity signal
- For each Vienna code, sample pairs of trademarks sharing that code
- Create negative pairs (different Vienna codes)

**Output**: `data/validation/similar_pairs.json`, `data/validation/dissimilar_pairs.json`

---

## Phase 1: Preprocessing Pipeline

**Duration**: 4-5 days
**Goal**: Normalize input images and remove text to isolate figurative elements

### Architecture
```
Input Image (276KB avg)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  1. Text Detection (EAST or CRAFT)                          │
│     - Detect text regions                                   │
│     - Return bounding boxes                                 │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Text Masking (OpenCV inpainting)                        │
│     - Create binary mask from text boxes                    │
│     - Inpaint masked regions                                │
│     - Output: masked image                                  │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Normalization                                           │
│     - Resize to 256x256 (maintain aspect ratio, pad)        │
│     - Center crop to 224x224                                │
│     - Convert to RGB                                        │
│     - White background for transparent images               │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Binarization (optional)                                 │
│     - Grayscale conversion                                  │
│     - Otsu thresholding                                     │
│     - Output: structure-focused version                    │
└─────────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  Output: Two versions per image                             │
│     - original.npy: 224x224x3 float32 (color)               │
│     - binary.npy: 224x224 float32 (structure)               │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### Task 1.1: Text Detection Module
**File**: `logo_similarity/preprocessing/text_detector.py`

**Options**:
1. **Tesseract OCR** (simpler, good for Latin text)
2. **CRAFT** (more accurate, deep learning-based)
3. **EAST** (fast, deep learning-based)

**Recommendation**: Start with Tesseract for baseline, upgrade to CRAFT if needed

**Implementation**:
```python
class TextDetector:
    def __init__(self, method='tesseract'):
        ...

    def detect_text(self, image_path: str) -> List[BoundingBox]:
        """Return list of (x, y, w, h) bounding boxes"""

    def visualize_detections(self, image_path: str, output_path: str):
        """Save image with text boxes drawn"""
```

#### Task 1.2: Text Masking Module
**File**: `logo_similarity/preprocessing/text_masker.py`

**Implementation**:
```python
class TextMasker:
    def __init__(self, inpaint_method='telea'):
        ...

    def mask_text(self, image: np.ndarray, boxes: List[BoundingBox]) -> np.ndarray:
        """Inpaint text regions, return masked image"""

    def create_binary_mask(self, image_shape, boxes: List[BoundingBox]) -> np.ndarray:
        """Create binary mask for visualization"""
```

#### Task 1.3: Image Normalization Module
**File**: `logo_similarity/preprocessing/image_normalizer.py`

**Implementation**:
```python
class ImageNormalizer:
    def __init__(self, target_size=224, background='white'):
        ...

    def normalize(self, image_path: str) -> Dict[str, np.ndarray]:
        """Return dict with 'color' and 'binary' versions"""
```

#### Task 1.4: Preprocessing Pipeline
**File**: `logo_similarity/preprocessing/pipeline.py`

**Implementation**:
```python
class PreprocessingPipeline:
    def __init__(self, config):
        self.text_detector = TextDetector(...)
        self.text_masker = TextMasker(...)
        self.normalizer = ImageNormalizer(...)

    def process(self, image_path: str) -> PreprocessedImage:
        """Run full pipeline, return preprocessed result"""

    def process_batch(self, image_paths: List[str]) -> List[PreprocessedImage]:
        """Process multiple images in parallel"""
```

#### Task 1.5: Batch Preprocessing Script
**File**: `scripts/02_preprocess_all.py`

**Features**:
- Progress bar with tqdm
- Multiprocessing for speed
- Save results to `data/preprocessed/`
- Resume capability (skip already processed)
- Error logging

**Output Structure**:
```
data/preprocessed/
├── 00000434-64ed-4841-974f-96b3c7c5b369.npz  # Contains: color, binary, mask
├── ...
└── manifest.json  # List of processed files with metadata
```

---

## Phase 2: Stage 1 - Global Embedding Retrieval

**Duration**: 5-6 days
**Goal**: Build efficient ANN search using pre-trained EfficientNet embeddings

### Architecture
```
Preprocessed Image
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  1. EfficientNet-B0 Feature Extractor                        │
│     - Input: 224x224x3 image                                 │
│     - Output: 1280-dim global feature vector                 │
│     - Pre-trained ImageNet weights                          │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  2. PCA Dimensionality Reduction                            │
│     - Input: 1280-dim vector                                 │
│     - Output: 512-dim vector                                 │
│     - Fit on trademark dataset                              │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  3. ANN Index (HNSW)                                        │
│     - Fast approximate nearest neighbor search               │
│     - Query: ~100ms for top-1000                            │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
    Top-1000 Candidates
```

### Tasks

#### Task 2.1: EfficientNet Wrapper
**File**: `logo_similarity/embeddings/efficientnet.py`

**Implementation**:
```python
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetEmbedder(nn.Module):
    def __init__(self, model_name='efficientnet-b0', pretrained=True):
        super().__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        # Remove classification head
        self.model._fc = nn.Identity()

    def forward(self, x):
        # x: [B, 3, 224, 224]
        features = self.model.extract_features(x)  # [B, 1280, 7, 7]
        pooled = torch.nn.functional.adaptive_avg_pool2d(features, 1)  # [B, 1280, 1, 1]
        return pooled.squeeze(-1).squeeze(-1)  # [B, 1280]

    def get_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for single image"""
```

#### Task 2.2: PCA Reducer
**File**: `logo_similarity/embeddings/pca_reducer.py`

**Implementation**:
```python
from sklearn.decomposition import PCA

class PCAReducer:
    def __init__(self, n_components=512):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, embeddings: np.ndarray):
        """Fit PCA on dataset embeddings"""
        self.pca.fit(embeddings)
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

    def transform(self, embedding: np.ndarray) -> np.ndarray:
        """Reduce single embedding"""
        return self.pca.transform(embedding.reshape(1, -1))[0]

    def save(self, path: str):
        """Save PCA model"""
        joblib.dump(self.pca, path)

    def load(self, path: str):
        """Load PCA model"""
        self.pca = joblib.load(path)
```

#### Task 2.3: Vector Store (pgvector wrapper)
**File**: `logo_similarity/retrieval/vector_store.py`

**Options**:
1. **FAISS** (simpler, in-memory)
2. **pgvector** (production-ready, persistent)
3. **HNSWlib** (fast, in-memory)

**Recommendation**: Start with FAISS for development, migrate to pgvector for production

**Implementation (FAISS version)**:
```python
import faiss
import numpy as np

class VectorStore:
    def __init__(self, dimension: int, index_type='hnsw'):
        self.dimension = dimension
        self.index = self._create_index(index_type)
        self.ids = []

    def _create_index(self, index_type):
        if index_type == 'hnsw':
            return faiss.IndexHNSWFlat(self.dimension, 32)
        elif index_type == 'ivf':
            quantizer = faiss.IndexFlatL2(self.dimension)
            return faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            return faiss.IndexFlatL2(self.dimension)

    def add(self, embeddings: np.ndarray, ids: List[str]):
        """Add embeddings to index"""
        self.index.add(embeddings.astype('float32'))
        self.ids.extend(ids)

    def search(self, query: np.ndarray, k: int = 1000) -> List[Tuple[str, float]]:
        """Search for top-k similar vectors"""
        distances, indices = self.index.search(query.astype('float32').reshape(1, -1), k)
        return [(self.ids[i], float(d)) for i, d in zip(indices[0], distances[0])]

    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, path)
        with open(f'{path}.ids', 'w') as f:
            json.dump(self.ids, f)

    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(path)
        with open(f'{path}.ids', 'r') as f:
            self.ids = json.load(f)
```

#### Task 2.4: Index Builder Script
**File**: `scripts/03_build_index.py`

**Performance Estimates** (RTX 3060):
- **Per-image inference**: ~50ms
- **Total time for 770K images**: ~14 hours (one-time job)
- **Optimal batch size**: 32-64 (maximizes GPU utilization)
- **Memory per batch (64)**: ~2GB VRAM

**Features**:
- Load all preprocessed images
- Generate embeddings (batch processing on GPU)
- **Incremental saving** - save chunks to disk (resumable if crash)
- Fit PCA on all embeddings
- Build ANN index
- Save everything to disk

**Implementation**:
```python
import torch
from tqdm import tqdm
import json

def build_index_incremental(
    data_dir,
    output_dir,
    batch_size=64,
    chunk_size=10000
):
    """Build index incrementally with checkpoint support"""

    # Load metadata
    metadata = load_all_metadata(data_dir)

    # Check for existing checkpoints
    checkpoint_dir = Path(output_dir) / "chunks"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Find where to resume
    processed_files = set()
    for chunk_file in checkpoint_dir.glob("chunk_*.npz"):
        # Track which files already processed
        chunk_data = np.load(chunk_file)
        processed_files.update(chunk_data['ids'])

    print(f"Resuming from {len(processed_files)} processed files")

    # Process in chunks
    all_embeddings = []
    all_ids = []

    for i in tqdm(range(0, len(metadata), chunk_size)):
        chunk_metadata = metadata[i:i+chunk_size]

        # Skip already processed
        chunk_metadata = [m for m in chunk_metadata if m['id'] not in processed_files]
        if not chunk_metadata:
            continue

        # Load and batch process
        chunk_embeddings = process_chunk_batched(
            chunk_metadata,
            data_dir,
            batch_size=batch_size
        )

        # Save chunk immediately (resumable!)
        chunk_path = checkpoint_dir / f"chunk_{i//chunk_size}.npz"
        np.savez_compressed(
            chunk_path,
            embeddings=chunk_embeddings,
            ids=[m['id'] for m in chunk_metadata]
        )

        all_embeddings.append(chunk_embeddings)

    # Concatenate all chunks
    all_embeddings = np.vstack(all_embeddings)

    # Fit PCA
    pca = PCA(n_components=512)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # Save PCA
    joblib.dump(pca, f"{output_dir}/pca.pkl")

    # Build FAISS index
    index = faiss.IndexHNSWFlat(512, 32)
    index.add(reduced_embeddings.astype('float32'))

    # Save index
    faiss.write_index(index, f"{output_dir}/faiss.index")

    return index, pca
```

**Command**:
```bash
# Build index with incremental saving (resumable)
python scripts/03_build_index.py \
    --data-dir data/preprocessed \
    --output-dir indexes/embeddings \
    --batch-size 64 \
    --chunk-size 10000 \
    --num-workers 8

# If it crashes at 500K, just re-run - it will resume!
```

#### Task 2.5: Query Interface
**File**: `logo_similarity/retrieval/searcher.py`

**Implementation**:
```python
class LogoSearcher:
    def __init__(self, config):
        self.embedder = EfficientNetEmbedder()
        self.pca = PCAReducer.load('models/pca.pkl')
        self.vector_store = VectorStore.load('indexes/embeddings/faiss.index')
        self.preprocessor = PreprocessingPipeline(config)

    def search(self, image_path: str, top_k: int = 100) -> List[SearchResult]:
        """Search for similar logos"""
        # Preprocess
        processed = self.preprocessor.process(image_path)
        # Embed
        embedding = self.embedder.get_embedding(processed.color)
        # Reduce
        reduced = self.pca.transform(embedding)
        # Search
        results = self.vector_store.search(reduced, k=top_k)
        return results
```

#### Task 2.6: ONNX Export with FP16 Optimization
**File**: `logo_similarity/embeddings/onnx_exporter.py`

**Goal**: Export model to ONNX for production (no Python dependency) with FP16 optimization

**Why FP16**:
- RTX 3060 supports native FP16 (Tensor Cores)
- **Memory**: Halves VRAM usage
- **Speed**: 1.5-2× faster inference
- **Accuracy**: No loss for inference (gradients only matter for training)

**Implementation**:
```python
import torch.onnx
import onnx
from onnxconverter_common import float16

def export_to_onnx(model, output_path, input_size=(1, 3, 224, 224), fp16=True):
    dummy_input = torch.randn(input_size)

    # Export to ONNX (FP32 first)
    torch.onnx.export(
        model,
        dummy_input,
        output_path.replace('.onnx', '_fp32.onnx'),
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=14
    )

    if fp16:
        # Convert to FP16
        model_fp32 = onnx.load(output_path.replace('.onnx', '_fp32.onnx'))
        model_fp16 = float16.convert_float_to_float16(
            model_fp32,
            min_positive_val=1e-7,
            max_finite_val=1e4,
            keep_io_types=False  # Allow FP16 I/O
        )
        onnx.save(model_fp16, output_path)
        print(f"Exported FP16 model to {output_path}")

    return output_path
```

**Command**:
```bash
# Export with FP16 optimization (recommended for RTX 3060)
python scripts/05_export_onnx.py \
    --checkpoint models/checkpoints/efficientnet_b0_logo.pth \
    --output models/onnx/efficientnet_b0_fp16.onnx \
    --fp16

# Verify ONNX output matches PyTorch
python scripts/verify_onnx.py \
    --pytorch-model models/checkpoints/best.pth \
    --onnx-model models/onnx/efficientnet_b0_fp16.onnx
```

**FP16 Inference**:
```python
import onnxruntime as ort

# ONNX Runtime automatically uses FP16 if model is FP16
session = ort.InferenceSession(
    'models/onnx/efficientnet_b0_fp16.onnx',
    providers=['CUDAExecutionProvider'],  # Uses Tensor Cores
    provider_options=[{'device_id': 0}]
)
# Inference will be ~2× faster with half the memory!
```

---

## Phase 3: Stage 2 - Re-ranking with Local Features

**Duration**: 3-4 days
**Goal**: Re-rank top-1000 candidates using spatial/structural verification

### Architecture
```
Query Image        Candidate Image
     │                   │
     ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│  EfficientNet Feature Map Extraction                         │
│  - Output: 7x7x1280 spatial feature map                      │
│  - 49 patches, each 1280-dim                                │
└─────────────────────────────────────────────────────────────┘
     │                   │
     └────────┬──────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Pairwise Cosine Similarity Matrix                          │
│  - Compute 49x49 similarity matrix                          │
│  - Each cell: cosine(query_patch, candidate_patch)          │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Max Pooling per Query Patch                                │
│  - For each query patch, take MAX across all candidate      │
│    patches (translation invariance)                         │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  Alignment Score                                            │
│  - Average of max-scores = spatial alignment score          │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
        Re-ranked Results
```

### Tasks

#### Task 3.1: Spatial Feature Extraction
**File**: `logo_similarity/reranking/spatial_matcher.py`

**Implementation**:
```python
import torch
import torch.nn.functional as F

class SpatialMatcher:
    def __init__(self, embedder):
        self.embedder = embedder

    def extract_spatial_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract 7x7x1280 spatial feature map"""
        with torch.no_grad():
            input_tensor = self._preprocess(image)
            features = self.embedder.model.extract_features(input_tensor)
        return features  # [1, 1280, 7, 7]

    def compute_spatial_similarity(
        self,
        query_features: torch.Tensor,  # [1, 1280, 7, 7]
        candidate_features: torch.Tensor  # [1, 1280, 7, 7]
    ) -> float:
        """Compute max-pooled pairwise similarity score"""
        # Flatten spatial dimensions
        q = query_features.squeeze(0).permute(1, 2, 0)  # [7, 7, 1280]
        c = candidate_features.squeeze(0).permute(1, 2, 0)  # [7, 7, 1280]

        # Reshape for matrix multiplication
        q_flat = q.reshape(49, 1280)  # [49, 1280]
        c_flat = c.reshape(49, 1280)  # [49, 1280]

        # Normalize
        q_norm = F.normalize(q_flat, dim=1)
        c_norm = F.normalize(c_flat, dim=1)

        # Compute pairwise similarity matrix
        similarity = torch.mm(q_norm, c_norm.T)  # [49, 49]

        # Max pooling per query patch (translation invariant)
        max_per_query = similarity.max(dim=1).values  # [49]

        # Average = alignment score
        alignment_score = max_per_query.mean().item()

        return alignment_score
```

#### Task 3.2: Re-ranking Pipeline
**File**: `logo_similarity/reranking/reranker.py`

**Implementation**:
```python
class ReRanker:
    def __init__(self, spatial_matcher, top_k=1000):
        self.spatial_matcher = spatial_matcher
        self.top_k = top_k

    def rerank(
        self,
        query_image: np.ndarray,
        candidates: List[CandidateResult]
    ) -> List[CandidateResult]:
        """Re-rank candidates using spatial matching"""
        query_features = self.spatial_matcher.extract_spatial_features(query_image)

        # Extract features for all candidates (batched)
        candidate_features = self._extract_batch_features(candidates)

        # Compute spatial scores
        for i, candidate in enumerate(candidates):
            candidate.spatial_score = self.spatial_matcher.compute_spatial_similarity(
                query_features,
                candidate_features[i]
            )

        # Combine scores and re-rank
        for candidate in candidates:
            candidate.combined_score = (
                0.50 * candidate.global_score +
                0.50 * candidate.spatial_score
            )

        # Sort by combined score
        return sorted(candidates, key=lambda x: x.combined_score, reverse=True)
```

#### Task 3.3: Composite Scoring
**File**: `logo_similarity/reranking/scoring.py`

**Implementation**:
```python
class CompositeScorer:
    def __init__(self, weights=None):
        self.weights = weights or {
            'global': 0.50,
            'spatial': 0.35,
            'text': 0.10,
            'color': 0.05
        }

    def compute_color_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute color histogram similarity"""
        # Extract dominant colors and compare
        hist1 = self._color_histogram(img1)
        hist2 = self._color_histogram(img2)
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def compute_final_score(
        self,
        global_sim: float,
        spatial_sim: float,
        text_sim: float = 0.0,
        color_sim: float = 0.0
    ) -> float:
        """Compute final composite score"""
        return (
            self.weights['global'] * global_sim +
            self.weights['spatial'] * spatial_sim +
            self.weights['text'] * text_sim +
            self.weights['color'] * color_sim
        )
```

---

## Phase 4: Fine-tuning with Contrastive Learning

**Duration**: 2-3 weeks
**Goal**: Fine-tune EfficientNet on trademark data using contrastive learning

**Critical Hardware Constraint**: RTX 3060 has 12GB VRAM. InfoNCE with batch_size=512 needs ~8-10GB VRAM (tight fit).

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                Contrastive Learning Framework                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Batch of Images (N pairs)                                   │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Data Augmentation                                       │ │
│  │  - Rotation, scale, color jitter                        │ │
│  │  - Random crops                                         │ │
│  │  - Partial occlusion                                    │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  EfficientNet Encoder                                   │ │
│  │  - Output: N embeddings (1280-dim)                      │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Projection Head                                        │ │
│  │  - MLP: 1280 -> 512 -> 256                              │ │
│  └─────────────────────────────────────────────────────────┘ │
│       │                                                       │
│       ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  InfoNCE Loss                                           │ │
│  │  - Each embedding as query                              │ │
│  │  - All other embeddings as negative samples             │ │
│  │  - Vienna codes as positive pairs (weak supervision)     │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### Task 4.1: Training Dataset
**File**: `logo_similarity/training/dataset.py`

**Implementation**:
```python
import torch
from torch.utils.data import Dataset

class LogoContrastiveDataset(Dataset):
    def __init__(self, data_path, split='train', transform=None):
        self.data = self._load_data(data_path, split)
        self.transform = transform
        # Group by Vienna codes for positive pair sampling
        self.vienna_groups = self._build_vienna_groups()

    def _build_vienna_groups(self):
        """Group indices by Vienna codes"""
        groups = defaultdict(list)
        for idx, item in enumerate(self.data):
            for code in item['vienna_codes']:
                groups[code].append(idx)
        return groups

    def __getitem__(self, idx):
        item = self.data[idx]

        # Load image
        image = self._load_image(item['file'])

        # Get positive pair (same Vienna code)
        positive_idx = self._get_positive(idx)
        positive_image = self._load_image(self.data[positive_idx]['file'])

        if self.transform:
            image = self.transform(image)
            positive_image = self.transform(positive_image)

        return {
            'anchor': image,
            'positive': positive_image,
            'vienna_codes': item['vienna_codes']
        }

    def _get_positive(self, idx):
        """Sample a positive pair based on Vienna codes"""
        codes = self.data[idx]['vienna_codes']
        code = random.choice(codes)
        candidates = [i for i in self.vienna_groups[code] if i != idx]
        return random.choice(candidates) if candidates else idx
```

#### Task 4.2: InfoNCE Loss Implementation
**File**: `logo_similarity/training/contrastive.py`

**Implementation**:
```python
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [B*N, D] where B is batch size, N is views (2 for pairs)
        labels: [B*N] - group IDs (Vienna codes)
        """
        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(features, features.T) / self.temperature

        # Create positive mask (same Vienna code, different augmentations)
        batch_size = labels.size(0) // 2
        positive_mask = torch.zeros_like(similarity)

        for i in range(batch_size):
            # Anchor-positive pair
            positive_mask[i, i + batch_size] = 1
            positive_mask[i + batch_size, i] = 1

        # Compute loss
        exp_sim = torch.exp(similarity)
        positive_exp = (exp_sim * positive_mask).sum(dim=1)
        all_exp = exp_sim.sum(dim=1)

        loss = -torch.log(positive_exp / all_exp)
        return loss.mean()
```

#### Task 4.3: Training Script with Gradient Accumulation
**File**: `scripts/04_train_model.py`

**Strategy**: Use gradient accumulation to achieve effective batch size of 512 with actual batch size of 64.

**Implementation**:
```python
def train(config):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = EfficientNetEmbedder(pretrained=True).to(device)

    # Projection head
    projection = ProjectionHead(input_dim=1280, hidden_dim=512, output_dim=256).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Data
    train_dataset = LogoContrastiveDataset(
        config.data_path, split='train',
        transform=get_augmentation_transform()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.actual_batch_size,  # 64 for RTX 3060
        shuffle=True, num_workers=8, pin_memory=True
    )

    # Loss
    criterion = InfoNCELoss(temperature=0.07)

    # Gradient accumulation settings
    effective_batch_size = 512  # Target for good InfoNCE performance
    actual_batch_size = config.actual_batch_size  # 64
    accumulation_steps = effective_batch_size // actual_batch_size  # 8

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        projection.train()

        for i, batch in enumerate(train_loader):
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)

            # Forward
            anchor_feat = model(anchor)
            positive_feat = model(positive)

            # Concatenate for contrastive loss
            features = torch.cat([anchor_feat, positive_feat], dim=0)
            features = projection(features)

            # Create pseudo-labels from Vienna codes
            labels = create_vienna_labels(batch['vienna_codes'])

            # Loss (scaled by accumulation steps)
            loss = criterion(features, labels) / accumulation_steps

            # Backward
            loss.backward()

            # Only step optimizer after accumulating gradients
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # Validation
        if epoch % 5 == 0:
            val_metrics = validate(model, val_loader)
            print(f"Epoch {epoch}: {val_metrics}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'projection_state_dict': projection.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/checkpoints/epoch_{epoch}.pth')
```

**Command**:
```bash
# For RTX 3060 (12GB VRAM)
python scripts/04_train_model.py \
    --data-dir data/preprocessed \
    --actual-batch-size 64 \
    --effective-batch-size 512 \
    --epochs 100 \
    --lr 1e-4 \
    --output-dir models/checkpoints
```

#### Task 4.3b: Alternative - MoCo v3 Training (If InfoNCE problematic)
**File**: `scripts/04_train_model_moco.py`

**When to use**: If gradient accumulation is unstable or too slow, use MoCo v3.

**Advantages of MoCo v3**:
- Small batch size (64) works well
- Queue of 65536 negatives provides massive contrastive signal
- More stable training than InfoNCE with small batches
- Specifically designed for limited VRAM scenarios

**Implementation**:
```python
import queue
from collections import deque

class MoCoV3Trainer:
    def __init__(self, model_q, model_k, queue_size=65536, momentum=0.999):
        self.model_q = model_q  # Query encoder
        self.model_k = model_k  # Key encoder (momentum updated)
        self.queue = deque(maxlen=queue_size)
        self.momentum = momentum

        # Initialize queue with random features
        for _ in range(queue_size):
            self.queue.append(torch.randn(256))  # Projection dim

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Update key encoder with momentum"""
        for param_q, param_k in zip(
            self.model_q.parameters(),
            self.model_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    def train_step(self, batch):
        # Query embeddings
        q = self.projection(self.model_q(batch['query']))

        # Key embeddings (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.projection(self.model_k(batch['key']))

        # Positive pairs
        pos_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative pairs (queue)
        neg_logits = torch.einsum('nc,ck->nk', [q, self.queue.clone()])

        # Concatenate and compute loss
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)

        loss = F.cross_entropy(logits, labels)

        # Update queue
        self.queue.extend(k.detach())

        return loss
```

**Command**:
```bash
# MoCo v3 - smaller batch size works
python scripts/04_train_model_moco.py \
    --data-dir data/preprocessed \
    --batch-size 64 \
    --queue-size 65536 \
    --momentum 0.999 \
    --epochs 100 \
    --lr 1e-4 \
    --output-dir models/checkpoints
```

#### Task 4.4: Training Script (Original - with accumulation)
**File**: `scripts/04_train_model.py`

**Implementation**:
```python
def train(config):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = EfficientNetEmbedder(pretrained=True).to(device)

    # Projection head
    projection = ProjectionHead(input_dim=1280, hidden_dim=512, output_dim=256).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=config.lr,
        weight_decay=config.weight_decay
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Data
    train_dataset = LogoContrastiveDataset(
        config.data_path, split='train',
        transform=get_augmentation_transform()
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=8, pin_memory=True
    )

    # Loss
    criterion = InfoNCELoss(temperature=0.07)

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        projection.train()

        for batch in train_loader:
            anchor = batch['anchor'].to(device)
            positive = batch['positive'].to(device)

            # Forward
            anchor_feat = model(anchor)
            positive_feat = model(positive)

            # Concatenate for contrastive loss
            features = torch.cat([anchor_feat, positive_feat], dim=0)
            features = projection(features)

            # Create pseudo-labels from Vienna codes
            labels = create_vienna_labels(batch['vienna_codes'])

            # Loss
            loss = criterion(features, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation
        if epoch % 5 == 0:
            val_metrics = validate(model, val_loader)
            print(f"Epoch {epoch}: {val_metrics}")

        # Save checkpoint
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'projection_state_dict': projection.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'models/checkpoints/epoch_{epoch}.pth')
```

**Command**:
```bash
python scripts/04_train_model.py \
    --data-dir data/preprocessed \
    --batch-size 512 \
    --epochs 100 \
    --lr 1e-4 \
    --output-dir models/checkpoints
```

#### Task 4.4: Evaluation Metrics
**File**: `logo_similarity/utils/metrics.py`

**Metrics to Track**:
- **Recall@K**: Percentage of true similar pairs in top-K results
- **Precision@K**: Percentage of top-K results that are actually similar
- **Mean Reciprocal Rank (MRR)**: Average of 1/rank_of_first_correct
- **Normalized Discounted Cumulative Gain (NDCG)**: Ranking quality

**Implementation**:
```python
def recall_at_k(true_pairs, predictions, k=100):
    """Compute Recall@K"""
    correct = 0
    for query_id, true_matches in true_pairs.items():
        pred_ids = [p.id for p in predictions[query_id][:k]]
        if any(tid in true_matches for tid in pred_ids):
            correct += 1
    return correct / len(true_pairs)

def precision_at_k(true_pairs, predictions, k=10):
    """Compute Precision@K"""
    precisions = []
    for query_id, true_matches in true_pairs.items():
        pred_ids = [p.id for p in predictions[query_id][:k]]
        correct = sum(1 for pid in pred_ids if pid in true_matches)
        precisions.append(correct / k)
    return np.mean(precisions)
```

---

## Phase 5: Composite Mark Integration

**Duration**: 1 week
**Goal**: Integrate visual similarity with phonetic text similarity

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Composite Mark Scoring                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Mark (Logo + Text)                                    │
│       │                                                       │
│       ├─────────────────────────────────────────────────┐    │
│       │                                                 │    │
│       ▼                                                 ▼    │
│  ┌─────────────────┐                          ┌─────────────┤
│  │  Visual Search  │                          │ Text Search │
│  │  (Stages 1-2)   │                          │  (ALINE)    │
│  └────────┬────────┘                          └──────┬──────┤
│           │                                          │       │
│           │ visual_score (0-1)                       │       │
│           │                                          │       │
│           └────────────────┬─────────────────────────┘       │
│                            │                                  │
│                            ▼                                  │
│                   ┌─────────────────┐                         │
│                   │  Final Score    │                         │
│                   │  = w_v * visual │                         │
│                   │  + w_t * text   │                         │
│                   └─────────────────┘                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### Task 5.1: Text Extraction
**File**: `logo_similarity/text/extractor.py`

**Implementation**:
```python
class TextExtractor:
    def __init__(self):
        self.ocr = pytesseract

    def extract_text(self, image_path: str) -> str:
        """Extract text from logo image"""
        image = cv2.imread(image_path)
        text = self.ocr.image_to_string(image)
        return self._clean_text(text)

    def extract_from_metadata(self, metadata: dict) -> str:
        """Get text from metadata if available"""
        return metadata.get('text', None)
```

#### Task 5.2: Phonetic Similarity (ALINE)
**File**: `logo_similarity/text/phonetic.py`

**Note**: The implementation plan mentions an "existing ALINE module" - we need to integrate it.

**Implementation**:
```python
class ALINEPhoneticSimilarity:
    def __init__(self):
        # Load ALINE algorithm implementation
        self.aline = self._load_aline()

    def similarity(self, text1: str, text2: str) -> float:
        """Compute phonetic similarity using ALINE algorithm"""
        if not text1 or not text2:
            return 0.0

        # Convert to phonemes
        phonemes1 = self.aline.text_to_phonemes(text1)
        phonemes2 = self.aline.text_to_phonemes(text2)

        # Compute ALINE similarity
        return self.aline.phonetic_similarity(phonemes1, phonemes2)
```

#### Task 5.3: Composite Scoring Integration
**File**: `logo_similarity/reranking/composite_scorer.py`

**Implementation**:
```python
class CompositeMarkScorer:
    def __init__(self, visual_scorer, text_scorer, weights=None):
        self.visual_scorer = visual_scorer
        self.text_scorer = text_scorer
        self.weights = weights or {
            'visual': 0.85,  # 0.50 + 0.35 from stages 1-2
            'text': 0.15     # 0.10 + 0.05 from text + color
        }

    def score(
        self,
        query_mark: TrademarkMark,
        candidate_mark: TrademarkMark,
        visual_score: float
    ) -> float:
        """Compute composite similarity score"""
        # Text similarity (if both have text)
        text_score = 0.0
        if query_mark.text and candidate_mark.text:
            text_score = self.text_scorer.similarity(
                query_mark.text,
                candidate_mark.text
            )

        # Composite score
        final_score = (
            self.weights['visual'] * visual_score +
            self.weights['text'] * text_score
        )

        return final_score
```

---

## Phase 6: Production Deployment

**Duration**: 1 week
**Goal**: Deploy system as production API

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                        Production API                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  REST API (FastAPI)                                      │ │
│  │  - POST /search - Search by image                       │ │
│  │  - GET /health - Health check                           │ │
│  │  - POST /batch - Batch search                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ONNX Runtime                                           │ │
│  │  - EfficientNet inference                               │ │
│  │  - Spatial feature extraction                           │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Vector Index (FAISS or pgvector)                        │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  PostgreSQL + pgvector (optional, for persistence)      │ │
│  └─────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Tasks

#### Task 6.1: ONNX Inference Module
**File**: `logo_similarity/api/onnx_inference.py`

**Implementation**:
```python
import onnxruntime as ort

class ONNXEmbedder:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, image: np.ndarray) -> np.ndarray:
        """Generate embedding using ONNX runtime"""
        # Preprocess
        input_tensor = self._preprocess(image)
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )
        return outputs[0]
```

#### Task 6.2: FastAPI Application
**File**: `logo_similarity/api/app.py`

**Implementation**:
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Logo Similarity API")

# Global instances
embedder = ONNXEmbedder('models/onnx/efficientnet_b0.onnx')
searcher = LogoSearcher(config=load_config())

@app.post("/search")
async def search(file: UploadFile = File(...)):
    """Search for similar logos"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Search
        results = searcher.search(tmp_path, top_k=100)

        # Format response
        return {
            "query": file.filename,
            "results": [
                {
                    "id": r.id,
                    "score": float(r.score),
                    "metadata": r.metadata
                }
                for r in results
            ]
        }
    finally:
        os.unlink(tmp_path)

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "model": "efficientnet-b0"}
```

**Deployment**:
```bash
# Install dependencies
pip install fastapi uvicorn python-multipart

# Run server
uvicorn logo_similarity.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Task 6.3: Docker Container
**File**: `Dockerfile`

**Implementation**:
```dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    tesseract-ocr \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY logo_similarity/ logo_similarity/
COPY models/ models/
COPY indexes/ indexes/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "logo_similarity.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Testing & Validation

### Unit Tests
**File**: `tests/test_*.py`

**Coverage**:
- `test_preprocessing.py`: Text detection, masking, normalization
- `test_embeddings.py`: EfficientNet forward pass, PCA reduction
- `test_retrieval.py`: Vector store operations, search accuracy
- `test_reranking.py`: Spatial matching, composite scoring

### Integration Tests
**File**: `tests/integration/test_full_pipeline.py`

**Test Scenarios**:
1. End-to-end search: upload image → get results
2. Batch processing: process 100 images → measure latency
3. Accuracy: check known similar pairs rank high

### Validation Dataset
**File**: `data/validation/benchmark.json`

**Strategy**:
- Collect 100 known similar logo pairs (manual labeling or opposition decisions)
- Measure Recall@K for K = 1, 10, 100
- Target: Recall@100 > 85% after fine-tuning

---

## Hardware Optimization Summary

### RTX 3060 (12GB VRAM) - Specific Optimizations

| Phase | Challenge | Solution | Result |
|-------|-----------|----------|--------|
| **Index Building** | 14 hours processing | Batch size 64, incremental checkpoints | Resumable, ~2GB VRAM |
| **Training (InfoNCE)** | Batch 512 needs 10GB VRAM | Gradient accumulation (8×64) | Effective 512, actual 64 |
| **Training (Alt)** | InfoNCE unstable | MoCo v3 with queue | Batch 64, 65536 negatives |
| **Inference** | Memory bottleneck | FP16 ONNX export | 2× faster, half memory |
| **Vector Index** | 32GB system RAM limit | HNSW for <2M vectors | Fits, IVFFlat fallback |

### Command Reference (Hardware-Aware)

```bash
# 1. Build index (14 hours, resumable)
python scripts/03_build_index.py --batch-size 64 --chunk-size 10000

# 2. Train with gradient accumulation
python scripts/04_train_model.py --actual-batch-size 64 --effective-batch-size 512

# 3. OR train with MoCo v3
python scripts/04_train_model_moco.py --batch-size 64 --queue-size 65536

# 4. Export to FP16 ONNX for production
python scripts/05_export_onnx.py --fp16
```

---

## Risk Mitigation

### Risk 1: Text Detection Accuracy
**Risk**: OCR may fail on stylized text, leaving text in images
**Mitigation**:
- Start with Tesseract, evaluate accuracy
- If needed, upgrade to CRAFT or custom text detector
- Consider manual labeling for validation set

### Risk 2: Computational Cost
**Risk**: Full dataset processing may take weeks
**Mitigation**:
- Use aggressive multiprocessing
- Process on GPU where possible
- Consider cloud computing for training phase

### Risk 3: Memory Constraints
**Risk**: Vector index may not fit in RAM
**Mitigation**:
- Use IVF index (partitions data)
- Consider pgvector with disk-based storage
- Reduce embedding dimension (256 instead of 512)

### Risk 4: Model Overfitting
**Risk**: Fine-tuned model may overfit to Vienna code patterns
**Mitigation**:
- Use strong data augmentation
- Validate on opposition decisions (real similarity labels)
- Keep pre-trained baseline for comparison

### Risk 5: Production Deployment
**Risk**: ONNX export may fail or produce different results
**Mitigation**:
- Validate ONNX output matches PyTorch output
- Use same preprocessing in both
- Test extensively before deployment

---

## Success Metrics

### Week 2 (Phase 1 Complete)
| Metric | Target | How to Measure |
|--------|--------|----------------|
| System functional | Yes | Can search and get results |
| Query latency | <200ms | Time measurement |
| Recall@100 | >60% | Validation dataset |

### Week 3.5 (Phase 2 Complete)
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Precision@10 | >30% | Validation dataset |
| Re-rank latency | <2s | Time measurement |

### Week 6.5 (Phase 4 Complete)
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Recall@100 | >85% | Validation dataset |
| Precision@10 | >55% | Validation dataset |
| Model size | <50MB | File size |

### Week 8.5 (Phase 6 Complete)
| Metric | Target | How to Measure |
|--------|--------|----------------|
| API uptime | >99% | Monitoring |
| Query latency (p99) | <500ms | Monitoring |
| End-to-end accuracy | >80% | Human evaluation |

---

## Next Steps

### Immediate (This Week)
1. Set up environment and directory structure
2. Run dataset analysis script
3. Implement and test text detection
4. Implement preprocessing pipeline

### Week 2
1. Implement EfficientNet embedder
2. Build vector index for full dataset
3. Implement basic search interface
4. Benchmark baseline performance

### Week 3-4
1. Implement spatial re-ranking
2. Build composite scoring
3. Create validation dataset
4. Evaluate and document results

---

## References

1. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", 2019
2. **InfoNCE Loss**: van den Oord et al., "Representation Learning with Contrastive Predictive Coding", 2018
3. **Spatial Feature Matching**: Tolias et al., "Delta encoding: A technique for compact image retrieval", 2015
4. **ALINE Algorithm**: Kondrak, "A New Algorithm for the Alignment of Phonetic Sequences", 2000
5. **pgvector**: https://github.com/pgvector/pgvector
6. **FAISS**: Johnson et al., " Billion-scale similarity search with GPUs", 2019
