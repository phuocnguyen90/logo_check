# Logo Similarity Detection — Revised Implementation Plan v2

## Executive Summary

**Goal**: Build a production-ready logo similarity search system for trademark images with **robust error handling, checkpointing, and hardware optimization**.

**Timeline**: 8-10 weeks

**Key Changes from v1**:
- ✅ **MoCo v3 as primary training** (not InfoNCE with gradient accumulation)
- ✅ **Atomic checkpoints** with corruption recovery across all phases
- ✅ **Mixed precision training** (AMP) for 40% memory reduction
- ✅ **On-the-fly preprocessing** (no 200GB+ disk storage)
- ✅ **Comprehensive error handling** with graceful degradation
- ✅ **Structured logging** and resource monitoring

---

## Architecture: 2-Stage Pipeline + Preprocessing

```
Input Logo → Preprocess (OCR + mask text) → Stage 1: Retrieve candidates (ANN) → Stage 2: Re-rank → Results
```

**Key principle**: Deep embeddings from day one. No hand-crafted features.

---

## Dataset Information

- **Location**: `~/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/`
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

---

## Environment

- **Python Environment**: `torch12` (conda/mamba env)
- **CUDA**: Version 12 preinstalled
- **GPU**: RTX 3060 (12GB VRAM) - **Critical constraint for training**
- **System RAM**: 32GB
- **Working Directory**: `/home/phuoc/git/l3d/tm-dataset/`

---

## Phase 0: Data Analysis & Preparation

**Duration**: 2-3 days  
**Goal**: Understand dataset characteristics and prepare train/val/test splits

### Key Tasks

1. **Dataset Statistics with Incremental Checkpointing**
   - Image size distribution, aspect ratios, color vs grayscale
   - Vienna code distribution, year distribution
   - **NEW**: Save stats every 10K images to `stats_checkpoint.json`
   - **NEW**: Skip already-analyzed images on resume
   - **NEW**: Pre-flight validation: check for corrupted images

2. **Create Train/Val/Test Splits**
   - Train: 70% (538,771 images)
   - Validation: 15% (115,451 images)
   - Test: 15% (115,452 images)
   - Stratified by Vienna codes

3. **Create Known-Similar Pairs for Validation**
   - Use Vienna codes as weak similarity signal
   - Output: `validation/similar_pairs.json`, `validation/dissimilar_pairs.json`

### Robustness Features
- ✅ Incremental stats saving (resume from crash)
- ✅ Corrupted image detection and logging
- ✅ Disk space pre-flight check
- ✅ Structured logging with `loguru`

---

## Phase 1: Preprocessing Pipeline

**Duration**: 4-5 days  
**Goal**: Normalize input images and remove text to isolate figurative elements

### Architecture

```
Input Image → Text Detection (Tesseract) → Text Masking (Inpainting) → Normalization (224×224) → Cache
```

### Key Change: **On-the-Fly Preprocessing**

Instead of storing 770K preprocessed images as `.npz` (~200-464GB), we:
1. **Preprocess on-the-fly** during training/indexing
2. **Cache preprocessed images** in memory (LRU cache) for repeated access
3. **Store only text masks** (sparse, RLE-encoded) for reproducibility

**Why**: Saves ~200GB disk, faster iteration during development, augmentation happens anyway during training.

### Tasks

1. **Text Detection Module** (`preprocessing/text_detector.py`)
   - Tesseract OCR for baseline
   - Return bounding boxes for text regions
   - **NEW**: Try/except around OCR with fallback to "no text detected"

2. **Text Masking Module** (`preprocessing/text_masker.py`)
   - OpenCV inpainting (Telea or NS method)
   - **NEW**: Handle edge cases (empty mask, full-image text)

3. **Image Normalization Module** (`preprocessing/image_normalizer.py`)
   - Resize to 256×256 (maintain aspect ratio, pad)
   - Center crop to 224×224
   - Convert to RGB, white background for transparent images

4. **Preprocessing Pipeline** (`preprocessing/pipeline.py`)
   - Combines all steps
   - **NEW**: LRU cache for repeated access
   - **NEW**: Multiprocessing with worker isolation (one bad image doesn't kill pool)
   - **NEW**: Error logging to `preprocessing_errors.jsonl`

### Robustness Features
- ✅ Worker isolation in multiprocessing (no pool crashes)
- ✅ Per-image try/except with detailed error logging
- ✅ Graceful degradation (skip text masking if OCR fails)
- ✅ Signal handlers for graceful shutdown (SIGTERM/SIGINT)
- ✅ LRU cache for performance

---

## Phase 2: Stage 1 - Global Embedding Retrieval

**Duration**: 5-6 days  
**Goal**: Build efficient ANN search using pre-trained EfficientNet embeddings

### Architecture

```
Preprocessed Image → EfficientNet-B0 (1280-dim) → PCA (512-dim) → HNSW Index → Top-1000 Candidates
```

### Tasks

1. **EfficientNet Wrapper** (`embeddings/efficientnet.py`)
   - Pre-trained ImageNet weights
   - Remove classification head, use global average pooling
   - **NEW**: FP16 inference option for 2× speed

2. **PCA Reducer** (`embeddings/pca_reducer.py`)
   - **NEW**: Use `IncrementalPCA` to fit on chunks (avoid loading 770K embeddings at once)
   - Reduce 1280-dim → 512-dim
   - Save fitted PCA model

3. **Vector Store** (`retrieval/vector_store.py`)
   - FAISS HNSW index for development
   - **NEW**: Use `faiss.IndexIDMap` to store IDs directly in index (no separate JSON)
   - pgvector for production (optional)

4. **Index Builder Script** (`scripts/03_build_index.py`)
   - **NEW**: Atomic chunk writes (write to `.tmp`, then rename)
   - **NEW**: Chunk integrity validation on resume
   - **NEW**: GPU OOM recovery (retry with smaller batch)
   - **NEW**: Per-image error handling (skip corrupted images)
   - **NEW**: Progress bar with ETA and throughput metrics

### Performance Estimates (RTX 3060)
- Per-image inference: ~50ms
- Total time for 770K images: ~14 hours (one-time job)
- Optimal batch size: 32-64
- Memory per batch (64): ~2GB VRAM

### Robustness Features
- ✅ Atomic chunk writes (no corrupt checkpoints)
- ✅ Chunk integrity validation on resume
- ✅ GPU OOM recovery with automatic batch size reduction
- ✅ Per-image error handling with skip + log
- ✅ IncrementalPCA for memory efficiency
- ✅ Resource monitoring (GPU util, VRAM, throughput)

---

## Phase 3: Stage 2 - Re-ranking with Local Features

**Duration**: 3-4 days  
**Goal**: Re-rank top-1000 candidates using spatial/structural verification

### Architecture

```
Query Image → EfficientNet (7×7×1280 feature map) → Pairwise Similarity (49×49 matrix) → Max-pool → Alignment Score
```

### Method: Max-Pooled Pairwise Similarity

For each query patch, find best match across all candidate patches (translation invariant).

**Why not grid 1-to-1**: Breaks with even slight shifts.  
**Why not RANSAC**: Too slow for 1000 candidates.

### Tasks

1. **Spatial Feature Extraction** (`reranking/spatial_matcher.py`)
   - Extract 7×7×1280 feature maps from EfficientNet
   - Compute 49×49 pairwise cosine similarity matrix
   - Max-pool per query patch, average for alignment score

2. **Re-ranking Pipeline** (`reranking/reranker.py`)
   - Batch extract features for all 1000 candidates
   - Compute spatial scores
   - Combine with global scores (0.50 global + 0.50 spatial)

3. **Composite Scoring** (`reranking/scoring.py`)
   - Visual: 0.85 (0.50 global + 0.35 spatial)
   - Text: 0.10 (phonetic similarity via ALINE)
   - Color: 0.05 (histogram similarity)

### Robustness Features
- ✅ Batch processing for efficiency
- ✅ Error handling for feature extraction failures

---

## Phase 4: Fine-tuning with MoCo v3 (Primary Approach)

**Duration**: 2-3 weeks  
**Goal**: Fine-tune EfficientNet on trademark data using MoCo v3 contrastive learning

### Why MoCo v3 Over InfoNCE with Gradient Accumulation?

| Approach | Batch Size | Negatives | VRAM | Quality |
|----------|-----------|-----------|------|---------|
| InfoNCE + grad accum | 64 actual, 512 "effective" | **64 in-batch** | ~2GB | ❌ Poor (only 64 negatives) |
| MoCo v3 | 64 actual | **65,536 queue** | ~2GB | ✅ Excellent (massive negatives) |

**Key insight**: Gradient accumulation does NOT give you large-batch contrastive quality for InfoNCE. Each micro-batch only sees its own 64 negatives. MoCo v3 solves this with a momentum-updated queue of 65K negatives.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MoCo v3 Framework                     │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Batch (64 pairs)                                        │
│       │                                                  │
│       ├──────────────────┬──────────────────┐            │
│       │                  │                  │            │
│       ▼                  ▼                  ▼            │
│  Query Encoder    Key Encoder (momentum)  Queue (65K)   │
│  (trainable)      (no grad)                             │
│       │                  │                               │
│       └──────────┬───────┘                               │
│                  │                                       │
│                  ▼                                       │
│           Contrastive Loss                              │
│           (query vs key + queue)                        │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

### Tasks

1. **Training Dataset** (`training/dataset.py`)
   - Group images by Vienna codes for positive pair sampling
   - Strong augmentation: rotation, scale, color jitter, partial occlusion
   - **NEW**: On-the-fly preprocessing (no disk storage)

2. **MoCo v3 Trainer** (`training/moco_trainer.py`)
   - Query encoder (trainable)
   - Key encoder (momentum-updated, no gradient)
   - Queue of 65,536 negative embeddings
   - Momentum: 0.999
   - **NEW**: Mixed precision training (`torch.cuda.amp`)
   - **NEW**: Save checkpoints every epoch (not every 10)
   - **NEW**: Track best model based on validation Recall@100

3. **Training Script** (`scripts/04_train_model_moco.py`)
   - **NEW**: Comprehensive checkpoint saving:
     ```python
     {
       'epoch': epoch,
       'step': global_step,
       'model_q_state_dict': model_q.state_dict(),
       'model_k_state_dict': model_k.state_dict(),
       'projection_state_dict': projection.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'scheduler_state_dict': scheduler.state_dict(),
       'scaler_state_dict': scaler.state_dict(),  # AMP
       'queue': queue.clone(),
       'best_val_recall': best_val_recall,
     }
     ```
   - **NEW**: Automatic resume from latest checkpoint
   - **NEW**: Validation every epoch with early stopping
   - **NEW**: Resource monitoring (GPU util, VRAM, throughput)

### Training Configuration (RTX 3060 Optimized)

```python
# MoCo v3 settings
batch_size = 64              # Fits in ~2GB VRAM with AMP
queue_size = 65536           # Massive negative pool
momentum = 0.999             # For key encoder updates

# Mixed precision (AMP)
use_amp = True               # 40% memory reduction, 30% speedup

# Optimization
lr = 1e-4
weight_decay = 1e-4
epochs = 100
warmup_epochs = 10

# Checkpointing
save_every_n_epochs = 1      # Save often (not every 10)
validate_every_n_epochs = 1
early_stopping_patience = 10
```

### Expected Performance
- **Baseline (pre-trained)**: Recall@100 > 60%
- **After MoCo v3**: Recall@100 > 85%
- **Training time**: ~3-5 days on RTX 3060

### Robustness Features
- ✅ Mixed precision training (AMP) for memory efficiency
- ✅ Comprehensive checkpoint saving (all state dicts)
- ✅ Automatic resume from latest checkpoint
- ✅ Best model tracking (save best separately)
- ✅ Early stopping to prevent overfitting
- ✅ Validation every epoch
- ✅ Resource monitoring and logging
- ✅ Graceful shutdown on SIGTERM/SIGINT

---

## Phase 5: Composite Mark Integration

**Duration**: 1 week  
**Goal**: Integrate visual similarity with phonetic text similarity

### Architecture

```
Input Mark (Logo + Text)
     │
     ├─────────────────────────┐
     │                         │
     ▼                         ▼
Visual Search            Text Search
(Stages 1-2)             (ALINE phonetic)
     │                         │
     └────────┬────────────────┘
              │
              ▼
       Final Score
       = 0.85 × visual
       + 0.15 × text
```

### Tasks

1. **Text Extraction** (`text/extractor.py`)
   - Extract text from metadata (preferred)
   - Fallback to OCR if metadata missing

2. **Phonetic Similarity** (`text/phonetic.py`)
   - ALINE algorithm for phonetic matching
   - Handle missing text gracefully

3. **Composite Scoring** (`reranking/composite_scorer.py`)
   - Combine visual + text scores
   - Configurable weights

---

## Phase 6: Production Deployment

**Duration**: 1 week  
**Goal**: Deploy system as production API

### Architecture

```
REST API (FastAPI) → ONNX Runtime (FP16) → FAISS Index → PostgreSQL + pgvector (optional)
```

### Tasks

1. **ONNX Export with FP16** (`scripts/05_export_onnx.py`)
   - Export fine-tuned model to ONNX FP16
   - **NEW**: Validation script to verify ONNX vs PyTorch outputs (cosine similarity > 0.999)
   - 2× faster inference, half memory

2. **ONNX Inference Module** (`api/onnx_inference.py`)
   - ONNX Runtime with CUDA provider
   - Batch inference support

3. **FastAPI Application** (`api/app.py`)
   - `POST /search` - Search by image
   - `GET /health` - Health check
   - `POST /batch` - Batch search
   - **NEW**: Error handling and rate limiting

4. **Docker Container** (`Dockerfile`)
   - CUDA 12 runtime
   - All dependencies
   - Health checks

### Robustness Features
- ✅ ONNX validation (verify correctness)
- ✅ FP16 for production (2× speed)
- ✅ Health checks and monitoring
- ✅ Error handling in API

---

## Testing & Validation

### Unit Tests
- `test_preprocessing.py`: Text detection, masking, normalization
- `test_embeddings.py`: EfficientNet forward pass, PCA reduction
- `test_retrieval.py`: Vector store operations, search accuracy
- `test_reranking.py`: Spatial matching, composite scoring
- `test_moco_trainer.py`: MoCo v3 training loop

### Integration Tests
- End-to-end search: upload image → get results
- Batch processing: process 100 images → measure latency
- Accuracy: check known similar pairs rank high

### Validation Dataset
- Collect 100 known similar logo pairs
- Measure Recall@K for K = 1, 10, 100
- Target: Recall@100 > 85% after fine-tuning

---

## Success Metrics

| Phase | Metric | Target | How to Measure |
|-------|--------|--------|----------------|
| **Week 2** (Phase 2) | System functional | Yes | Can search and get results |
| | Query latency | <200ms | Time measurement |
| | Recall@100 | >60% | Validation dataset |
| **Week 3.5** (Phase 3) | Precision@10 | >30% | Validation dataset |
| | Re-rank latency | <2s | Time measurement |
| **Week 6.5** (Phase 4) | Recall@100 | >85% | Validation dataset |
| | Precision@10 | >55% | Validation dataset |
| | Model size | <50MB | File size |
| **Week 8.5** (Phase 6) | API uptime | >99% | Monitoring |
| | Query latency (p99) | <500ms | Monitoring |
| | End-to-end accuracy | >80% | Human evaluation |

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Inference | ONNX Runtime (FP16) | No Python in production |
| Vector index | FAISS (HNSW) → pgvector | FAISS for dev, pgvector for prod |
| Image processing | PIL + OpenCV | On-the-fly preprocessing |
| OCR | Tesseract | Text detection |
| Training (offline) | PyTorch + AMP → ONNX export | One-time, not in production path |
| Logging | Loguru | Structured logging |
| API | FastAPI | Production API |
| Containerization | Docker | CUDA 12 runtime |

---

## Key Improvements from v1

| Issue | v1 Approach | v2 Approach | Impact |
|-------|-------------|-------------|--------|
| **Training** | InfoNCE + grad accum | **MoCo v3** | ✅ True large-batch contrastive |
| **Memory** | No AMP | **Mixed precision (AMP)** | ✅ 40% memory reduction |
| **Checkpoints** | Every 10 epochs, missing scheduler | **Every epoch, all state dicts** | ✅ Resume without issues |
| **Preprocessing** | Store 770K `.npz` (~200GB) | **On-the-fly + LRU cache** | ✅ Save 200GB disk |
| **Error handling** | Minimal | **Comprehensive try/except, worker isolation** | ✅ No pipeline crashes |
| **Chunk saves** | Direct write | **Atomic writes (.tmp → rename)** | ✅ No corrupt checkpoints |
| **Logging** | `print()` | **Structured logging (loguru)** | ✅ Better debugging |
| **Best model** | Not tracked | **Track + save best separately** | ✅ Don't lose good checkpoints |
| **ONNX validation** | Not mentioned | **Verify PyTorch vs ONNX** | ✅ Catch FP16 issues |
| **Resource monitoring** | None | **GPU util, VRAM, throughput** | ✅ Performance insights |

---

## Next Steps

### Immediate (This Week)
1. Set up environment and directory structure
2. Implement structured logging framework
3. Run dataset analysis with incremental checkpointing
4. Implement preprocessing pipeline with error handling

### Week 2
1. Implement EfficientNet embedder with FP16 option
2. Build vector index with atomic checkpoints
3. Implement basic search interface
4. Benchmark baseline performance

### Week 3-4
1. Implement spatial re-ranking
2. Build composite scoring
3. Create validation dataset
4. Evaluate and document results

### Week 5-7
1. Implement MoCo v3 trainer with AMP
2. Train on full dataset
3. Track metrics and validate
4. Export to ONNX FP16

### Week 8-9
1. Build FastAPI application
2. Deploy with Docker
3. End-to-end testing
4. Production monitoring

---

## References

1. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", 2019
2. **MoCo v3**: Chen et al., "An Empirical Study of Training Self-Supervised Vision Transformers", 2021
3. **Spatial Feature Matching**: Tolias et al., "Particular object retrieval with integral max-pooling of CNN activations", 2015
4. **ALINE Algorithm**: Kondrak, "A New Algorithm for the Alignment of Phonetic Sequences", 2000
5. **pgvector**: https://github.com/pgvector/pgvector
6. **FAISS**: Johnson et al., "Billion-scale similarity search with GPUs", 2019
7. **Mixed Precision Training**: Micikevicius et al., "Mixed Precision Training", 2018
