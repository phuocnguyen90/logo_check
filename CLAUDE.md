# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The L3D (Large Labelled Logo Dataset) project is a trademark logo dataset with ~770K images from EUIPO TMView (1996-2020). The codebase serves three purposes:

1. **Dataset Building Pipeline** - Scripts to download, process, and clean trademark data (legacy, moved to `legacy/`)
2. **Logo Similarity Detection** - A 2-stage pipeline for visual trademark similarity search (active development)
3. **Baseline Models** - DCGAN generation, GPT-2 generation, NASNet classification (moved to `legacy/baselines/`)

## Environment

- **Python Environment**: Use `torch12` conda/mamba environment (has CUDA 12 preinstalled)
- **GPU**: RTX 3060 (12GB VRAM) - **critical constraint for training**
- **System RAM**: 32GB
- **Dataset Location**: `~/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/`

## Project Structure

```
logo_similarity/
├── api/              # FastAPI production endpoints
├── config/           # Configuration files
├── embeddings/       # EfficientNet embedder, PCA, ONNX export
├── preprocessing/    # Text detection, masking, normalization
├── retrieval/        # Vector store (FAISS), search interface
├── reranking/        # Spatial feature matching, composite scoring
├── text/             # Text extraction, phonetic similarity (ALINE)
├── training/         # MoCo v3 contrastive learning
└── utils/            # Logging, metrics, data utilities
```

## Logo Similarity Detection System

### Architecture (2-Stage Pipeline)

```
Input Logo → Preprocess (OCR + mask text) → Stage 1: ANN Retrieve Top-1000 → Stage 2: Re-rank → Results
```

### Key Design Decisions (v2)

| Decision | Rationale |
|----------|-----------|
| **MoCo v3** over InfoNCE | True large-batch contrastive (65K negatives) with small actual batch (64) |
| **On-the-fly preprocessing** | Save ~200GB disk (no .npz storage), LRU cache for repeated access |
| **Mixed precision (AMP)** | 40% VRAM reduction, 30% speedup |
| **Atomic checkpoints** | Write to `.tmp` then rename (no corrupt checkpoints) |
| **Structured logging (loguru)** | Better debugging than `print()` |

### Development Commands

```bash
# Activate environment
mamba activate torch12

# Phase 2: Build embedding index (~14 hours, resumable)
python scripts/03_build_index.py --batch-size 64 --chunk-size 10000

# Phase 4: Train with MoCo v3 (primary approach)
python scripts/04_train_model_moco.py --batch-size 64 --queue-size 65536

# Phase 6: Export to FP16 ONNX
python scripts/05_export_onnx.py --fp16
```

## Hardware Constraints (RTX 3060 12GB)

### Training Configuration (MoCo v3 + AMP)

| Setting | Value | VRAM Usage |
|---------|-------|------------|
| Batch size | 64 | ~2GB (with AMP) |
| Queue size | 65,536 | ~1GB (CPU) |
| Mixed precision | True | -40% vs FP32 |
| Index building batch | 64 | ~2GB |

### Why MoCo v3?

Gradient accumulation does NOT give large-batch contrastive quality for InfoNCE. Each micro-batch only sees its own 64 negatives. MoCo v3 solves this with a momentum-updated queue of 65K negatives.

| Approach | Negatives | Quality |
|----------|-----------|---------|
| InfoNCE + grad accum | 64 in-batch | Poor |
| MoCo v3 | 65,536 queue | Excellent |

## Dataset Format

```json
{
  "file": "uuid.jpg",
  "text": "BRAND_NAME" | null,  // 90.1% have text
  "vienna_codes": ["27.05.01"],
  "year": 2016
}
```

## Vienna Codes

Figurative element classifications (e.g., "27.05.01" = quadrilaterals). Used for:
- Weak supervision signal for contrastive learning (same code ≈ similar)
- Multi-label classification in baselines
- Dataset grouping for validation pairs

Level 2 codes (first two components) are commonly used to reduce label space.

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Week 2 | Recall@100 | >60% |
| Week 3.5 | Precision@10 | >30% |
| Week 6.5 | Recall@100 | >85% |
| Week 8.5 | Query latency (p99) | <500ms |

## Legacy Code

Dataset building scripts (a-g) and baselines have been moved to `legacy/`:
- `legacy/a_download_data.py` - EUIPO FTP download
- `legacy/b_build_dataset_multiproc.py` - Dataset builder
- `legacy/baselines/` - DCGAN, GPT-2, NASNet models

These use TensorFlow and expect data in `output/` directory.

## References

1. **MoCo v3**: Chen et al., "An Empirical Study of Training Self-Supervised Vision Transformers", 2021
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", 2019
3. **ALINE Algorithm**: Kondrak, "A New Algorithm for the Alignment of Phonetic Sequences", 2000
