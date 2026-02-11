# L3D (Large Labelled Logo Dataset) — Agent Guide

## Project Overview

The L3D project is a trademark logo dataset containing ~770,000 images from EUIPO TMView (1996-2020). The codebase serves three purposes:

1. **Dataset Building Pipeline** — Scripts to download, process, and clean trademark data (legacy, moved to `legacy/`)
2. **Logo Similarity Detection** — A 2-stage pipeline for visual trademark similarity search (active development)
3. **Baseline Models** — DCGAN generation, GPT-2 generation, NASNet classification (moved to `legacy/baselines/`)

**Website:** https://lhf-labs.github.io/tm-dataset  
**Dataset:** https://doi.org/10.5281/zenodo.5771006  
**Paper:** https://arxiv.org/abs/2112.05404

---

## Technology Stack

### Core Technologies
- **Python**: 3.8 or greater (required per instructions.md)
- **Primary Framework**: PyTorch (for logo similarity detection — active development)
- **Legacy Framework**: TensorFlow (for baselines in `legacy/`)
- **Image Processing**: PIL/Pillow, OpenCV, ImageMagick (external tool)
- **OCR**: Tesseract (pytesseract)
- **Vector Search**: FAISS
- **Logging**: loguru
- **Progress Bars**: tqdm

### External Dependencies
- **ImageMagick**: Required for image normalization (format conversion, resizing to 256x256)
- **Tesseract OCR**: Used for text detection in the logo similarity pipeline
- **Dataset Source**: KaggleHub (`konradb/ziilogos`) or EUIPO FTP server

### Environment
- **Recommended Environment**: `torch12` conda/mamba environment with CUDA 12 preinstalled
- **Hardware**: Developed with RTX 3060 (12GB VRAM) — **critical constraint for training**
- **System RAM**: 32GB
- **Dataset Location**: `~/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/`

---

## Project Structure

```
.
├── logo_similarity/             # Active development: Logo similarity detection
│   ├── api/                     # FastAPI production endpoints
│   ├── config/                  # Configuration files
│   ├── embeddings/              # EfficientNet embedder, PCA, ONNX export
│   ├── preprocessing/           # Text detection, masking, normalization
│   ├── retrieval/               # Vector store (FAISS), search interface
│   ├── reranking/               # Spatial feature matching, composite scoring
│   ├── text/                    # Text extraction, phonetic similarity (ALINE)
│   ├── training/                # MoCo v3 contrastive learning
│   └── utils/                   # Logging, metrics, data utilities
├── legacy/                      # Legacy dataset building scripts
│   ├── a_download_data.py       # Download dataset from EUIPO FTP
│   ├── b_build_dataset_multiproc.py
│   ├── c_dataset_size_stats.py
│   ├── d_filter_dataset.py
│   ├── e_fix_images.sh
│   ├── f_fix_json.py
│   ├── g_clean_images.py
│   ├── h_statistics.py
│   ├── i_statistics.py
│   ├── z_vienna_codes.py
│   ├── kaggle_download.py
│   └── baselines/               # Three baseline implementations
│       ├── tm_basic_generation/ # DCGAN for logo generation
│       ├── tm_generation/       # GPT-2 based text/logo generation (fairseq)
│       └── tm_multi_classification/ # NASNet-based Vienna code classification
├── analysis/                    # Output directory for statistics plots
├── *.md                         # Documentation and implementation plans
└── .claude/                     # Claude-specific files
```

---

## Build/Run Instructions

### Logo Similarity Detection System (Active Development)

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

### Legacy Dataset Building Pipeline (in `legacy/`)

The dataset building scripts (prefixes a-g) must be run sequentially:

```bash
cd legacy

# 1. Download from EUIPO FTP (~53GB, 1996-2020 data)
python a_download_data.py

# 2. Build dataset with multiprocessing
# NOTE: Edit line 92 in b_build_dataset_multiproc.py to adjust process count
python b_build_dataset_multiproc.py

# 3. (Optional) Dataset statistics
python c_dataset_size_stats.py

# 4. Filter out small images (<20px)
python d_filter_dataset.py

# 5. ImageMagick processing — convert to JPG and resize to 256x256
cd output/images
find . -name \*.TIF -exec mogrify -format jpg '{}' \;
find . -name \*.JPG -exec mogrify -resize 256x256 -background white -gravity center -extent 256x256 '{}' \;
cd ../..

# 6. Fix JSON metadata and verify images
python f_fix_json.py

# 7. Remove orphaned image files
python g_clean_images.py
```

### Alternative: Download Pre-built Dataset

```bash
# From Kaggle (recommended for faster setup)
python legacy/kaggle_download.py

# Or manually from Zenodo: https://doi.org/10.5281/zenodo.5771006
```

### Running Legacy Baselines

Each baseline is self-contained in its subdirectory:

```bash
# DCGAN for logo generation
cd legacy/baselines/tm_basic_generation
python main.py

# Vienna code classification
cd legacy/baselines/tm_multi_classification
python main.py

# GPT-2 generation (requires fairseq installation)
cd legacy/baselines/tm_generation
bash a_install.sh
bash b_train.sh
```

---

## Logo Similarity Detection Architecture (2-Stage Pipeline)

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

### Hardware Constraints (RTX 3060 12GB)

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

---

## Data Format

### JSON Metadata Structure

```json
{
  "file": "uuid.jpg",
  "text": "BRAND_NAME",
  "vienna_codes": ["27.05.01"],
  "year": 2016
}
```

Fields:
- `file`: UUID-based filename (images stored in `output/images/`)
- `text`: Brand name text (null if not present; ~90.1% have text)
- `vienna_codes`: Array of Vienna classification codes (figurative element categories)
- `year`: Filing year from EUIPO data

### Vienna Codes

Vienna codes classify figurative elements (e.g., "27.05.01" = quadrilaterals). Used for:
- Weak supervision signal for contrastive learning (same code ≈ similar)
- Multi-label classification in baselines
- Dataset grouping for validation pairs

**Level 2 codes** (first two components, e.g., "27.05") are commonly used to reduce label space.

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| Week 2 | Recall@100 | >60% |
| Week 3.5 | Precision@10 | >30% |
| Week 6.5 | Recall@100 | >85% |
| Week 8.5 | Query latency (p99) | <500ms |

---

## Code Style Guidelines

### Conventions Observed
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Constants**: UPPER_CASE at module level (e.g., `PATH`, `BATCH_SIZE`)
- **File naming**: Prefix letters (a_, b_, c_) indicate execution order for pipeline scripts
- **Path handling**: Hardcoded relative paths (e.g., `../output/`, `../../../images_test/`)
- **Encoding**: Explicit UTF-8 for file operations (`encoding='utf-8'`)

### Dependencies
No formal dependency management (no requirements.txt, pyproject.toml, etc.). Install manually:

```bash
# Core dependencies
pip install tqdm pillow xmltodict matplotlib numpy torch transformers

# Logo similarity work
pip install faiss-cpu opencv-python-headless pytesseract loguru

# Legacy baselines
pip install tensorflow
```

---

## Testing Strategy

**No formal test suite exists.** Testing is manual/integration-based:

1. **Pipeline validation**: Run scripts sequentially and verify output files
2. **Image verification**: `legacy/f_fix_json.py` includes PIL `img.verify()` for corruption detection
3. **Statistics validation**: `legacy/h_statistics.py` and `legacy/i_statistics.py` generate plots for visual inspection

For logo similarity detection, validation uses:
- Known-similar pairs from Vienna code groupings
- Opposition decisions for ground-truth calibration

---

## Key Implementation Plans

The repository contains detailed implementation plans for the 2-stage logo similarity detection system:

| File | Description |
|------|-------------|
| `implementation_plan.md` | Original 2-stage pipeline overview (EfficientNet + spatial features) |
| `detailed_implementation_plan_v1.5.md` | Phase-by-phase breakdown with hardware constraints |
| `revised_implementation_plan_V2..md` | Updated plan with MoCo v3, checkpointing, mixed precision |

---

## Important Notes

### Path Conventions
- Most scripts expect to be run from the repository root
- Legacy baseline scripts use hardcoded relative paths (`../../../output/`)
- Output directory structure: `output/images/` for images, `output/*.json` for metadata

### Memory Considerations
- Dataset is ~53GB raw, ~200GB+ when processed
- Multiprocessing in `legacy/b_build_dataset_multiproc.py` can consume significant RAM
- Consider reducing `processes` in Pool (line 92) if memory constrained

### Checkpointing
The revised implementation plan (V2) emphasizes atomic checkpoints for long-running operations. Current logo similarity scripts implement this; legacy pipeline scripts do not.

---

## References

1. **MoCo v3**: Chen et al., "An Empirical Study of Training Self-Supervised Vision Transformers", 2021
2. **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", 2019
3. **ALINE Algorithm**: Kondrak, "A New Algorithm for the Alignment of Phonetic Sequences", 2000

---

## Citation

```bibtex
@misc{gutierrezfandino2021L3D,
  title={The Large Labelled Logo Dataset (L3D): A Multipurpose and Hand-Labelled Continuously Growing Dataset}, 
  author={Asier Gutiérrez-Fandiño and David Pérez-Fernández and Jordi Armengol-Estapé},
  year={2021},
  eprint={2112.05404},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Contact

For additional details: Asier Gutiérrez-Fandiño <asier.gutierrez@bsc.es>
