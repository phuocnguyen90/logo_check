# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The L3D (Large Labelled Logo Dataset) project is a trademark logo dataset with ~770K images from EUIPO TMView (1996-2020). The codebase serves two purposes:

1. **Dataset Building Pipeline** - Scripts to download, process, and clean trademark data
2. **Logo Similarity Detection** - A new 2-stage pipeline for visual trademark similarity search (in planning phase)

## Environment

- **Python Environment**: Use `torch12` conda/mamba environment (has CUDA 12 preinstalled)
- **GPU**: RTX 3060 (12GB VRAM) - critical constraint for training
- **Dataset Location**: `~/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/`

## Dataset Building Pipeline

The dataset building scripts (prefixes a-g) must be run in order:

```bash
# 1. Download from EUIPO FTP (1996-2020 data, ~53GB)
python a_download_data.py

# 2. Build dataset with multiprocessing (edit line 92 for process count)
python b_build_dataset_multiproc.py

# 3. (Optional) Dataset statistics
python c_dataset_size_stats.py

# 4. Filter out small images (<20px)
python d_filter_dataset.py

# 5. Fix JSON metadata and verify images
python f_fix_json.py

# 6. Remove orphaned image files
python g_clean_images.py
```

**Output Structure**: Each year produces `output_<YEAR>.json` with format:
```json
{"file": "uuid.JPG", "text": "BRAND_NAME" | null, "vienna_codes": ["27.05.01"], "year": 2016}
```

## ImageMagick Processing

Before running `f_fix_json.py`, images must be normalized:

```bash
cd output/images
find . -name \*.TIF -exec mogrify -format jpg '{}' \;
find . -name \*.JPG -exec mogrify -resize 256x256 -background white -gravity center -extent 256x256 '{}' \;
```

## Baselines

Three baseline approaches exist in `baselines/`:

- **tm_basic_generation**: DCGAN for logo image generation
- **tm_generation**: GPT-2 based text/logo generation
- **tm_multi_classification**: NASNet-based Vienna code classification

All baselines use TensorFlow and expect data in `output/` directory.

## Logo Similarity Detection (Planned)

A detailed implementation plan exists in `detailed_implementation_plan.md` for building a 2-stage similarity search system:

1. **Stage 0**: Preprocessing (OCR text detection + masking)
2. **Stage 1**: Global embedding retrieval with EfficientNet-B0 + ANN index (FAISS/pgvector)
3. **Stage 2**: Re-ranking with spatial feature matching
4. **Stage 3**: Fine-tuning with contrastive learning (InfoNCE or MoCo v3)

**Hardware constraints** for training (RTX 3060 12GB):
- InfoNCE batch size 512 needs ~8-10GB VRAM (tight fit)
- Use gradient accumulation (8×64) or MoCo v3 as alternative
- Index building: ~14 hours for 770K images at ~50ms/image
- FP16 ONNX export recommended for production inference

## Vienna Codes

Vienna codes are figurative element classifications (e.g., "27.05.01" = quadrilaterals). Used for:
- Weak supervision signal for contrastive learning
- Multi-label classification in baselines
- Dataset grouping for validation pairs

Level 2 codes (first two components) are commonly used for classification to reduce label space.
