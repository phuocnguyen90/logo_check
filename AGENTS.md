# L3D (Large Labelled Logo Dataset) — Agent Guide

## Project Overview

The L3D project is a trademark logo dataset containing ~770,000 images from EUIPO TMView (1996-2020). The codebase serves two purposes:

1. **Dataset Building Pipeline** — Scripts to download, process, and clean trademark data from EUIPO's FTP server
2. **Logo Similarity Detection Research** — Baseline implementations and implementation plans for visual trademark similarity search

**Website:** https://lhf-labs.github.io/tm-dataset  
**Dataset:** https://doi.org/10.5281/zenodo.5771006  
**Paper:** https://arxiv.org/abs/2112.05404

---

## Technology Stack

### Core Technologies
- **Python**: 3.8 or greater (required per instructions.md)
- **Primary Framework**: TensorFlow (for existing baselines)
- **Planned Framework**: PyTorch (for logo similarity detection — implementation phase)
- **Image Processing**: PIL/Pillow, ImageMagick (external tool)
- **XML Parsing**: xmltodict
- **Progress Bars**: tqdm

### External Dependencies
- **ImageMagick**: Required for image normalization (format conversion, resizing to 256x256)
- **Tesseract OCR**: Planned for text detection in the logo similarity pipeline
- **Dataset Source**: KaggleHub (`konradb/ziilogos`) or EUIPO FTP server

### Environment
- **Recommended Environment**: `torch12` conda/mamba environment with CUDA 12 preinstalled
- **Hardware**: Developed with RTX 3060 (12GB VRAM) — critical constraint for training
- **Dataset Location**: `~/.cache/kagglehub/datasets/konradb/ziilogos/versions/1/L3D dataset/`

---

## Project Structure

```
.
├── a_download_data.py           # Download dataset from EUIPO FTP
├── b_build_dataset_multiproc.py # Extract and process ZIP files (multiprocessing)
├── c_dataset_size_stats.py      # Compute image size statistics
├── d_filter_dataset.py          # Remove small/outlier images (<20px)
├── e_fix_images.sh              # ImageMagick batch processing script
├── f_fix_json.py                # Fix JSON metadata, verify images
├── g_clean_images.py            # Remove orphaned image files
├── h_statistics.py              # Generate analysis plots (by year, vienna codes)
├── i_statistics.py              # Generate sunburst visualization data
├── z_vienna_codes.py            # Normalize Vienna code categories
├── kaggle_download.py           # Alternative: download from Kaggle
├── baselines/                   # Three baseline implementations
│   ├── tm_basic_generation/     # DCGAN for logo generation
│   ├── tm_generation/             # GPT-2 based text/logo generation (fairseq)
│   └── tm_multi_classification/ # NASNet-based Vienna code classification
├── analysis/                    # Output directory for statistics plots
├── *.md                         # Documentation and implementation plans
└── .claude/                     # Claude-specific files
```

---

## Build/Run Instructions

### Dataset Building Pipeline (Run in Order)

The dataset building scripts (prefixes a-g) must be run sequentially:

```bash
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
python kaggle_download.py

# Or manually from Zenodo: https://doi.org/10.5281/zenodo.5771006
```

### Running Baselines

Each baseline is self-contained in its subdirectory:

```bash
# DCGAN for logo generation
cd baselines/tm_basic_generation
python main.py

# Vienna code classification
cd baselines/tm_multi_classification
python main.py

# GPT-2 generation (requires fairseq installation)
cd baselines/tm_generation
bash a_install.sh
bash b_train.sh
```

---

## Data Format

### JSON Metadata Structure

After processing, `results.json` contains entries like:

```json
{
  "file": "uuid.JPG",
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
- Weak supervision signal for contrastive learning
- Multi-label classification in baselines
- Dataset grouping for validation pairs

**Level 2 codes** (first two components, e.g., "27.05") are commonly used to reduce label space.

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
pip install tqdm pillow xmltodict matplotlib numpy tensorflow torch transformers
```

For logo similarity work, also install:
```bash
pip install faiss-cpu opencv-python-headless pytesseract loguru
```

---

## Testing Strategy

**No formal test suite exists.** Testing is manual/integration-based:

1. **Pipeline validation**: Run scripts sequentially and verify output files
2. **Image verification**: `f_fix_json.py` includes PIL `img.verify()` for corruption detection
3. **Statistics validation**: `h_statistics.py` and `i_statistics.py` generate plots for visual inspection

For planned logo similarity detection, validation uses:
- Known-similar pairs from Vienna code groupings
- Opposition decisions for ground-truth calibration

---

## Key Implementation Plans

The repository contains detailed implementation plans for a 2-stage logo similarity detection system:

| File | Description |
|------|-------------|
| `implementation_plan.md` | Original 2-stage pipeline overview (EfficientNet + spatial features) |
| `detailed_implementation_plan_v1.5.md` | Phase-by-phase breakdown with hardware constraints |
| `revised_implementation_plan_V2..md` | Updated plan with MoCo v3, checkpointing, mixed precision |

**Architecture Overview:**
```
Input Logo → Preprocess (OCR + mask text) → Stage 1: ANN Retrieval → Stage 2: Re-rank → Results
```

**Hardware Constraints:**
- RTX 3060 12GB VRAM limits batch sizes
- InfoNCE requires large batches (512+); use gradient accumulation or MoCo v3
- Index building: ~14 hours for 770K images

---

## Important Notes

### Path Conventions
- Most scripts expect to be run from the repository root
- Baseline scripts use hardcoded relative paths (`../../../output/`)
- Output directory structure: `output/images/` for images, `output/*.json` for metadata

### Memory Considerations
- Dataset is ~53GB raw, ~200GB+ when processed
- Multiprocessing in `b_build_dataset_multiproc.py` can consume significant RAM
- Consider reducing `processes` in Pool (line 92) if memory constrained

### Checkpointing
The revised implementation plan (V2) emphasizes atomic checkpoints for long-running operations. Current pipeline scripts do not implement this — planned for logo similarity implementation.

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
