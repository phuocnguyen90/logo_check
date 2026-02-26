# Logo Similarity Detection System

A production-ready logo similarity search system for trademark images. Built with deep embeddings, structural re-ranking, and high-performance inference.

## Features

- **2-Stage Retrieval Pipeline**:
  - **Stage 1 (Global)**: EfficientNet-B0 embeddings + PCA + FAISS (HNSW) for fast candidate retrieval.
  - **Stage 2 (Local)**: 49x49 max-pooled patch similarity for structural/spatial verification.
- **MoCo v3 Training**: Contrastive learning framework optimized for RTX 3060 (12GB VRAM) using a 65K negative queue.
- **On-the-Fly Preprocessing**: Integrated text detection (Tesseract) and inpainting (OpenCV) with LRU caching.
- **Multi-Model Serving**: Serves both Visual Similarity and Figurative/Semantic models from a single API with automatic S3 synchronization.
- **Rich Metadata Search**: Returns trademark names, Vienna classifications, and filing years alongside visual matches.
- **Production-Ready**: FastAPI service with ONNX Runtime (FP16/CPU) and FAISS memory-mapping for low RAM footprint.

## Quick Start

### 1. Prerequisites
- CUDA 12 support
- Python >= 3.9
- Tesseract OCR engine

### 2. Installation
```bash
pip install -e .
```

### 3. Usage

#### Initialize Index
```bash
python scripts/03_build_index.py
```

#### Fine-tune Model
```bash
python scripts/04_train_model_moco.py
```

#### Deploy API
```bash
python scripts/05_export_onnx.py
python logo_similarity/api/app.py
```

## Project Structure

- `logo_similarity/`: Main package containing preprocessing, embeddings, retrieval, and API logic.
- `scripts/`: Operational scripts for indexing, training, and deployment.
- `legacy/`: Original repository code and baseline implementations.
- `models/`: Checkpoints and exported ONNX models.
- `indexes/`: FAISS indexes and mapping files.

## Tech Stack

- **Model**: EfficientNet-B0 (PyTorch)
- **Contrastive Learning**: MoCo v3
- **Vector Search**: FAISS
- **Deployment**: FastAPI + ONNX Runtime (FP16)
- **Logging**: Loguru
- **OCR**: Tesseract
