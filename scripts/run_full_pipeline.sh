#!/bin/bash
# run_full_pipeline.sh - End-to-end production training and indexing

set -e
export PYTHONPATH=.

# 1. Run Setup (Downloads data if missing)
./scripts/setup_remote.sh

# 2. Analyze Dataset & Create Splits
# This is Phase 1 (EDA)
echo "Step 01: Creating Stratified Splits..."
python scripts/01_run_eda.py

# 3. Train MoCo Model
# This is Phase 4
echo "Step 04: Training MoCo Model (Full Dataset)..."
python scripts/04_train_model_moco.py

# 4. Build FAISS Index
# This is Phase 2 & 3
echo "Step 03: Building Search Index..."
python scripts/03_build_index.py --checkpoint models/checkpoints/latest.pth

# 5. Export to ONNX
# This is Phase 6 preparation
echo "Step 05: Exporting to ONNX FP16..."
python scripts/05_export_onnx.py

echo "Full Production Pipeline Complete!"
