#!/bin/bash
set -e
export PYTHONPATH=.

# Scaffolding
# 00. Create Toy Dataset (Small 10 samples/class)
TOY_METADATA="data/toy_results.json"
echo "Step 00: Creating Toy Dataset (10 samples/class)..."
# Force recreate
rm -f "$TOY_METADATA"
rm -rf data/toy_splits/*
/home/phuoc/miniforge3/envs/torch12/bin/python scripts/00_create_toy_dataset.py --samples 10

# 01. Create Splits
echo "Step 01: Creating Stratified Splits for Toy Dataset..."
/home/phuoc/miniforge3/envs/torch12/bin/python scripts/01_run_eda.py --toy

# 04. Train MoCo Model (Short run for pilot)
echo "Step 04: Training MoCo Model on Toy Dataset..."
# Resetting checkpoints for clean pilot run
rm -rf models/checkpoints/*
/home/phuoc/miniforge3/envs/torch12/bin/python scripts/04_train_model_moco.py --toy

# 03. Build Index with Trained Model
echo "Step 03: Building Index with Trained Model..."
rm -rf indexes/embeddings/chunks/*
/home/phuoc/miniforge3/envs/torch12/bin/python scripts/03_build_index.py --toy --checkpoint models/checkpoints/latest.pth --workers 4

echo "End-to-End Toy Pipeline Complete!"
