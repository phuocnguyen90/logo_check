#!/bin/bash
# setup_remote.sh - Prepare a remote VPS (e.g., Vast.ai) for training

set -e

echo "Setting up remote environment..."

# 1. Install System Dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y tesseract-ocr libgl1-mesa-glx unzip

# 2. Install Python Dependencies
echo "Installing python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download Dataset from Kaggle
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "ERROR: KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set."
    echo "Get them from your Kaggle account settings (Create New API Token)."
    exit 1
fi

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

if [ ! -f "$RAW_DIR/results.json" ]; then
    echo "Downloading dataset from Kaggle..."
    mkdir -p ~/.kaggle
    echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_KEY\"}" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
    
    # Download the dataset
    kaggle datasets download -d konradb/ziilogos -p "$RAW_DIR"
    
    echo "Extracting dataset..."
    unzip -o "$RAW_DIR/ziilogos.zip" -d "$RAW_DIR"
    
    # The ziilogos dataset usually extracts into a subfolder. 
    # Let's align it with our RAW_DATASET_DIR path.
    # Check if 'L3D dataset' folder exists inside and move its content up
    if [ -d "$RAW_DIR/L3D dataset" ]; then
        mv "$RAW_DIR/L3D dataset/"* "$RAW_DIR/"
        rmdir "$RAW_DIR/L3D dataset"
    fi
    
    rm "$RAW_DIR/ziilogos.zip"
else
    echo "Dataset already found in $RAW_DIR, skipping download."
fi

echo "Setup complete!"
