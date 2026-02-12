#!/bin/bash
set -e

# Configuration
VERSION="v1_semantic"
EPOCHS=20
LOG_FILE="training_semantic_${VERSION}.log"

echo "=================================================="
echo "Starting Semantic Alignment Training (Phase 4b)"
echo "Version: ${VERSION}"
echo "Epochs: ${EPOCHS}"
echo "Log File: ${LOG_FILE}"
echo "=================================================="

# Ensure data is ready (not needed if just resuming training, but good practice)
# echo "Checking dataset..."

# Run Training
# buffer output to log file
nohup python -u scripts/06_train_semantic_alignment.py \
    --epochs ${EPOCHS} \
    --name "${VERSION}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "To monitor progress, run: tail -f ${LOG_FILE}"
