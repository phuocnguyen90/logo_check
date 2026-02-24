---
description: Steps for deploying and running the training pipeline on Vast.ai
---

# Vast.ai Deployment Workflow

This workflow documents the commands needed to connect to a remote Vast.ai instance and run the full logo similarity training pipeline.

## 1. Local Machine: Prepare SSH Key
Ensure you have the `vast_server` key generated:
```bash
# Generated earlier in the conversation
# Key resides at ~/.ssh/vast_server
```
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAMvBOZuH4KgvSfZADWLCMubUpntWsFvdAvqM0g5cpvu phuoc@Phuoc-PC
```

## 2. Connect to the VPS
Replace `PORT` and `IP_ADDRESS` with the details from your Vast.ai instance:
```bash
ssh -i ~/.ssh/vast_server -p [PORT] root@[IP_ADDRESS] -L 8080:localhost:8080
```

## 3. Remote VPS: Initial Setup
Once logged in, setup your Kaggle credentials and the codebase:
```bash
# 1. Set Kaggle Credentials (Required for data download)
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"

# 2. Clone the repository
git clone https://github.com/phuocnguyen90/logo_check.git
cd logo_check/tm-dataset

# 3. Setup Python Path
export PYTHONPATH=.
```

## 4. Run the Pipeline
Use `nohup` to ensure the process continues even if you disconnect:

### Option A: Full End-to-End Run
Includes data download, splitting, training, indexing, and ONNX export.
```bash
nohup bash scripts/run_full_pipeline.sh > pipeline.log 2>&1 &
```

### Option B: Resume Training Only
If data is already downloaded and splits are created:
```bash
nohup python scripts/04_train_model_moco.py > train.log 2>&1 &
```

## 5. Monitoring
Check the logs in real-time:
```bash
# To watch progress
tail -f train.log

# To check GPU utilization
nvidia-smi -l 5
```

## 6. Troubleshooting
If you need to stop all training processes:
```bash
pkill -f python
pkill -f bash
```
