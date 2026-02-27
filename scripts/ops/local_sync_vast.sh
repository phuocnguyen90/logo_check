#!/bin/bash
# local_sync_vast.sh - Run this on your LOCAL PC
# Periodically downloads checkpoints from Vast.ai to your local machine.

# --- CONFIGURATION ---
VAST_IP="212.85.84.41"
VAST_PORT="31860"
SSH_KEY="$HOME/.ssh/vast_server"
REMOTE_PATH="/workspace/logo_check/models/checkpoints/"
LOCAL_PATH="./models/checkpoints/"
INTERVAL=600 # Sync every 10 minutes (600 seconds)
# ---------------------

echo "=================================================="
echo "   Vast.ai Checkpoint Synchronizer"
echo "   Syncing: root@$VAST_IP:$VAST_PORT"
echo "   To:      $LOCAL_PATH"
echo "=================================================="

mkdir -p "$LOCAL_PATH"

while true; do
    echo "[$(date)] Syncing latest checkpoints..."
    
    # -a: archive mode, -v: verbose, -z: compress, -e: specify ssh command
    rsync -avz -e "ssh -i $SSH_KEY -p $VAST_PORT -o StrictHostKeyChecking=no" \
          root@$VAST_IP:$REMOTE_PATH $LOCAL_PATH
          
    if [ $? -eq 0 ]; then
        echo "[$(date)] Sync SUCCESS."
    else
        echo "[$(date)] Sync FAILED. Will retry in $INTERVAL seconds."
    fi
    
    echo "Next sync in $((INTERVAL/60)) minutes. Press Ctrl+C to stop."
    sleep $INTERVAL
done
