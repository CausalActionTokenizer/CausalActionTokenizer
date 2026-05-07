#!/bin/bash
# Watchdog: auto-restart aid decoder training from latest checkpoint on OOM kill.
set -e
cd .
PYTHON=python
SAVE_DIR=ckpt/aid_decoder
LOG_DIR=logs
ZQ=/dev/shm/aid_zq.npy
ACTIONS=/dev/shm/aid_actions.npy

mkdir -p "$LOG_DIR"

run=0
while true; do
    # Find latest checkpoint
    LATEST=$(ls -1 "$SAVE_DIR"/*.pth 2>/dev/null | grep -E '/[0-9]+\.pth$' | sort -V | tail -1)
    if [ -z "$LATEST" ]; then
        echo "[watchdog] No checkpoint found, exiting."
        exit 1
    fi

    # Extract step number from checkpoint name
    STEP=$(basename "$LATEST" .pth | sed 's/^0*//')
    TOTAL_STEPS=188400  # 200 epochs * 942 steps/epoch
    if [ "$STEP" -ge "$TOTAL_STEPS" ]; then
        echo "[watchdog] Training complete at step $STEP."
        exit 0
    fi

    run=$((run + 1))
    LOGFILE="$LOG_DIR/train_aid_run${run}.log"
    echo "[watchdog] Run $run: resuming from $LATEST (step $STEP) → $LOGFILE"

    $PYTHON -u scripts/train_aid_decoder.py \
        --resume_ckpt "$LATEST" \
        --save_dir "$SAVE_DIR" \
        --zq_npy "$ZQ" \
        --actions_npy "$ACTIONS" \
        --batch_size 256 \
        --epochs 200 \
        --save_every 1000 \
        2>&1 | tee -a "$LOGFILE"

    EXIT=$?
    echo "[watchdog] Run $run exited with code $EXIT. Restarting in 5s..."
    sleep 5
done
