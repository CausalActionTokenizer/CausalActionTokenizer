#!/bin/bash
cd .
source miniconda3/etc/profile.d/conda.sh
conda activate catok

EXPS=(
    "rvq_adj_w10"
    "adj_w10_k32"
    "adj_combined_w10"
)

echo "[$(date)] Auto-eval loop started"

while true; do
    for exp in "${EXPS[@]}"; do
        for ckpt in ckpt/${exp}/*.pth; do
            [ -f "$ckpt" ] || continue
            step=$(basename $ckpt .pth)
            result_file="$(dirname $ckpt)/eval_overlap_rate_${step}.json"
            if [ ! -f "$result_file" ] || [ ! -s "$result_file" ]; then
                echo "[$(date)] Evaluating $ckpt ..."
                python3 utils/multidatasets/evaluation_vanilla_diffusion.py \
                    -m "$ckpt" 2>&1 | tail -3
                default_out="$(dirname $ckpt)/eval_overlap_rate.json"
                [ -f "$default_out" ] && [ -s "$default_out" ] && cp "$default_out" "$result_file"
                echo "[$(date)] Done: $ckpt -> $result_file"
            fi
        done
    done
    sleep 120
done
