#!/usr/bin/env bash
set -euo pipefail
set -x

# Usage:
#   sh scripts/pretrain/train_vanilla_diti.sh single 0
#   sh scripts/pretrain/train_vanilla_diti.sh multi 0,1,2,3
# Optional env:
#   CONFIG_NAME=vanilla_diffusion_multidataset_zcy
#   CONFIG_PATH=conf/pretrain

MODE="${1:-single}"                  # single | multi
GPU_IDS="${2:-0}"                    # e.g. 0 or 0,1,2,3
CONFIG_NAME="${CONFIG_NAME:-vanilla_diffusion_multidataset_zcy}"
CONFIG_PATH="${CONFIG_PATH:-../conf/pretrain}"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=SYS

MASTER_ADDR="127.0.0.1"
MASTER_PORT="$(shuf -i 20000-65535 -n 1)"
MODEL_SCRIPT="scripts/pretrain_multidataset.py"

NUM_GPUS=1
USE_DDP=False
if [ "${MODE}" = "multi" ]; then
  NUM_GPUS="$(echo "${GPU_IDS}" | tr ',' '\n' | wc -l)"
  USE_DDP=True
fi

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${MODEL_SCRIPT}" \
  --config-path="${CONFIG_PATH}" \
  --config-name="${CONFIG_NAME}" \
  +use_ddp="${USE_DDP}" \
  ++tokenizer.basic.flow_type=vanilla \
  ++tokenizer.decoder.vanilla_use_diti=true \
  ++tokenizer.decoder.vanilla_diti_type=cont \
  "++tokenizer.decoder.vanilla_diti_stages='100,600,1000'" \
  "++tokenizer.decoder.vanilla_diti_k_per_stage='2,10,4'" \
  ++tokenizer.decoder.vanilla_t2k=1.0 \
  ++tokenizer.decoder.vanilla_diti_input_mode=auto
