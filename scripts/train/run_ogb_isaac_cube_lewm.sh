#!/bin/bash
# Train LeWM on the Isaac Lab OGBench cube policy corpus (40k trajectories).
#
# Source: /root/ogbench_isaaclab/data/trajectory-generations/train-episodes-40000/
# Uses the streaming ogbench_policy HDF5 reader (raw uint8 CHW pixels;
# ImageNet normalization applied in scripts/train/lewm.py).
#
# Usage:
#   ./scripts/train/run_ogb_isaac_cube_lewm.sh
#   MAX_EPOCHS=1 BATCH_SIZE=32 LIMIT_TRAIN_BATCHES=20 ./scripts/train/run_ogb_isaac_cube_lewm.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/stable-worldmodel}"
DATA_PATH="${DATA_PATH:-/root/ogbench_isaaclab/data/trajectory-generations/train-episodes-40000/train.hdf5}"
RUN_NAME="${RUN_NAME:-lewm_ogb_isaac_cube_40k}"
# Match scripts/train/config/lewm.yaml defaults unless overridden via env.
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-6}"
DEVICES="${DEVICES:-auto}"
LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-}"
STABLEWM_HOME="${STABLEWM_HOME:-${REPO_DIR}/.stablewm}"

cd "${REPO_DIR}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: policy HDF5 not found at ${DATA_PATH}" >&2
  exit 1
fi

export STABLEWM_HOME
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME}}"

EXTRA_ARGS=()
if [[ -n "${LIMIT_TRAIN_BATCHES}" ]]; then
  # Hydra struct: must use + to add keys absent from lewm.yaml trainer block
  EXTRA_ARGS+=("+trainer.limit_train_batches=${LIMIT_TRAIN_BATCHES}")
fi
if [[ -n "${LIMIT_VAL_BATCHES}" ]]; then
  EXTRA_ARGS+=("+trainer.limit_val_batches=${LIMIT_VAL_BATCHES}")
fi

echo "Run: ${RUN_NAME} (epochs=${MAX_EPOCHS}, batch=${BATCH_SIZE}, workers=${NUM_WORKERS}, devices=${DEVICES}, data=${DATA_PATH})"
# lewm.yaml defaults + Isaac Lab policy corpus (ogb_isaac_cube: frameskip=1, pixels+action).
# action_encoder.input_dim is auto-set to frameskip * action_dim (= 25).
uv run --extra train --extra format python scripts/train/lewm.py \
  data=ogb_isaac_cube \
  "data.dataset.name=${DATA_PATH}" \
  "output_model_name=${RUN_NAME}" \
  "subdir=${RUN_NAME}" \
  "trainer.max_epochs=${MAX_EPOCHS}" \
  "trainer.devices=${DEVICES}" \
  "trainer.accelerator=gpu" \
  "trainer.precision=bf16" \
  "trainer.gradient_clip_val=1.0" \
  "loader.batch_size=${BATCH_SIZE}" \
  "num_workers=${NUM_WORKERS}" \
  "loader.persistent_workers=true" \
  "loader.prefetch_factor=3" \
  "loader.pin_memory=true" \
  "loader.shuffle=true" \
  "loader.drop_last=true" \
  "train_split=0.9" \
  "seed=3072" \
  "img_size=224" \
  "patch_size=14" \
  "encoder_scale=tiny" \
  "embed_dim=192" \
  wandb.enabled=false \
  "${EXTRA_ARGS[@]}"
