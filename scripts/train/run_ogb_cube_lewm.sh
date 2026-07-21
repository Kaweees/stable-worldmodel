#!/bin/bash

# Train LeWM on the locally collected OGBench cube-single Lance dataset.
# Usage:
#   ./scripts/train/run_ogb_cube_lewm.sh

#
# Private Impl
#

REPO_DIR="${REPO_DIR:-/workspace/stable-worldmodel}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/outputs/ogb_train_1k}"
DATASET_NAME="${DATASET_NAME:-ogbench/cube_single_expert.lance}"
DATASET_PATH="${DATA_ROOT}/datasets/${DATASET_NAME}"
RUN_NAME="${RUN_NAME:-lewm_cube_single_1k}"
MAX_EPOCHS="${MAX_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-6}"
NUM_TRAJ="${NUM_TRAJ:-1000}"

generate() {
  cd "${REPO_DIR}"

  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Generating ${NUM_TRAJ} trajectories at ${DATASET_PATH}"
    MUJOCO_GL=egl uv run --frozen --no-dev --extra train --no-sync \
      python scripts/data/collect_cube.py \
      env_type=single \
      "num_traj=${NUM_TRAJ}" \
      "cache_dir=${DATA_ROOT}"
  fi

  echo "Run: ${RUN_NAME} (${MAX_EPOCHS} epochs, batch size ${BATCH_SIZE}, ${NUM_WORKERS} workers)"
  LOCAL_DATASET_DIR="${DATA_ROOT}" STABLEWM_HOME="${DATA_ROOT}" \
    uv run --frozen --no-dev --extra train python scripts/train/lewm.py \
    data=ogb \
    "data.dataset.name=${DATASET_NAME}" \
    '~data.dataset.keys_to_merge' \
    "output_model_name=${RUN_NAME}" \
    "subdir=${RUN_NAME}" \
    "trainer.max_epochs=${MAX_EPOCHS}" \
    "loader.batch_size=${BATCH_SIZE}" \
    "num_workers=${NUM_WORKERS}"
}

# Main script logic
set -xeuo pipefail
generate
