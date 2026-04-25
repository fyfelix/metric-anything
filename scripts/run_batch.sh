#!/bin/bash
set -e

if [[ "$#" -lt 1 ]]; then
    echo "Usage: bash scripts/run_batch.sh <ckpt_directory> [model_type] [dataset_jsonl] [camera_type]"
    exit 1
fi

ROOT_DIR="$1"
shift

MODEL_TYPE="${METRIC_ANYTHING_MODEL_TYPE:-}"
if [[ "$#" -gt 0 && "$1" =~ ^(student_pointmap|student_depthmap)$ ]]; then
    MODEL_TYPE="$1"
    shift
fi

DATASET_PATH="${1:-${METRIC_ANYTHING_DATASET_PATH:-data/HAMMER/test.jsonl}}"
CAMERA_TYPE="${2:-${METRIC_ANYTHING_CAMERA_TYPE:-d435}}"

echo "Processing checkpoints in ${ROOT_DIR}"
if [[ -n "${MODEL_TYPE}" ]]; then
    echo "Model Type: ${MODEL_TYPE}"
fi
echo "Dataset: ${DATASET_PATH}"
echo "Camera Type: ${CAMERA_TYPE}"

shopt -s nullglob
for ckpt in "${ROOT_DIR}"/*.pt "${ROOT_DIR}"/*.pth "${ROOT_DIR}"/*.ckpt; do
    if [[ -f "${ckpt}" ]]; then
        echo "=========================================================="
        echo "Processing checkpoint: ${ckpt}"
        if [[ -n "${MODEL_TYPE}" ]]; then
            bash scripts/run_bs.sh "${MODEL_TYPE}" "${ckpt}" "${DATASET_PATH}" "${CAMERA_TYPE}"
        else
            bash scripts/run_bs.sh "${ckpt}" "${DATASET_PATH}" "${CAMERA_TYPE}"
        fi
        echo "=========================================================="
    fi
done

echo "All batch jobs completed."
