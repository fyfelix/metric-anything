#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="$PWD:$PYTHONPATH"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_bs.sh
  bash scripts/run_bs.sh student_pointmap
  bash scripts/run_bs.sh student_depthmap --align

Optional flags:
  --align                 Enable alignment during evaluation.
  --model-path PATH       Override checkpoint path.
  --dataset PATH          Override dataset jsonl path.
  --camera-type TYPE      Override camera type (d435/l515/tof).

Defaults:
  model type:   student_pointmap
  model path:   ckpts/<model_type>.pt
  dataset:      data/HAMMER/test.jsonl
  camera type:  d435

Legacy positional form is still supported:
  bash scripts/run_bs.sh <model_path> [dataset_jsonl] [camera_type] [--align]
EOF
}

MODEL_TYPE="student_pointmap"
MODEL_PATH=""
DATASET_PATH="data/HAMMER/test.jsonl"
CAMERA_TYPE="d435"
EXTRA_ARGS=()
POSITIONAL_INDEX=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        student_pointmap|student_depthmap)
            MODEL_TYPE="$1"
            shift
            ;;
        --align)
            EXTRA_ARGS+=("--align")
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            shift 2
            ;;
        --camera-type)
            CAMERA_TYPE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ ${POSITIONAL_INDEX} -eq 0 ]]; then
                MODEL_PATH="$1"
            elif [[ ${POSITIONAL_INDEX} -eq 1 ]]; then
                DATASET_PATH="$1"
            elif [[ ${POSITIONAL_INDEX} -eq 2 ]]; then
                CAMERA_TYPE="$1"
            else
                EXTRA_ARGS+=("$1")
            fi
            POSITIONAL_INDEX=$((POSITIONAL_INDEX + 1))
            shift
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    MODEL_PATH="ckpts/${MODEL_TYPE}.pt"
fi

echo "Using Model: ${MODEL_PATH}"
echo "Model Type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET_PATH}"
echo "Camera Type: ${CAMERA_TYPE}"

MODEL_NAME=$(basename "${MODEL_PATH}")
MODEL_STUB="${MODEL_NAME%%.*}"
DIRNAME=$(dirname "${MODEL_PATH}")

OUTPUT_DIR="${DIRNAME}/eval_hammer_${MODEL_STUB}_${CAMERA_TYPE}"
echo "Output Directory: ${OUTPUT_DIR}"

BS=${BS:-16}
NUM_WORKERS=${NUM_WORKERS:-8}

echo "[1/2] 开始在数据集上推理模型..."
time python scripts/infer_dataset_bs.py \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --raw-type "${CAMERA_TYPE}" \
    --output "${OUTPUT_DIR}" \
    --batch-size ${BS} \
    --num-workers ${NUM_WORKERS}

echo "[2/2] 开始在数据集上评估指标..."
time python scripts/eval_mp.py \
    --dataset "${DATASET_PATH}" \
    --output "${OUTPUT_DIR}" \
    --raw-type "${CAMERA_TYPE}" \
    --batch-size ${BS} \
    --num-workers ${NUM_WORKERS} \
    "${EXTRA_ARGS[@]}"

echo "全部评估完成！结果保存在 ${OUTPUT_DIR}"
