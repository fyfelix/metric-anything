#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_bs.sh
  bash scripts/run_bs.sh student_pointmap
  bash scripts/run_bs.sh student_depthmap --align
  bash scripts/run_bs.sh <model_path> [dataset_jsonl] [camera_type] [--align]

Common flags:
  --model-type TYPE        student_pointmap or student_depthmap.
  --model-path PATH        Local checkpoint path or Hugging Face repo id.
  --dataset PATH           HAMMER/ClearPose JSONL path.
  --camera-type TYPE       Raw camera type: d435, l515, or tof.
  --output PATH            Prediction/evaluation output directory.
  --intrinsics-path PATH   3x3 intrinsics matrix for student_depthmap.
  --f-px VALUE             Focal length in pixels for student_depthmap.
  --align                  Enable per-image scale/shift alignment in eval.

Environment overrides:
  PYTHON_BIN, METRIC_ANYTHING_MODEL_TYPE, METRIC_ANYTHING_MODEL_PATH,
  METRIC_ANYTHING_DATASET_PATH, METRIC_ANYTHING_OUTPUT_DIR,
  METRIC_ANYTHING_INTRINSICS_PATH, METRIC_ANYTHING_F_PX,
  METRIC_ANYTHING_BS, METRIC_ANYTHING_NUM_WORKERS,
  METRIC_ANYTHING_DEPTH_SCALE, METRIC_ANYTHING_MIN_DEPTH,
  METRIC_ANYTHING_MAX_DEPTH, METRIC_ANYTHING_PREDICTION_RESIZE_MODE
EOF
}

default_model_path() {
    case "$1" in
        student_depthmap)
            printf "%s\n" "${METRIC_ANYTHING_DEPTHMAP_MODEL_PATH:-ckpts/student_depthmap.pt}"
            ;;
        *)
            printf "%s\n" "${METRIC_ANYTHING_POINTMAP_MODEL_PATH:-ckpts/student_pointmap.pt}"
            ;;
    esac
}

MODEL_TYPE="${METRIC_ANYTHING_MODEL_TYPE:-student_pointmap}"
MODEL_TYPE_EXPLICIT=false
if [[ -n "${METRIC_ANYTHING_MODEL_TYPE:-}" ]]; then
    MODEL_TYPE_EXPLICIT=true
fi

MODEL_PATH="${METRIC_ANYTHING_MODEL_PATH:-}"
DATASET_PATH="${METRIC_ANYTHING_DATASET_PATH:-data/HAMMER/test.jsonl}"
CAMERA_TYPE="${METRIC_ANYTHING_CAMERA_TYPE:-d435}"
OUTPUT_DIR="${METRIC_ANYTHING_OUTPUT_DIR:-}"
INTRINSICS_PATH="${METRIC_ANYTHING_INTRINSICS_PATH:-data/HAMMER/intrinsics.txt}"
F_PX="${METRIC_ANYTHING_F_PX:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BS="${METRIC_ANYTHING_BS:-16}"
NUM_WORKERS="${METRIC_ANYTHING_NUM_WORKERS:-8}"
DEPTH_SCALE="${METRIC_ANYTHING_DEPTH_SCALE:-1000.0}"
MIN_DEPTH="${METRIC_ANYTHING_MIN_DEPTH:-0.1}"
MAX_DEPTH="${METRIC_ANYTHING_MAX_DEPTH:-6.0}"
PREDICTION_RESIZE_MODE="${METRIC_ANYTHING_PREDICTION_RESIZE_MODE:-bilinear}"

DATASET_FROM_FLAG=false
CAMERA_FROM_FLAG=false
POSITIONAL=()
EVAL_EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        student_pointmap|student_depthmap)
            MODEL_TYPE="$1"
            MODEL_TYPE_EXPLICIT=true
            shift
            ;;
        --model-type)
            MODEL_TYPE="$2"
            MODEL_TYPE_EXPLICIT=true
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_PATH="$2"
            DATASET_FROM_FLAG=true
            shift 2
            ;;
        --camera-type|--raw-type)
            CAMERA_TYPE="$2"
            CAMERA_FROM_FLAG=true
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --intrinsics-path)
            INTRINSICS_PATH="$2"
            shift 2
            ;;
        --f-px)
            F_PX="$2"
            shift 2
            ;;
        --depth-scale)
            DEPTH_SCALE="$2"
            shift 2
            ;;
        --min-depth)
            MIN_DEPTH="$2"
            shift 2
            ;;
        --max-depth)
            MAX_DEPTH="$2"
            shift 2
            ;;
        --batch-size|-b)
            BS="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --prediction-resize-mode)
            PREDICTION_RESIZE_MODE="$2"
            shift 2
            ;;
        --align)
            EVAL_EXTRA_ARGS+=("--align")
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EVAL_EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${MODEL_PATH}" && -n "${POSITIONAL[0]:-}" ]]; then
    MODEL_PATH="${POSITIONAL[0]}"
fi
if [[ "${DATASET_FROM_FLAG}" == "false" && -n "${POSITIONAL[1]:-}" ]]; then
    DATASET_PATH="${POSITIONAL[1]}"
fi
if [[ "${CAMERA_FROM_FLAG}" == "false" && -n "${POSITIONAL[2]:-}" ]]; then
    CAMERA_TYPE="${POSITIONAL[2]}"
fi
if [[ "${#POSITIONAL[@]}" -gt 3 ]]; then
    EVAL_EXTRA_ARGS+=("${POSITIONAL[@]:3}")
fi

if [[ -z "${MODEL_PATH}" ]]; then
    MODEL_PATH="$(default_model_path "${MODEL_TYPE}")"
fi

if [[ "${MODEL_TYPE_EXPLICIT}" == "false" ]]; then
    case "${MODEL_PATH}" in
        *depthmap*|*DepthMap*|*student_depth*)
            MODEL_TYPE="student_depthmap"
            ;;
        *pointmap*|*PointMap*|*student_point*)
            MODEL_TYPE="student_pointmap"
            ;;
    esac
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}"
    echo "Refusing to continue so from_pretrained() cannot download weights from the internet."
    exit 1
fi

MODEL_NAME="$(basename "${MODEL_PATH}")"
MODEL_STUB="${MODEL_NAME%%.*}"
DATASET_NAME="$(basename "$(dirname "${DATASET_PATH}")")"
if [[ -z "${DATASET_NAME}" || "${DATASET_NAME}" == "." ]]; then
    DATASET_NAME="dataset"
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
    if [[ -e "${MODEL_PATH}" || -d "$(dirname "${MODEL_PATH}")" ]]; then
        OUTPUT_PARENT="$(dirname "${MODEL_PATH}")"
    else
        OUTPUT_PARENT="${METRIC_ANYTHING_EVAL_ROOT:-eval}"
    fi
    OUTPUT_DIR="${OUTPUT_PARENT}/eval_${DATASET_NAME}_${MODEL_STUB}_${CAMERA_TYPE}"
fi

echo "Using Model: ${MODEL_PATH}"
echo "Model Type: ${MODEL_TYPE}"
echo "Dataset: ${DATASET_PATH}"
echo "Camera Type: ${CAMERA_TYPE}"
echo "Python: ${PYTHON_BIN}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Batch Size: ${BS}"
echo "Workers: ${NUM_WORKERS}"
echo "Depth Scale: ${DEPTH_SCALE}"
echo "Eval Depth Range Defaults: [${MIN_DEPTH}, ${MAX_DEPTH}]"
echo "Prediction Resize Mode: ${PREDICTION_RESIZE_MODE}"
if [[ -n "${INTRINSICS_PATH}" ]]; then
    echo "Intrinsics Path: ${INTRINSICS_PATH}"
fi
if [[ -n "${F_PX}" ]]; then
    echo "DepthMap f_px: ${F_PX}"
fi

INFER_ARGS=(
    scripts/infer_dataset_bs.py
    --model-path "${MODEL_PATH}"
    --model-type "${MODEL_TYPE}"
    --dataset "${DATASET_PATH}"
    --raw-type "${CAMERA_TYPE}"
    --output "${OUTPUT_DIR}"
    --depth-scale "${DEPTH_SCALE}"
    --min-depth "${MIN_DEPTH}"
    --max-depth "${MAX_DEPTH}"
    --prediction-resize-mode "${PREDICTION_RESIZE_MODE}"
    --batch-size "${BS}"
    --num-workers "${NUM_WORKERS}"
)

if [[ -n "${INTRINSICS_PATH}" ]]; then
    INFER_ARGS+=(--intrinsics-path "${INTRINSICS_PATH}")
fi
if [[ -n "${F_PX}" ]]; then
    INFER_ARGS+=(--f-px "${F_PX}")
fi

echo "[1/2] Running MetricAnything batch inference..."
time "${PYTHON_BIN}" "${INFER_ARGS[@]}"

echo "[2/2] Evaluating benchmark metrics..."
time "${PYTHON_BIN}" scripts/eval_mp.py \
    --dataset "${DATASET_PATH}" \
    --output "${OUTPUT_DIR}" \
    --raw-type "${CAMERA_TYPE}" \
    --depth-scale "${DEPTH_SCALE}" \
    --min-depth "${MIN_DEPTH}" \
    --max-depth "${MAX_DEPTH}" \
    --batch-size "${BS}" \
    --num-workers "${NUM_WORKERS}" \
    "${EVAL_EXTRA_ARGS[@]}"

echo "Evaluation completed. Results saved in ${OUTPUT_DIR}"
