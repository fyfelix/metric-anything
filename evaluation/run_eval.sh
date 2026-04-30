#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  ./evaluation/run_eval.sh [model_ref=ckpts/student_depthmap.pt] [raw_type=d435] [cleanup_npy=false]

Environment overrides:
  DATASET_PATH          HAMMER JSONL path. Default: data/HAMMER/test.jsonl
  OUTPUT_DIR            Prediction/evaluation output directory. Default: evaluation/output
  BATCH_SIZE            Inference batch size. Default: 1
  NUM_WORKERS           Inference DataLoader workers. Default: 0
  DEVICE                Inference device, e.g. cuda, cuda:0, cpu, or mps.
  F_PX                  Optional fixed focal length in pixels for all images.
  INTRINSICS_PATH       Camera intrinsics txt path. Default: <DATASET_PATH dir>/intrinsics.txt
  REQUIRE_FOCAL         Set true to fail when no explicit focal is available.
  LOCAL_FILES_ONLY      Set true to avoid Hugging Face downloads.
  SAVE_VIS              Set false to skip depth preview PNGs. Default: true
  PYTHON_BIN            Python executable. Default: python3

This wrapper is fixed to the current project model:
  models/student_depthmap/depth_model.py::MetricAnythingDepthMap
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_ref="${1:-${MODEL_PATH:-ckpts/student_depthmap.pt}}"
raw_type="${2:-${RAW_TYPE:-d435}}"
cleanup_npy="${3:-${CLEANUP_NPY:-false}}"

dataset_path="${DATASET_PATH:-data/HAMMER/test.jsonl}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
arch="${ARCH:-vitl}"
intrinsics_path="${INTRINSICS_PATH:-$(dirname "${dataset_path}")/intrinsics.txt}"
output_dir="${OUTPUT_DIR:-${PROJECT_ROOT}/evaluation/output}"

echo "model ref: ${model_ref}"
echo "fixed model class: MetricAnythingDepthMap"
echo "dataset path: ${dataset_path}"
echo "raw type: ${raw_type} (kept for HAMMERDataset; ignored by RGB-only inference)"
echo "intrinsics path: ${intrinsics_path}"
echo "batch size: ${batch_size}"
echo "num workers: ${num_workers}"
echo "output dir: ${output_dir}"
echo "cleanup npy: ${cleanup_npy}"
echo "save vis: ${SAVE_VIS:-true}"

infer_args=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py"
    --model-path "${model_ref}"
    --dataset "${dataset_path}"
    --raw-type "${raw_type}"
    --output "${output_dir}"
    --batch-size "${batch_size}"
    --num-workers "${num_workers}"
    --intrinsics-path "${intrinsics_path}"
    --encoder "${arch}"
)

if [[ -n "${DEVICE:-}" ]]; then
    infer_args+=(--device "${DEVICE}")
fi

if [[ -n "${F_PX:-}" ]]; then
    infer_args+=(--f-px "${F_PX}")
fi

if [[ "${REQUIRE_FOCAL:-false}" == "true" ]]; then
    infer_args+=(--require-focal)
fi

if [[ "${LOCAL_FILES_ONLY:-false}" == "true" ]]; then
    infer_args+=(--local-files-only)
fi

if [[ "${SAVE_VIS:-true}" == "true" ]]; then
    infer_args+=(--save-vis)
fi

"${infer_args[@]}"

echo "evaluating predictions"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder "${arch}" \
    --model-path "${model_ref}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${raw_type}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}"
    find "${output_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi
