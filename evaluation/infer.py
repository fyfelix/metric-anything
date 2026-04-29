#!/usr/bin/env python3
"""HAMMER inference adapter for MetricAnything Student DepthMap."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
STUDENT_DEPTHMAP_DIR = PROJECT_ROOT / "models" / "student_depthmap"

for import_path in (str(EVAL_DIR), str(STUDENT_DEPTHMAP_DIR)):
    if import_path in sys.path:
        sys.path.remove(import_path)
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(1, str(STUDENT_DEPTHMAP_DIR))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MetricAnything Student DepthMap inference on HAMMER JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="yjh001/metricanything_student_depthmap",
        help="Local checkpoint path or Hugging Face model repo id.",
    )
    parser.add_argument(
        "--hf-filename",
        type=str,
        default="student_depthmap.pt",
        help="Checkpoint filename when --model-path is a Hugging Face repo id.",
    )
    parser.add_argument("--dataset", type=str, required=True, help="HAMMER JSONL path.")
    parser.add_argument("--output", type=str, default="output_dir", help="Prediction output directory.")
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="HAMMER raw depth type. Kept for dataset compatibility; this RGB-only model ignores raw depth.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="DataLoader batch size. Inference runs one image at a time.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g. cuda, cuda:0, cpu, or mps.")
    parser.add_argument("--f-px", type=float, default=None, help="Override focal length in pixels for every image.")
    parser.add_argument(
        "--intrinsics-path",
        type=str,
        default=None,
        help="Path to a 3x3 camera intrinsics txt file. Defaults to <dataset_dir>/intrinsics.txt.",
    )
    parser.add_argument(
        "--require-focal",
        action="store_true",
        help="Fail if focal length is not provided by --f-px, intrinsics txt, HAMMER JSONL, or sidecar JSON.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached Hugging Face files when loading a repo id.",
    )
    parser.add_argument("--save-vis", action="store_true", help="Save lightweight depth preview PNGs.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")

    # Compatibility flags accepted from the original wrapper. They are recorded in args.json.
    parser.add_argument("--encoder", type=str, default="vitl", help=argparse.SUPPRESS)
    parser.add_argument("--input-size", type=int, default=384, help=argparse.SUPPRESS)
    parser.add_argument("--depth-scale", type=float, default=1000.0, help=argparse.SUPPRESS)
    parser.add_argument("--max-depth", type=float, default=6.0, help=argparse.SUPPRESS)
    parser.add_argument("--image-min", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--image-max", type=float, default=5.0, help=argparse.SUPPRESS)
    return parser.parse_args()


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_transform():
    import torch
    from torchvision.transforms import v2

    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_rgb(rgb_path: str | Path):
    from PIL import Image

    path = Path(rgb_path)
    if not path.is_file():
        raise FileNotFoundError(f"RGB image does not exist: {path}")
    image = Image.open(path)
    if image.size[0] <= 0 or image.size[1] <= 0:
        raise ValueError(f"Invalid image size for {path}: {image.size}")
    return image.convert("RGB")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_focal_from_record(record: dict[str, Any]) -> float | None:
    for key in ("f_px", "fx", "focal", "focal_px", "focal_length", "focal_length_px"):
        value = _as_float(record.get(key))
        if value is not None:
            return value

    cam_in = record.get("cam_in")
    if isinstance(cam_in, (list, tuple)) and cam_in:
        return _as_float(cam_in[0])
    if isinstance(cam_in, dict):
        for key in ("fx", "f_x", "focal", "focal_px", "focal_length", "focal_length_px"):
            value = _as_float(cam_in.get(key))
            if value is not None:
                return value

    for key in ("intrinsics", "camera_intrinsics", "K"):
        intrinsics = record.get(key)
        if isinstance(intrinsics, (list, tuple)):
            if len(intrinsics) >= 1 and not isinstance(intrinsics[0], (list, tuple)):
                return _as_float(intrinsics[0])
            if intrinsics and isinstance(intrinsics[0], (list, tuple)) and intrinsics[0]:
                return _as_float(intrinsics[0][0])
        if isinstance(intrinsics, dict):
            for inner_key in ("fx", "f_x", "focal", "focal_px", "focal_length", "focal_length_px"):
                value = _as_float(intrinsics.get(inner_key))
                if value is not None:
                    return value

    return None


def load_focal_lookup(jsonl_path: str | Path) -> dict[str, float]:
    root = Path(jsonl_path).parent
    lookup: dict[str, float] = {}
    with Path(jsonl_path).open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            focal = extract_focal_from_record(record)
            rgb_rel = record.get("rgb")
            if focal is not None and rgb_rel:
                lookup[str((root / rgb_rel).resolve())] = focal
    return lookup


def default_intrinsics_path(dataset_path: str | Path) -> Path:
    return Path(dataset_path).parent / "intrinsics.txt"


def resolve_intrinsics_path(dataset_path: str | Path, intrinsics_path: str | None) -> Path:
    if intrinsics_path:
        candidate = Path(intrinsics_path).expanduser()
        if not candidate.is_absolute():
            project_candidate = PROJECT_ROOT / candidate
            if project_candidate.exists():
                return project_candidate
        return candidate
    return default_intrinsics_path(dataset_path)


def load_intrinsics_focal(intrinsics_path: str | Path) -> float | None:
    path = Path(intrinsics_path)
    if not path.is_file():
        return None

    values: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            values.extend(float(token) for token in line.split())

    if len(values) < 9:
        raise ValueError(f"Expected a 3x3 intrinsics matrix in {path}, got {len(values)} values.")

    return float(values[0])


def resolve_f_px(
    rgb_path: str | Path,
    image_width: int,
    override: float | None,
    intrinsics_focal: float | None = None,
    focal_lookup: dict[str, float] | None = None,
    require_focal: bool = False,
) -> tuple[float, str]:
    if override is not None:
        return float(override), "override"

    if intrinsics_focal is not None:
        return float(intrinsics_focal), "intrinsics_txt"

    lookup_key = str(Path(rgb_path).resolve())
    if focal_lookup and lookup_key in focal_lookup:
        return float(focal_lookup[lookup_key]), "hammer_jsonl"

    json_path = Path(rgb_path).with_suffix(".json")
    if json_path.is_file():
        try:
            with json_path.open("r", encoding="utf-8") as f:
                payload: dict[str, Any] = json.load(f)
            cam_in = payload.get("cam_in")
            if isinstance(cam_in, (list, tuple)) and cam_in:
                return float(cam_in[0]), "sidecar_json"
            LOGGER.warning("Invalid cam_in in %s; falling back to image width.", json_path)
        except Exception as exc:
            LOGGER.warning("Failed to read intrinsics from %s: %s; falling back to image width.", json_path, exc)

    if require_focal:
        raise ValueError(
            f"Missing focal length for {rgb_path}. Provide --f-px, add <dataset_dir>/intrinsics.txt, "
            "add focal/cam_in to HAMMER JSONL, or place a sidecar JSON next to the RGB image."
        )

    return float(image_width), "image_width_fallback"


def sample_id_from_rgb_path(rgb_path: str | Path) -> str:
    parts = str(rgb_path).split("/")
    if len(parts) < 4:
        raise ValueError(
            f"Cannot derive HAMMER sample id from {rgb_path!s}; expected at least four path components."
        )
    scene_name = parts[-4]
    stem = Path(parts[-1]).stem
    return f"{scene_name}#{stem}"


def load_model(args: argparse.Namespace, device: str):
    from depth_model import DEFAULT_CONFIG, MetricAnythingDepthMap

    if not STUDENT_DEPTHMAP_DIR.is_dir():
        raise FileNotFoundError(f"Missing student depthmap directory: {STUDENT_DEPTHMAP_DIR}")

    model_ref = args.model_path
    model_candidate = Path(model_ref).expanduser()
    if model_candidate.exists():
        model_ref = str(model_candidate.resolve())
    else:
        project_candidate = (PROJECT_ROOT / model_ref).expanduser()
        if project_candidate.exists():
            model_ref = str(project_candidate.resolve())

    hf_kwargs: dict[str, Any] = {"filename": args.hf_filename}
    if args.local_files_only:
        hf_kwargs["local_files_only"] = True

    old_cwd = Path.cwd()
    try:
        # depth_model.create_vit() uses torch.hub.load("network", source="local"),
        # which is relative to the official student_depthmap working directory.
        os.chdir(STUDENT_DEPTHMAP_DIR)
        model = MetricAnythingDepthMap.from_pretrained(
            model_ref,
            model_kwargs={
                "device": device,
                "config": DEFAULT_CONFIG,
            },
            **hf_kwargs,
        )
    finally:
        os.chdir(old_cwd)

    model.eval()
    LOGGER.info("Loaded MetricAnythingDepthMap from %s on %s", model_ref, device)
    return model


def save_depth_preview(depth: np.ndarray, save_path: Path, min_depth: float, max_depth: float) -> None:
    import cv2
    import numpy as np

    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        preview = np.zeros((*depth.shape, 3), dtype=np.uint8)
    else:
        clipped = np.clip(depth, min_depth, max_depth)
        normalized = ((clipped - min_depth) / max(max_depth - min_depth, 1e-6) * 255.0).astype(np.uint8)
        preview = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
        preview[~valid] = 0
    cv2.imwrite(str(save_path), preview)


def inference(args: argparse.Namespace) -> None:
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset JSONL does not exist: {dataset_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from dataset import HAMMERDataset
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    device = args.device or default_device()
    model = load_model(args, device)
    transform = make_transform()

    if "hammer" not in str(dataset_path).lower():
        raise ValueError(f"Invalid dataset for this adapter: {dataset_path}")
    dataset = HAMMERDataset(str(dataset_path), args.raw_type)
    intrinsics_path = resolve_intrinsics_path(dataset_path, args.intrinsics_path)
    intrinsics_focal = load_intrinsics_focal(intrinsics_path)
    focal_lookup = load_focal_lookup(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    args.resolved_model_module = "models/student_depthmap/depth_model.py"
    args.resolved_model_class = "MetricAnythingDepthMap"
    args.prediction_kind = "metric_depth_meters"
    args.raw_depth_used = False
    args.depth_range = dataset.depth_range
    args.intrinsics_path_resolved = str(intrinsics_path)
    args.intrinsics_focal = intrinsics_focal
    args.focal_records_loaded = len(focal_lookup)

    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    with torch.no_grad():
        focal_source_counts: dict[str, int] = {}
        for batch_items in tqdm(loader, desc="Infer"):
            rgb_paths, _raw_depth_paths, _gt_depth_paths = batch_items
            for rgb_path in rgb_paths:
                rgb_path = str(rgb_path)
                name = sample_id_from_rgb_path(rgb_path)
                image = load_rgb(rgb_path)
                input_tensor = transform(image).unsqueeze(0).to(device)
                f_px, focal_source = resolve_f_px(
                    rgb_path,
                    image.width,
                    args.f_px,
                    intrinsics_focal=intrinsics_focal,
                    focal_lookup=focal_lookup,
                    require_focal=args.require_focal,
                )
                focal_source_counts[focal_source] = focal_source_counts.get(focal_source, 0) + 1

                prediction = model.infer(input_tensor, f_px=f_px)
                depth = prediction["depth"].detach().cpu().numpy().squeeze().astype(np.float32)

                if depth.ndim != 2:
                    raise RuntimeError(f"Expected HxW depth for {rgb_path}, got shape {depth.shape}")

                np.save(output_dir / f"{name}.npy", depth)
                if args.save_vis:
                    save_depth_preview(
                        depth,
                        output_dir / f"{name}_depth_vis.png",
                        min_depth=float(dataset.depth_range[0]),
                        max_depth=float(dataset.depth_range[1]),
                    )

    with (output_dir / "focal_sources.json").open("w", encoding="utf-8") as f:
        json.dump(focal_source_counts, f, indent=2)


def main() -> None:
    args = parse_arguments()
    inference(args)


if __name__ == "__main__":
    main()
