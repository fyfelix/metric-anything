#!/usr/bin/env python3
import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (
    REPO_ROOT,
    REPO_ROOT / "models" / "student_depthmap",
    REPO_ROOT / "models" / "student_pointmap",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from depth_model import DEFAULT_CONFIG, MetricAnythingDepthMap
from moge.model.v2 import MoGeModel
from scripts.utils.test_datasets import ClearPoseDataset, HAMMERDataset, prediction_name


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="MetricAnything batch inference for the HAMMER/ClearPose benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path or HF repo id")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["student_pointmap", "student_depthmap"],
        help="Override model type detection from --model-path",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output", type=str, default="output_dir", help="Output directory")
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Camera raw depth type; kept for benchmark dataset selection",
    )
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--batch-size", "-b", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--intrinsics-path",
        type=str,
        default=None,
        help=(
            "3x3 intrinsics matrix for student_depthmap; used after --f-px "
            "and per-image JSON cam_in, before image-width fallback"
        ),
    )
    parser.add_argument(
        "--f-px",
        "--f_px",
        dest="f_px",
        type=float,
        default=None,
        help="Focal length in pixels for student_depthmap; overrides all inferred focal values",
    )
    parser.add_argument(
        "--fov-x",
        type=float,
        default=None,
        help="Horizontal field of view in degrees for student_pointmap; defaults to model inference",
    )
    parser.add_argument(
        "--prediction-resize-mode",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest"],
        help="Interpolation mode used to resize predictions back to GT resolution",
    )
    return parser.parse_args()


def parse_device(value):
    if value is not None:
        return torch.device(f"cuda:{value}" if value.isdigit() else value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate_inputs(args, model_type):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file does not exist: {args.dataset}")
    if (
        model_type == "student_depthmap"
        and args.intrinsics_path is not None
        and not os.path.exists(args.intrinsics_path)
    ):
        raise FileNotFoundError(f"Intrinsics file does not exist: {args.intrinsics_path}")
    if args.f_px is not None and args.f_px <= 0:
        raise ValueError(f"--f-px must be positive, got {args.f_px}")
    if args.fov_x is not None and args.fov_x <= 0:
        raise ValueError(f"--fov-x must be positive, got {args.fov_x}")
    os.makedirs(args.output, exist_ok=True)


def detect_model_type(model_path, explicit):
    if explicit is not None:
        return explicit

    normalized = model_path.lower()
    if "pointmap" in normalized:
        return "student_pointmap"
    if "depthmap" in normalized:
        return "student_depthmap"
    raise ValueError(
        f"Could not infer model type from '{model_path}'. "
        "Pass --model-type student_pointmap or --model-type student_depthmap."
    )


def load_dataset(args):
    dataset_name = args.dataset.lower()
    if "clearpose" in dataset_name:
        if args.raw_type != "d435":
            raise ValueError("ClearPose dataset only supports d435 raw type")
        return ClearPoseDataset(args.dataset)
    if "hammer" in dataset_name:
        return HAMMERDataset(args.dataset, args.raw_type)
    raise ValueError(f"Invalid dataset: {args.dataset}")


def batch_collate(batch):
    rgb_paths = [item[0] for item in batch]
    raw_depth_paths = [item[1] for item in batch]
    gt_depth_paths = [item[2] for item in batch]
    return rgb_paths, raw_depth_paths, gt_depth_paths


def build_depthmap_transform():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_pointmap_input(rgb_path):
    image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb / 255.0).float().permute(2, 0, 1)


def load_depthmap_input(rgb_path, transform):
    image = Image.open(rgb_path).convert("RGB")
    return transform(image)


def load_gt_shape(gt_path):
    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt is None:
        raise ValueError(f"Could not load GT depth from {gt_path}")
    return gt.shape[:2]


def load_intrinsics_fx(intrinsics_path):
    intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Invalid intrinsics matrix shape: {intrinsics.shape}")
    fx = float(intrinsics[0, 0])
    if fx <= 0:
        raise ValueError(f"Invalid fx in intrinsics matrix: {fx}")
    return fx


def resolve_intrinsics_path(args):
    if args.intrinsics_path:
        return args.intrinsics_path

    candidate = Path(args.dataset).resolve().parent / "intrinsics.txt"
    if candidate.exists():
        return str(candidate)
    return None


def resolve_global_intrinsics_fx(args):
    intrinsics_path = resolve_intrinsics_path(args)
    if intrinsics_path is not None:
        return load_intrinsics_fx(intrinsics_path), f"intrinsics:{intrinsics_path}"

    return None, None


def load_sidecar_json_fx(rgb_path):
    json_path = Path(rgb_path).with_suffix(".json")
    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"Failed to read intrinsics from {json_path}: {exc}. Using fallback.")
        return None

    cam_in = payload.get("cam_in")
    if not isinstance(cam_in, (list, tuple)) or len(cam_in) < 1:
        print(f"Invalid cam_in in {json_path}. Using fallback.")
        return None

    try:
        fx = float(cam_in[0])
    except (TypeError, ValueError):
        print(f"Invalid fx in {json_path}: {cam_in[0]}. Using fallback.")
        return None

    if fx <= 0:
        print(f"Invalid fx in {json_path}: {fx}. Using fallback.")
        return None
    return fx


def resolve_depthmap_record_f_px(args, rgb_path, image_width, global_f_px, global_source):
    if args.f_px is not None:
        return float(args.f_px), "cli"

    sidecar_fx = load_sidecar_json_fx(rgb_path)
    if sidecar_fx is not None:
        return sidecar_fx, "json"

    if global_f_px is not None:
        return float(global_f_px), global_source

    return float(image_width), "image_width"


def resize_prediction(pred_depth, target_shape, mode="bilinear"):
    pred_depth = np.asarray(pred_depth).squeeze()
    if pred_depth.ndim != 2:
        raise ValueError(f"Prediction must be 2D after squeeze, got {pred_depth.shape}")
    if pred_depth.shape == target_shape:
        return pred_depth.astype(np.float32)

    tensor = torch.from_numpy(pred_depth.astype(np.float32))
    resize_kwargs = {
        "size": target_shape,
        "mode": mode,
    }
    if mode == "bilinear":
        resize_kwargs["align_corners"] = False

    tensor = F.interpolate(tensor[None, None], **resize_kwargs)[0, 0]
    return tensor.detach().cpu().numpy().astype(np.float32)


def iter_prediction_arrays(pred_depths, expected_count):
    pred_depths = np.asarray(pred_depths)
    if expected_count == 1 and pred_depths.ndim == 2:
        return [pred_depths]
    if pred_depths.ndim == 3 and pred_depths.shape[0] == expected_count:
        return [pred_depths[i] for i in range(expected_count)]
    raise ValueError(
        f"Unexpected prediction shape: {pred_depths.shape}, batch={expected_count}"
    )


def load_model(args, model_type, device):
    print(f"Loading {model_type} from {args.model_path}")
    if model_type == "student_pointmap":
        model = MoGeModel.from_pretrained(args.model_path)
    elif model_type == "student_depthmap":
        model = MetricAnythingDepthMap.from_pretrained(
            args.model_path,
            model_kwargs={"device": str(device), "config": DEFAULT_CONFIG},
            filename="student_depthmap.pt",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    return model


@torch.no_grad()
def predict_pointmap_batch(model, records, device, fov_x):
    images = torch.stack([record["image"] for record in records]).to(device)
    output = model.infer(images, fov_x=fov_x, use_fp16=device.type == "cuda")
    pred_depths = output["depth"].detach().cpu().numpy()
    return iter_prediction_arrays(pred_depths, len(records))


def predict_pointmap_single(model, record, device, fov_x):
    output = model.infer(
        record["image"].unsqueeze(0).to(device),
        fov_x=fov_x,
        use_fp16=device.type == "cuda",
    )
    return iter_prediction_arrays(output["depth"].detach().cpu().numpy(), 1)[0]


def depthmap_f_px_tensor(records, device):
    f_px_values = [record["f_px"] for record in records]
    if all(value == f_px_values[0] for value in f_px_values):
        return f_px_values[0]
    return torch.tensor(f_px_values, dtype=torch.float32, device=device).view(-1, 1, 1, 1)


@torch.no_grad()
def predict_depthmap_batch(model, records, device):
    images = torch.stack([record["image"] for record in records]).to(device)
    output = model.infer(images, f_px=depthmap_f_px_tensor(records, device))
    pred_depths = output["depth"].detach().cpu().numpy()
    return iter_prediction_arrays(pred_depths, len(records))


def predict_depthmap_single(model, record, device):
    output = model.infer(
        record["image"].unsqueeze(0).to(device),
        f_px=record["f_px"],
    )
    return iter_prediction_arrays(output["depth"].detach().cpu().numpy(), 1)[0]


def save_prediction(args, record, pred):
    pred = resize_prediction(
        pred,
        record["target_shape"],
        mode=args.prediction_resize_mode,
    )
    np.save(os.path.join(args.output, f"{record['name']}.npy"), pred)


def build_records(
    args,
    model_type,
    rgb_paths,
    gt_depth_paths,
    transform,
    global_f_px,
    global_f_px_source,
):
    records = []
    for rgb_path, gt_depth_path in zip(rgb_paths, gt_depth_paths):
        name = prediction_name(rgb_path, args.dataset)
        try:
            if model_type == "student_pointmap":
                image = load_pointmap_input(rgb_path)
                resolved_f_px = None
                f_px_source = None
            else:
                image = load_depthmap_input(rgb_path, transform)
                resolved_f_px, f_px_source = resolve_depthmap_record_f_px(
                    args,
                    rgb_path,
                    image.shape[-1],
                    global_f_px,
                    global_f_px_source,
                )

            records.append(
                {
                    "name": name,
                    "image": image,
                    "target_shape": load_gt_shape(gt_depth_path),
                    "f_px": resolved_f_px,
                    "f_px_source": f_px_source,
                }
            )
        except Exception as exc:
            print(f"Failed to load sample {rgb_path}: {exc}")
    return records


@torch.no_grad()
def inference(args):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    model_type = detect_model_type(args.model_path, args.model_type)
    validate_inputs(args, model_type)
    device = parse_device(args.device)

    with open(os.path.join(args.output, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    model = load_model(args, model_type, device)

    transform = None
    global_f_px = None
    global_f_px_source = None
    f_px_sources = Counter()
    if model_type == "student_depthmap":
        transform = build_depthmap_transform()
        global_f_px, global_f_px_source = resolve_global_intrinsics_fx(args)
        if args.f_px is not None:
            print(f"student_depthmap f_px={args.f_px:.4f} source=cli")
        elif global_f_px is not None:
            print(f"student_depthmap global f_px={global_f_px:.4f} source={global_f_px_source}")
        else:
            print("student_depthmap f_px source: per-image JSON or image width fallback")
    elif args.fov_x is not None:
        print(f"student_pointmap fov_x={args.fov_x:.4f} degrees")

    dataset = load_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=batch_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    for rgb_paths, raw_depth_paths, gt_depth_paths in tqdm(
        dataloader, desc="Processing batches"
    ):
        del raw_depth_paths
        records = build_records(
            args,
            model_type,
            rgb_paths,
            gt_depth_paths,
            transform,
            global_f_px,
            global_f_px_source,
        )
        if not records:
            continue
        f_px_sources.update(
            record["f_px_source"] for record in records if record["f_px_source"]
        )

        try:
            if model_type == "student_pointmap":
                predictions = predict_pointmap_batch(model, records, device, args.fov_x)
            else:
                predictions = predict_depthmap_batch(model, records, device)

            for record, pred in zip(records, predictions):
                save_prediction(args, record, pred)
        except RuntimeError as exc:
            print(f"Batch shape mismatch or OOM, falling back to single-sample mode: {exc}")
            for record in records:
                if model_type == "student_pointmap":
                    pred = predict_pointmap_single(model, record, device, args.fov_x)
                else:
                    pred = predict_depthmap_single(model, record, device)
                save_prediction(args, record, pred)

    if f_px_sources:
        print("student_depthmap f_px source counts:")
        for source, count in sorted(f_px_sources.items()):
            print(f"  {source}: {count}")


if __name__ == "__main__":
    inference(parse_arguments())
