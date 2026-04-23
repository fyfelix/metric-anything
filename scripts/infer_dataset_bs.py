#!/usr/bin/env python3

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "models" / "student_depthmap"))
sys.path.insert(0, str(REPO_ROOT / "models" / "student_pointmap"))

from depth_model import DEFAULT_CONFIG, MetricAnythingDepthMap
from moge.model.v2 import MoGeModel
from scripts.utils.test_datasets import ClearPoseDataset, HAMMERDataset, sample_name_from_rgb_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="LingBot-Depth Batch Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model directory or .pt file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset file")
    parser.add_argument("--output", type=str, default="output_dir", help="Output directory")
    parser.add_argument("--raw-type", type=str, required=True, choices=["d435", "l515", "tof"], help="Camera raw type")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Scale factor for depth values")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--max-depth", type=float, default=6.0, help="Max depth limit for raw truncation")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Min valid depth")
    parser.add_argument(
        "--intrinsics-path",
        type=str,
        default="data/HAMMER/intrinsics.txt",
        help="Path to the camera intrinsics matrix used by student_depthmap",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["student_pointmap", "student_depthmap"],
        help="Optional override for automatic model type detection",
    )
    return parser.parse_args()


def batch_collate(batch):
    rgb_paths = [item[0] for item in batch]
    raw_depth_paths = [item[1] for item in batch]
    gt_depth_paths = [item[2] for item in batch]
    return rgb_paths, raw_depth_paths, gt_depth_paths


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def detect_model_type(model_path: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit

    name = Path(model_path).name.lower()
    if "pointmap" in name:
        return "student_pointmap"
    if "depthmap" in name:
        return "student_depthmap"
    raise ValueError(
        f"Could not infer model type from '{model_path}'. "
        "Please pass --model-type explicitly."
    )


def create_dataset(args):
    if "clearpose" in args.dataset.lower():
        return ClearPoseDataset(args.dataset)
    if "hammer" in args.dataset.lower():
        return HAMMERDataset(args.dataset, args.raw_type)
    raise ValueError(f"Invalid dataset: {args.dataset}")


def build_depth_transform():
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def load_pointmap_input(rgb_path: str) -> torch.Tensor:
    image_bgr = cv2.imread(rgb_path)
    if image_bgr is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image_rgb / 255.0).float().permute(2, 0, 1)


def load_depthmap_input(rgb_path: str, transform: v2.Compose) -> torch.Tensor:
    image = Image.open(rgb_path).convert("RGB")
    return transform(image)


def load_intrinsics_fx(intrinsics_path: str) -> float:
    intrinsics = np.loadtxt(intrinsics_path, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"Invalid intrinsics matrix shape: {intrinsics.shape}")
    fx = float(intrinsics[0, 0])
    if fx <= 0:
        raise ValueError(f"Invalid fx in intrinsics matrix: {fx}")
    return fx


def load_model(model_type: str, model_path: str, device: torch.device):
    if model_type == "student_pointmap":
        model = MoGeModel.from_pretrained(model_path).to(device)
    elif model_type == "student_depthmap":
        model = MetricAnythingDepthMap.from_pretrained(
            model_path,
            model_kwargs={"device": str(device), "config": DEFAULT_CONFIG},
            filename="student_depthmap.pt",
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()
    return model


def save_prediction(output_dir: str, name: str, pred: np.ndarray) -> None:
    np.save(os.path.join(output_dir, f"{name}.npy"), pred.astype(np.float32))


def run_pointmap_batch(model, device, rgb_paths, names):
    batch_tensors = []
    valid_names = []

    for rgb_path, name in zip(rgb_paths, names):
        try:
            batch_tensors.append(load_pointmap_input(rgb_path))
            valid_names.append(name)
        except Exception as exc:
            logger.error(f"Error loading {rgb_path}: {exc}")

    if not batch_tensors:
        return []

    use_fp16 = device.type == "cuda"
    try:
        batch_images = torch.stack(batch_tensors).to(device)
        output = model.infer(batch_images, use_fp16=use_fp16)
        pred_depths = output["depth"].detach().cpu().numpy()
        pred_masks = output.get("mask")
        if pred_masks is not None:
            pred_masks = pred_masks.detach().cpu().numpy().astype(bool)

        predictions = []
        for i, name in enumerate(valid_names):
            pred = pred_depths[i] if pred_depths.ndim == 3 else pred_depths
            if pred_masks is not None:
                mask = pred_masks[i] if pred_masks.ndim == 3 else pred_masks
                pred = np.where(mask, pred, np.inf)
            predictions.append((name, pred))
        return predictions
    except RuntimeError as exc:
        logger.warning(f"Images in batch have different sizes, processing one by one. Error: {exc}")

    predictions = []
    for tensor, name in zip(batch_tensors, valid_names):
        output = model.infer(tensor.unsqueeze(0).to(device), use_fp16=use_fp16)
        pred = output["depth"].detach().cpu().numpy()
        mask = output.get("mask")
        if mask is not None:
            mask = mask.detach().cpu().numpy().astype(bool)
            pred = np.where(mask, pred, np.inf)
        predictions.append((name, pred))
    return predictions


def run_depthmap_batch(model, device, rgb_paths, names, transform, f_px):
    batch_tensors = []
    valid_names = []

    for rgb_path, name in zip(rgb_paths, names):
        try:
            batch_tensors.append(load_depthmap_input(rgb_path, transform))
            valid_names.append(name)
        except Exception as exc:
            logger.error(f"Error loading {rgb_path}: {exc}")

    if not batch_tensors:
        return []

    try:
        batch_images = torch.stack(batch_tensors).to(device)
        output = model.infer(batch_images, f_px=f_px)
        pred_depths = output["depth"].detach().cpu().numpy()

        predictions = []
        for i, name in enumerate(valid_names):
            pred = pred_depths[i] if pred_depths.ndim == 3 else pred_depths
            predictions.append((name, pred))
        return predictions
    except RuntimeError as exc:
        logger.warning(f"Images in batch have different sizes, processing one by one. Error: {exc}")

    predictions = []
    for tensor, name in zip(batch_tensors, valid_names):
        output = model.infer(tensor.unsqueeze(0).to(device), f_px=f_px)
        pred = output["depth"].detach().cpu().numpy()
        predictions.append((name, pred))
    return predictions


@torch.no_grad()
def inference(args):
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    device = get_device()
    model_type = detect_model_type(args.model_path, args.model_type)
    logger.info(f"Loading {model_type} from {args.model_path} on {device}...")
    model = load_model(model_type, args.model_path, device)
    logger.info("Model loaded successfully.")

    transform = None
    f_px = None
    if model_type == "student_depthmap":
        transform = build_depth_transform()
        f_px = load_intrinsics_fx(args.intrinsics_path)
        logger.info(f"Using fx={f_px:.4f} from {args.intrinsics_path}")

    dataset = create_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=batch_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    for rgb_paths, raw_depth_paths, gt_depth_paths in tqdm(dataloader, desc="Processing batches"):
        del raw_depth_paths, gt_depth_paths
        names = [sample_name_from_rgb_path(rgb_path) for rgb_path in rgb_paths]

        if model_type == "student_pointmap":
            predictions = run_pointmap_batch(model, device, rgb_paths, names)
        else:
            predictions = run_depthmap_batch(model, device, rgb_paths, names, transform, f_px)

        for name, pred in predictions:
            save_prediction(args.output, name, pred)


if __name__ == "__main__":
    inference(parse_arguments())
