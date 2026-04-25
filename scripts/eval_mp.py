#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from os.path import exists, join
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.metric import (
    abs_relative_difference,
    delta1_acc,
    delta4_acc_105,
    delta5_acc110,
    mae_linear,
    rmse_linear,
)
from scripts.utils.test_datasets import ClearPoseDataset, HAMMERDataset, prediction_name


METRIC_FIELDS = [
    "L1",
    "rmse_linear",
    "abs_relative_difference",
    "delta4_acc_105",
    "delta5_acc110",
    "delta1_acc",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate MetricAnything predictions on the HAMMER/ClearPose benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output", type=str, required=True, help="Prediction output directory")
    parser.add_argument("--raw-type", type=str, default="d435", help="Camera raw type")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--align", action="store_true", help="Fit scale/shift per image")
    return parser.parse_args()


def load_gt_depth(depth_path, depth_scale, max_depth, min_depth):
    depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_gt is None:
        raise ValueError(f"Could not load GT depth from {depth_path}")

    if depth_gt.ndim == 3:
        depth_gt = depth_gt[:, :, 0]

    depth_gt = np.asarray(depth_gt).astype(np.float32) / depth_scale
    valid_mask = np.isfinite(depth_gt) & (depth_gt >= min_depth) & (depth_gt <= max_depth)
    depth_gt[~valid_mask] = min_depth
    return depth_gt, valid_mask


def load_dataset(args):
    dataset_name = args.dataset.lower()
    if "clearpose" in dataset_name:
        if args.raw_type != "d435":
            raise ValueError("ClearPose dataset only supports d435 raw type")
        return ClearPoseDataset(args.dataset)
    if "hammer" in dataset_name:
        return HAMMERDataset(args.dataset, args.raw_type)
    raise ValueError(f"Invalid dataset: {args.dataset}")


class EvalDataset(Dataset):
    def __init__(self, dataset, prediction_path, args):
        self.dataset = dataset
        self.prediction_path = prediction_path
        self.args = args
        self.align = args.align

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rgb_path, _, gt_path = self.dataset[idx]
        name = prediction_name(rgb_path, self.args.dataset)
        pred_file = join(self.prediction_path, f"{name}.npy")

        if not exists(pred_file):
            raise FileNotFoundError(f"Prediction not found: {pred_file}")

        pred = np.load(pred_file).astype(np.float32).squeeze()
        if pred.ndim != 2:
            raise ValueError(f"Prediction must be 2D after squeeze: {pred_file}, {pred.shape}")

        depth_gt, valid_mask = load_gt_depth(
            gt_path,
            self.args.depth_scale,
            self.args.max_depth,
            self.args.min_depth,
        )

        if pred.shape != depth_gt.shape:
            raise ValueError(
                f"Prediction/GT shape mismatch for {name}: pred={pred.shape}, gt={depth_gt.shape}"
            )

        pred_invalid_mask = np.logical_or(np.isnan(pred), np.isinf(pred))
        if pred_invalid_mask.sum() > 0:
            valid_mask = valid_mask & ~pred_invalid_mask

        if not np.any(valid_mask):
            raise ValueError(f"No valid pixels for {name}")

        if self.align:
            depth_gt_reshaped = depth_gt[valid_mask].reshape((-1, 1))
            pred_reshaped = pred[valid_mask].reshape((-1, 1))

            ones = np.ones_like(pred_reshaped)
            system = np.concatenate([pred_reshaped, ones], axis=-1)
            scale, shift = np.linalg.lstsq(system, depth_gt_reshaped, rcond=None)[0]
            pred_reshaped = scale * pred_reshaped + shift
            pred_reshaped = np.clip(pred_reshaped, a_min=self.args.min_depth, a_max=None)

            return {
                "name": name,
                "pred": pred_reshaped.astype(np.float32),
                "gt": depth_gt_reshaped.astype(np.float32),
                "mask": np.ones_like(pred_reshaped, dtype=bool),
            }

        return {
            "name": name,
            "pred": pred.astype(np.float32),
            "gt": depth_gt.astype(np.float32),
            "mask": valid_mask.astype(bool),
        }


def write_all_metrics(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name"] + METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def mean_metrics(rows):
    if not rows:
        return {field: float("nan") for field in METRIC_FIELDS}
    return {
        field: float(np.mean([float(row[field]) for row in rows]))
        for field in METRIC_FIELDS
    }


def main():
    args = parse_arguments()
    os.makedirs(args.output, exist_ok=True)

    dataset = load_dataset(args)
    depth_range = getattr(dataset, "depth_range", None)
    if depth_range is not None:
        args.min_depth = depth_range[0]
        args.max_depth = depth_range[1]

    with open(join(args.output, "eval_args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    eval_dataset = EvalDataset(dataset, args.output, args)
    if len(eval_dataset) == 0:
        print("No samples found in dataset for evaluation.")
        return

    print(f"Eval depth range: min={args.min_depth}, max={args.max_depth}")
    loader = DataLoader(
        eval_dataset,
        batch_size=1 if args.align else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics = []

    for batch in tqdm(loader, desc="Evaluating Metrics"):
        names = batch["name"]
        pred_depth_ts = batch["pred"].to(device)
        gt_depth_ts = batch["gt"].to(device)
        mask_ts = batch["mask"].to(device)

        l1 = mae_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        rmse = rmse_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        abs_rel = abs_relative_difference(
            pred_depth_ts, gt_depth_ts, mask_ts, reduction="none"
        )
        d4 = delta4_acc_105(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        d5 = delta5_acc110(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")
        d1 = delta1_acc(pred_depth_ts, gt_depth_ts, mask_ts, reduction="none")

        metric_arrays = {
            "L1": l1.detach().cpu().numpy(),
            "rmse_linear": rmse.detach().cpu().numpy(),
            "abs_relative_difference": abs_rel.detach().cpu().numpy(),
            "delta4_acc_105": d4.detach().cpu().numpy(),
            "delta5_acc110": d5.detach().cpu().numpy(),
            "delta1_acc": d1.detach().cpu().numpy(),
        }

        for i, name in enumerate(names):
            row = {"name": name}
            for field, values in metric_arrays.items():
                row[field] = float(values[i])
            all_metrics.append(row)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    align_tag = str(bool(args.align))
    csv_out = join(args.output, f"all_metrics_{current_time}_{align_tag}.csv")
    json_out = join(args.output, f"mean_metrics_{current_time}_{align_tag}.json")

    write_all_metrics(csv_out, all_metrics)
    means = mean_metrics(all_metrics)
    with open(json_out, "w", encoding="utf-8") as f:
        f.write(json.dumps([means], ensure_ascii=False) + "\n")

    print("=" * 50)
    print(" ".join(f"{field}={means[field]:.6f}" for field in METRIC_FIELDS))
    print("=" * 50)
    print(f"Evaluation completed. Metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
