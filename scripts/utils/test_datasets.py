from __future__ import annotations

import json
import os
from glob import glob
from os.path import dirname, join
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset


MAX_RETRIES = 1000


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    """Load benchmark RGB and raw depth in meters.

    MetricAnything student models use RGB as model input. This helper is kept
    for benchmark compatibility and for scripts that inspect the selected raw
    depth stream.
    """
    rgb_src = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_src is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    rgb_src = np.asarray(rgb_src[:, :, ::-1])

    if not depth_path:
        raise ValueError("Raw depth path is empty")

    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    if depth_low_res.ndim == 3:
        depth_low_res = depth_low_res[:, :, 0]

    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    invalid_mask = (
        ~np.isfinite(depth_low_res)
        | (depth_low_res <= 0.0)
        | (depth_low_res > max_depth)
    )
    depth_low_res[invalid_mask] = 0.0

    return rgb_src, depth_low_res


def prediction_name(rgb_path, dataset_path):
    parts = rgb_path.split("/")
    dataset_name = dataset_path.lower()

    if "hammer" in dataset_name:
        if len(parts) < 4:
            return Path(rgb_path).stem
        scene_name = parts[-4]
        return scene_name + "#" + Path(parts[-1]).stem
    if "clearpose" in dataset_name:
        if len(parts) < 3:
            return Path(rgb_path).stem
        return "#".join(parts[-3:-1]) + "#" + Path(parts[-1]).stem

    raise ValueError(f"Invalid dataset: {dataset_path}")


def sample_name_from_rgb_path(rgb_path: str) -> str:
    """Backward-compatible HAMMER-style prediction name."""
    parts = rgb_path.split("/")
    if len(parts) >= 4:
        scene_name = parts[-4]
        return scene_name + "#" + Path(parts[-1]).stem
    return Path(rgb_path).stem


def _resolve_raw_depth(root, item, raw_type):
    raw_type = raw_type.lower()
    field_by_type = {
        "d435": "d435_depth",
        "l515": "l515_depth",
        "tof": "tof_depth",
    }
    if raw_type not in field_by_type:
        raise ValueError(f"Invalid raw type: {raw_type}")

    raw_field = field_by_type[raw_type]
    raw_value = item.get(raw_field, "")
    return join(root, raw_value) if raw_value else ""


class HAMMERDataset(Dataset):
    def __init__(self, jsonl_path, raw_type="d435", require_raw_depth=False):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.raw_type = raw_type
        self.require_raw_depth = require_raw_depth
        self.data = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.depth_range = self.data[0].get("depth-range", [0.01, 6.0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                item = self.data[idx]
                rgb = join(self.root, item["rgb"])
                raw_depth = _resolve_raw_depth(self.root, item, self.raw_type)
                gt_depth = join(self.root, item["depth"])

                if not (os.path.exists(rgb) and os.path.exists(gt_depth)):
                    raise FileNotFoundError(f"Missing RGB or GT depth for sample {idx}")
                if self.require_raw_depth and not os.path.exists(raw_depth):
                    raise FileNotFoundError(f"Missing raw depth for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    print(
                        f"Error loading sample {idx} in HAMMERDataset: {exc}. "
                        f"Retrying {attempt + 1}/{MAX_RETRIES}..."
                    )
                    idx = np.random.randint(0, len(self.data))
                else:
                    print("Failed to load sample after retries.")
                    raise


class ClearPoseDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        max_length_each_sequence=300,
        require_raw_depth=False,
    ):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.require_raw_depth = require_raw_depth
        self.data = []
        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item["depth-range"]

                rgbs = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["rgb-suffix"]))
                )[:max_length_each_sequence]
                raw_depths = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["raw_depth-suffix"]))
                )[:max_length_each_sequence]
                gt_depths = sorted(
                    glob(join(self.root, item["rgb"], "*" + item["depth-suffix"]))
                )[:max_length_each_sequence]

                self.rgbs.extend(rgbs)
                self.raw_depths.extend(raw_depths)
                self.gt_depths.extend(gt_depths)
                self.data.append(item)

        self.depth_range = depth_range

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                rgb = self.rgbs[idx]
                raw_depth = self.raw_depths[idx] if idx < len(self.raw_depths) else ""
                gt_depth = self.gt_depths[idx]

                if not (os.path.exists(rgb) and os.path.exists(gt_depth)):
                    raise FileNotFoundError(f"Missing RGB or GT depth for sample {idx}")
                if self.require_raw_depth and not os.path.exists(raw_depth):
                    raise FileNotFoundError(f"Missing raw depth for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    print(
                        f"Error loading sample {idx} in ClearPoseDataset: {exc}. "
                        f"Retrying {attempt + 1}/{MAX_RETRIES}..."
                    )
                    idx = np.random.randint(0, len(self.rgbs))
                else:
                    print("Failed to load sample after retries.")
                    raise
