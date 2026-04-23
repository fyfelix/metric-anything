from __future__ import annotations

import json
import os
from glob import glob
from os.path import dirname, join
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image
from loguru import logger
from torch.utils.data import Dataset

MAX_RETRIES = 1000


def sample_name_from_rgb_path(rgb_path: str) -> str:
    tmp = rgb_path.split("/")
    if len(tmp) >= 4:
        scene_name = tmp[-4]
        return scene_name + "#" + tmp[-1].split(".")[0]
    return Path(rgb_path).stem


def load_images(rgb_path, depth_path, depth_scale, max_depth):
    rgb_bgr = cv2.imread(rgb_path)
    if rgb_bgr is None:
        raise ValueError(f"Could not load RGB image from {rgb_path}")
    rgb_src = np.asarray(rgb_bgr[:, :, ::-1])

    depth_low_res = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_low_res is None:
        raise ValueError(f"Could not load depth image from {depth_path}")

    depth_low_res = np.asarray(depth_low_res).astype(np.float32) / depth_scale
    depth_low_res[depth_low_res > max_depth] = 0.0

    simi_depth_low_res = np.zeros_like(depth_low_res)
    simi_depth_low_res[depth_low_res > 0] = 1 / depth_low_res[depth_low_res > 0]

    return rgb_src, depth_low_res, simi_depth_low_res


def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    cm_func = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm_func(depth, bytes=False)[:, :, :, 0:3]
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np


def concat_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    return new_im


class HAMMERDataset(Dataset):
    def __init__(self, jsonl_path, raw_type="d435"):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

        self.raw_type = raw_type
        self.depth_range = self.data[0].get("depth-range", [0.01, 6.0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                item = self.data[idx]

                rgb = join(self.root, item["rgb"])

                if self.raw_type.lower() == "d435":
                    raw_depth = join(self.root, item["d435_depth"])
                elif self.raw_type.lower() == "l515":
                    raw_depth = join(self.root, item["l515_depth"])
                elif self.raw_type.lower() == "tof":
                    raw_depth = join(self.root, item["tof_depth"])
                else:
                    raise ValueError(f"Invalid raw type: {self.raw_type}")

                gt_depth = join(self.root, item["depth"])

                if not (os.path.exists(rgb) and os.path.exists(raw_depth) and os.path.exists(gt_depth)):
                    raise FileNotFoundError(f"Missing file(s) for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Error loading sample {idx} in HAMMERDataset: {exc}. "
                        f"Retrying {attempt + 1}/{MAX_RETRIES}..."
                    )
                    idx = np.random.randint(0, len(self.data))
                else:
                    logger.error(f"Failed to load sample after {MAX_RETRIES} retries.")
                    raise


class ClearPoseDataset(Dataset):
    def __init__(self, jsonl_path, max_length_each_sequence=300):
        self.jsonl_path = jsonl_path
        self.root = dirname(jsonl_path)
        self.data = []

        self.rgbs = []
        self.raw_depths = []
        self.gt_depths = []

        depth_range = None

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                if depth_range is None:
                    depth_range = item["depth-range"]

                rgb = sorted(glob(join(self.root, item["rgb"], "*" + item["rgb-suffix"])))[:max_length_each_sequence]
                raw_depth = sorted(glob(join(self.root, item["rgb"], "*" + item["raw_depth-suffix"])))[:max_length_each_sequence]
                gt_depth = sorted(glob(join(self.root, item["rgb"], "*" + item["depth-suffix"])))[:max_length_each_sequence]

                self.rgbs.extend(rgb)
                self.raw_depths.extend(raw_depth)
                self.gt_depths.extend(gt_depth)
                self.data.append(item)
        self.depth_range = depth_range

    def __len__(self):
        return len(self.rgbs)

    def __getitem__(self, idx):
        for attempt in range(MAX_RETRIES + 1):
            try:
                rgb = self.rgbs[idx]
                raw_depth = self.raw_depths[idx]
                gt_depth = self.gt_depths[idx]

                if not (os.path.exists(rgb) and os.path.exists(raw_depth) and os.path.exists(gt_depth)):
                    raise FileNotFoundError(f"Missing file(s) for sample {idx}")

                return rgb, raw_depth, gt_depth
            except Exception as exc:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        f"Error loading sample {idx} in ClearPoseDataset: {exc}. "
                        f"Retrying {attempt + 1}/{MAX_RETRIES}..."
                    )
                    idx = np.random.randint(0, len(self.rgbs))
                else:
                    logger.error(f"Failed to load sample after {MAX_RETRIES} retries.")
                    raise
