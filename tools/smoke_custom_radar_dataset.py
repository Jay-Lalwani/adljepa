#!/usr/bin/env python3
"""Dataloader smoke checks for CustomRadarDataset."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.custom_radar.custom_radar_dataset import CustomRadarDataset
import radar_common as rc


def load_model_dataset_cfg(cfg_path: Path):
    model_cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    data_cfg = dict(model_cfg.get("DATA_CONFIG", {}))
    base_cfg = data_cfg.pop("_BASE_CONFIG_", None)
    merged = {}
    if base_cfg:
        base_path = Path(base_cfg)
        if not base_path.is_absolute():
            base_path = Path(__file__).resolve().parent / base_path
        merged.update(yaml.safe_load(open(base_path, encoding="utf-8")))
    merged.update(data_cfg)
    return EasyDict(merged), list(model_cfg["CLASS_NAMES"])


def parse_args():
    parser = argparse.ArgumentParser()
    rc.add_run_config_arg(parser)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--max-samples", type=int, default=25)
    args = parser.parse_args()
    args.run_cfg = rc.load_run_config(args.run_config)
    return args


def main() -> None:
    args = parse_args()
    cfg = args.run_cfg
    dataset_cfg, class_names = load_model_dataset_cfg(cfg.model_cfg)
    dataset_cfg.DATA_PATH = str(cfg.data_root)
    dataset_cfg.DATA_SPLIT.train = args.split
    dataset_cfg.DATA_SPLIT.test = args.split
    dataset_cfg.INFO_PATH.train = [f"custom_radar_infos_{args.split}.pkl"]
    dataset_cfg.INFO_PATH.test = [f"custom_radar_infos_{args.split}.pkl"]

    dataset = CustomRadarDataset(dataset_cfg, class_names, training=args.split == "train", root_path=cfg.data_root)
    expected_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
    counts = []
    labels = []
    for i in range(min(args.max_samples, len(dataset.sample_id_list))):
        frame_id = dataset.sample_id_list[i]
        points = dataset.get_lidar(frame_id)
        gt_boxes, gt_names = dataset.get_label(frame_id)
        if points.ndim != 2 or points.shape[1] != expected_features:
            raise AssertionError(f"{frame_id}: bad point shape {points.shape}")
        if gt_boxes.ndim != 2 or gt_boxes.shape[1] not in (9,):
            raise AssertionError(f"{frame_id}: bad gt shape {gt_boxes.shape}")
        if not np.isfinite(points).all() or not np.isfinite(gt_boxes).all():
            raise AssertionError(f"{frame_id}: NaN/Inf detected")
        if len(points) == 0:
            raise AssertionError(f"{frame_id}: zero radar points")
        counts.append(len(points))
        labels.append(len(gt_boxes))
    print(f"dataset={cfg.data_root}")
    print(f"split={args.split} samples_checked={len(counts)} total_ids={len(dataset.sample_id_list)}")
    print(f"point_features={expected_features}")
    print(f"points_per_frame min/avg/max={min(counts)}/{np.mean(counts):.1f}/{max(counts)}")
    print(f"labels_per_frame min/avg/max={min(labels)}/{np.mean(labels):.1f}/{max(labels)}")


if __name__ == "__main__":
    main()
