#!/usr/bin/env python3
"""Generate CustomRadarDataset infos and GT database."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import yaml
from easydict import EasyDict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.datasets.custom_radar.custom_radar_dataset import CustomRadarDataset
import radar_common as rc


def load_model_dataset_cfg(cfg_path: Path) -> tuple[EasyDict, list[str]]:
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


def create_custom_radar_infos(dataset_cfg, class_names, data_path: Path, save_path: Path, workers: int = 4):
    dataset = CustomRadarDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False, logger=None
    )

    for split in ("train", "val", "test"):
        dataset.set_split(split)
        has_label = split != "test"
        infos = dataset.get_infos(num_workers=workers, has_label=has_label, count_inside_pts=has_label)
        info_path = save_path / f"custom_radar_infos_{split}.pkl"
        with open(info_path, "wb") as f:
            pickle.dump(infos, f)
        print(f"CustomRadar info {split}: {len(infos)} -> {info_path}")

    train_info_path = save_path / "custom_radar_infos_train.pkl"
    dataset.set_split("train")
    dataset.create_groundtruth_database(train_info_path, split="train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    rc.add_run_config_arg(parser)
    args = parser.parse_args()
    args.run_cfg = rc.load_run_config(args.run_config)
    return args


def main() -> None:
    args = parse_args()
    cfg: rc.RunConfig = args.run_cfg
    dataset_cfg, class_names = load_model_dataset_cfg(cfg.model_cfg)
    dataset_cfg.DATA_PATH = str(cfg.data_root)
    create_custom_radar_infos(dataset_cfg, class_names, cfg.data_root, cfg.data_root, workers=cfg.workers)


if __name__ == "__main__":
    main()
