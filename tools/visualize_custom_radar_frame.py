#!/usr/bin/env python3
"""Write a small MCAP for Foxglove: generated radar dataset cloud + GT box."""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import radar_common as rc


_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RADAR_FRAME = "radar_front"
FLOAT32 = 7
DEFAULT_FEATURE_NAMES = ["x", "y", "z", "rcs", "radial_velocity", "snr", "rss", "existence_prob", "time_delta"]
BOX_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


POINTCLOUD_SCHEMA = {
    "title": "foxglove.PointCloud",
    "description": "A collection of N-dimensional points.",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
            "required": ["sec", "nsec"],
        },
        "frame_id": {"type": "string"},
        "pose": {
            "title": "foxglove.Pose",
            "type": "object",
            "properties": {
                "position": {
                    "title": "foxglove.Vector3",
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                    },
                    "required": ["x", "y", "z"],
                },
                "orientation": {
                    "title": "foxglove.Quaternion",
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "w": {"type": "number"},
                    },
                    "required": ["x", "y", "z", "w"],
                },
            },
            "required": ["position", "orientation"],
        },
        "point_stride": {"type": "integer", "minimum": 0},
        "fields": {
            "type": "array",
            "items": {
                "title": "foxglove.PackedElementField",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "type": {
                        "title": "foxglove.NumericType",
                        "oneOf": [
                            {"title": "UNKNOWN", "const": 0},
                            {"title": "UINT8", "const": 1},
                            {"title": "INT8", "const": 2},
                            {"title": "UINT16", "const": 3},
                            {"title": "INT16", "const": 4},
                            {"title": "UINT32", "const": 5},
                            {"title": "INT32", "const": 6},
                            {"title": "FLOAT32", "const": 7},
                            {"title": "FLOAT64", "const": 8},
                        ],
                    },
                },
                "required": ["name", "offset", "type"],
            },
        },
        "data": {"type": "string", "contentEncoding": "base64"},
    },
    "required": ["timestamp", "frame_id", "pose", "point_stride", "fields", "data"],
}


def timestamp_from_seconds(t: float) -> dict[str, int]:
    sec = int(math.floor(t))
    nsec = int(round((t - sec) * 1.0e9))
    if nsec >= 1_000_000_000:
        sec += 1
        nsec -= 1_000_000_000
    return {"sec": sec, "nsec": nsec}


def identity_pose() -> dict[str, dict[str, float]]:
    return {
        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def load_feature_names(data_root: Path) -> list[str]:
    summary_path = data_root / "build_summary.yaml"
    if not summary_path.exists():
        return DEFAULT_FEATURE_NAMES
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = yaml.safe_load(f) or {}
    feature_names = summary.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"{summary_path} must contain a non-empty feature_names list")
    return [str(name) for name in feature_names]


def load_points(path: Path, num_features: int) -> np.ndarray:
    points = np.fromfile(str(path), dtype=np.float32)
    if points.size % num_features != 0:
        raise ValueError(f"{path} has {points.size} floats, not divisible by {num_features} features")
    return points.reshape(-1, num_features)


def read_label(path: Path) -> list[np.ndarray]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            boxes.append(np.asarray([float(x) for x in line.split()[:9]], dtype=np.float32))
    return boxes


def frame_ids_for_split(data_root: Path, split: str) -> list[str]:
    path = data_root / "ImageSets" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_frame_ids(ids: list[str], start_index: int, count: int, stride: int) -> list[str]:
    selected = ids[start_index::max(1, stride)]
    return selected if count <= 0 else selected[:count]


def pointcloud_message(points: np.ndarray, feature_names: list[str], t: float) -> dict[str, Any]:
    points = np.ascontiguousarray(points.astype("<f4", copy=False))
    return {
        "timestamp": timestamp_from_seconds(t),
        "frame_id": RADAR_FRAME,
        "pose": identity_pose(),
        "point_stride": 4 * len(feature_names),
        "fields": [
            {"name": name, "offset": 4 * i, "type": FLOAT32}
            for i, name in enumerate(feature_names)
        ],
        "data": base64.b64encode(points.tobytes()).decode("ascii"),
    }


def box_corners(box: np.ndarray) -> np.ndarray:
    x, y, z, dx, dy, dz, yaw = [float(v) for v in box[:7]]
    local = np.array(
        [
            [dx / 2, dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [dx / 2, dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
        ],
        dtype=np.float32,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return local @ rot.T + np.array([x, y, z], dtype=np.float32)


def boxes_to_wire_points(boxes: list[np.ndarray]) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    points = []
    for box in boxes:
        corners = box_corners(box)
        for a, b in BOX_EDGES:
            points.append([corners[a, 0], corners[a, 1], corners[a, 2], 1.0])
            points.append([corners[b, 0], corners[b, 1], corners[b, 2], 1.0])
    return np.asarray(points, dtype=np.float32).reshape(-1, 4)


class _OneFrameRadarDataset:
    """Small wrapper around the normal CustomRadarDataset inference path."""

    def __init__(self, dataset_cfg: Any, class_names: list[str], root_path: Path, logger: Any):
        from pcdet.datasets.custom_radar.custom_radar_dataset import CustomRadarDataset

        self.dataset = CustomRadarDataset(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=logger,
        )

    def prepare(self, points: np.ndarray, frame_id: str) -> dict[str, Any]:
        data_dict = self.dataset.prepare_data(
            data_dict={
                "points": self.dataset.normalize_features(points).astype(np.float32, copy=False),
                "frame_id": frame_id,
            }
        )
        return self.dataset.collate_batch([data_dict])


def build_predictor(cfg_file: Path, ckpt: Path, data_root: Path) -> tuple[Any, _OneFrameRadarDataset]:
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.models import build_network
    from pcdet.utils import common_utils

    cfg_file = Path(cfg_file).resolve()
    ckpt = Path(ckpt).resolve()

    old_cwd = Path.cwd()
    try:
        os.chdir(_TOOLS)
        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(old_cwd)

    cfg.DATA_CONFIG.DATA_PATH = str(data_root)
    logger = common_utils.create_logger()
    dataset = _OneFrameRadarDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, data_root, logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset.dataset)
    model.load_params_from_file(filename=str(ckpt), logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    return model, dataset


def predict_boxes(
    predictor: tuple[Any, _OneFrameRadarDataset],
    points: np.ndarray,
    frame_id: str,
    score_thresh: float,
) -> list[np.ndarray]:
    import torch
    from pcdet.models import load_data_to_gpu

    model, dataset = predictor
    if points.size == 0:
        return []
    with torch.no_grad():
        data_dict = dataset.prepare(points, frame_id)
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
    pred_boxes = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
    pred_scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()
    return [box.astype(np.float32, copy=False) for box, score in zip(pred_boxes, pred_scores) if score >= score_thresh]


def write_mcap(
    cfg: rc.RunConfig,
    frame_ids: list[str],
    out_path: Path,
    dt: float,
    predictor: tuple[Any, _OneFrameRadarDataset] | None = None,
    score_thresh: float = 0.1,
) -> int:
    from mcap.writer import CompressionType, Writer

    feature_names = load_feature_names(cfg.data_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        writer = Writer(f, compression=CompressionType.ZSTD)
        writer.start(profile="foxglove", library="radar-jepa-viz")
        schema_id = writer.register_schema(
            name="foxglove.PointCloud",
            encoding="jsonschema",
            data=json.dumps(POINTCLOUD_SCHEMA).encode("utf-8"),
        )
        ch_points = writer.register_channel("/radar/points", "json", schema_id)
        ch_gt = writer.register_channel("/radar/gt_opp", "json", schema_id)
        ch_pred = writer.register_channel("/radar/pred_opp", "json", schema_id)

        written = 0
        for i, frame_id in enumerate(frame_ids):
            point_path = cfg.data_root / "points" / f"{frame_id}.bin"
            if not point_path.exists():
                print(f"skip missing {point_path}")
                continue
            t = i * dt
            log_time = int(round(t * 1.0e9))
            points = load_points(point_path, len(feature_names))
            writer.add_message(
                channel_id=ch_points,
                log_time=log_time,
                publish_time=log_time,
                data=json.dumps(pointcloud_message(points, feature_names, t), separators=(",", ":")).encode("utf-8"),
            )
            wire = boxes_to_wire_points(read_label(cfg.data_root / "labels" / f"{frame_id}.txt"))
            writer.add_message(
                channel_id=ch_gt,
                log_time=log_time,
                publish_time=log_time,
                data=json.dumps(pointcloud_message(wire, ["x", "y", "z", "intensity"], t), separators=(",", ":")).encode("utf-8"),
            )
            pred_wire = np.zeros((0, 4), dtype=np.float32)
            if predictor is not None:
                pred_wire = boxes_to_wire_points(predict_boxes(predictor, points, frame_id, score_thresh))
            writer.add_message(
                channel_id=ch_pred,
                log_time=log_time,
                publish_time=log_time,
                data=json.dumps(pointcloud_message(pred_wire, ["x", "y", "z", "intensity"], t), separators=(",", ":")).encode("utf-8"),
            )
            written += 1
        writer.finish()
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generated radar dataset clip for Foxglove: /radar/points + /radar/gt_opp"
    )
    rc.add_run_config_arg(parser)
    parser.add_argument("--frame-id", default=None, help="Write one frame id. If omitted, use ImageSets split.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=12, help="Number of selected frames. Use <=0 for all selected frames.")
    parser.add_argument("--stride", type=int, default=250)
    parser.add_argument("--out", type=Path, default=None, help="Output MCAP path.")
    parser.add_argument("--dt", type=float, default=0.1, help="Synthetic playback interval in seconds.")
    parser.add_argument("--cfg-file", type=Path, default=None, help="OpenPCDet model config for prediction overlay.")
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint for /radar/pred_opp overlay.")
    parser.add_argument("--score-thresh", type=float, default=0.1, help="Prediction score threshold for /radar/pred_opp.")
    args = parser.parse_args()
    args.run_cfg = rc.load_run_config(args.run_config)
    return args


def main() -> None:
    args = parse_args()
    cfg = args.run_cfg
    if args.frame_id is not None:
        frame_ids = [args.frame_id]
    else:
        frame_ids = select_frame_ids(
            frame_ids_for_split(cfg.data_root, args.split),
            args.start_index,
            args.count,
            args.stride,
        )
    out_path = args.out or (cfg.log_dir / cfg.run_name / "visualize.mcap")
    ckpt = args.ckpt if args.ckpt is not None else cfg.checkpoint
    predictor = None
    if ckpt is not None:
        predictor = build_predictor(args.cfg_file or cfg.model_cfg, ckpt, cfg.data_root)
    written = write_mcap(
        cfg,
        frame_ids,
        out_path,
        max(float(args.dt), 1e-3),
        predictor=predictor,
        score_thresh=float(args.score_thresh),
    )
    print(f"Wrote {written} frame(s) to {out_path}")


if __name__ == "__main__":
    main()
