#!/usr/bin/env python3
"""Build a native OpenPCDet radar dataset from configured Putnam Park MCAPs."""
from __future__ import annotations

import argparse
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

import radar_common as rc


@dataclass
class RadarFrame:
    t: float
    points: np.ndarray
    T_world_radar: np.ndarray


def clean_output(root: Path) -> None:
    for subdir in ("points", "labels", "ImageSets", "debug"):
        target = root / subdir
        if target.is_dir():
            shutil.rmtree(target)
    for pkl in root.glob("custom_radar_*infos*.pkl"):
        pkl.unlink()
    db = root / "gt_database"
    if db.is_dir():
        shutil.rmtree(db)


def ensure_dirs(root: Path) -> None:
    for subdir in ("points", "labels", "ImageSets"):
        (root / subdir).mkdir(parents=True, exist_ok=True)


def label_line(center: np.ndarray, yaw: float, velocity: np.ndarray, cfg: rc.RunConfig) -> str:
    l, w, h = cfg.box_lwh
    return (
        f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f} "
        f"{l:.6f} {w:.6f} {h:.6f} {yaw:.6f} "
        f"{velocity[0]:.6f} {velocity[1]:.6f} {cfg.class_name}"
    )


def write_frame(root: Path, frame_id: str, points: np.ndarray, label: str | None) -> None:
    (root / "points" / f"{frame_id}.bin").write_bytes(points.astype(np.float32).tobytes())
    if label is not None:
        (root / "labels" / f"{frame_id}.txt").write_text(label + "\n", encoding="utf-8")
    else:
        (root / "labels" / f"{frame_id}.txt").write_text("", encoding="utf-8")


def keep_empty_low_point_frame(
    split: str,
    cfg: rc.RunConfig,
    positive_counts: dict[str, int],
    negative_counts: dict[str, int],
) -> bool:
    if cfg.dataset_mode == "ssl":
        return True
    target_fraction = cfg.empty_train_fraction if split == "train" else cfg.empty_val_fraction if split == "val" else 1.0
    target_fraction = min(max(float(target_fraction), 0.0), 1.0)
    if split == "test":
        return True
    if target_fraction <= 0.0:
        return False
    if target_fraction >= 1.0:
        return True
    max_negatives = int(np.floor(positive_counts[split] * target_fraction / max(1e-6, 1.0 - target_fraction)))
    return negative_counts[split] < max_negatives


def load_or_build_raw_cache(cfg: rc.RunConfig, run: rc.DatasetRun, *, dry_run: bool = False) -> dict[str, Any]:
    path = rc.cache_path(cfg, f"{run.name}_raw_radar")
    source_paths = [*run.ego_odom, *run.opp_odom, *run.radar]
    fingerprint = rc.source_fingerprint(source_paths)
    if cfg.cache_enabled:
        cached = rc.load_npz_cache(path, expected_fingerprint=fingerprint, expected_version=2)
        if cached is not None:
            return cached

    start = time.time()
    rc.log_kvs("cache miss", [("path", path), ("what", "raw radar/odom MCAP data")])
    ego = rc.load_odom_timeline(run.ego_odom, run.ego_odom_topic)
    opponent = rc.load_odom_timeline(run.opp_odom, run.opp_odom_topic)
    T_ram_radar = rc.load_static_transform(run.radar, rc.RAM_FRAME, rc.RADAR_FRAME)
    if T_ram_radar is None:
        rc.log(f"  WARNING: no /tf_static {rc.RAM_FRAME}->{rc.RADAR_FRAME}; using identity radar extrinsic")
        T_ram_radar = np.eye(4, dtype=np.float64)
    T_cog_radar = rc.T_COG_RAM @ T_ram_radar

    radar_timestamps: list[float] = []
    radar_points: list[np.ndarray] = []
    seen = 0
    decode_error = 0
    for mcap_path in rc.mcap_paths(run.radar, "radar"):
        rc.log(f"  radar decode: {mcap_path}")
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=[run.radar_topic]):
                seen += 1
                try:
                    radar_timestamps.append(rc.stamp_seconds(msg))
                    radar_points.append(rc.parse_ars548_detections(msg))
                except Exception as exc:
                    decode_error += 1
                    if decode_error <= 5:
                        rc.log(f"  decode error at message {seen}: {exc}")

    cache = {
        "radar_timestamps": np.asarray(radar_timestamps, dtype=np.float64),
        "radar_points": rc.pack_object_array(radar_points),
        "ego_t": ego.t,
        "ego_xyz": ego.xyz,
        "ego_quat_xyzw": ego.quat_xyzw,
        "opponent_t": opponent.t,
        "opponent_xyz": opponent.xyz,
        "opponent_quat_xyzw": opponent.quat_xyzw,
        "T_cog_radar": T_cog_radar,
        "seen": np.asarray([seen], dtype=np.int64),
        "decode_error": np.asarray([decode_error], dtype=np.int64),
    }
    if cfg.cache_enabled and not dry_run:
        rc.write_npz_cache(path, cache, source_fingerprint=fingerprint, version=2)
        rc.log_kvs("cache wrote", [("path", path), ("elapsed_seconds", f"{time.time() - start:.1f}")])
    return cache


def make_points_for_current(current: RadarFrame, buffer: deque[RadarFrame], cfg: rc.RunConfig) -> np.ndarray:
    pieces = []
    horizon = max(cfg.sweep_horizon_seconds, 1e-6)
    selected = list(buffer)[-cfg.sweep_count :]
    for frame in selected:
        dt = frame.t - current.t
        if abs(dt) > cfg.sweep_horizon_seconds + 1e-6:
            continue
        pts = rc.transform_radar_points(frame.points, frame.T_world_radar, current.T_world_radar, cfg.ego_compensated)
        time_col = np.full((len(pts), 1), dt / horizon, dtype=np.float32)
        pieces.append(np.hstack((pts, time_col)))
    if not pieces:
        return np.zeros((0, 9), dtype=np.float32)
    return np.vstack(pieces).astype(np.float32)


def apply_ars548_characteristic_filter(points: np.ndarray, cfg: rc.RunConfig) -> np.ndarray:
    if not cfg.radar_filter_enabled or len(points) == 0:
        return points
    p = cfg.radar_filter
    r = np.linalg.norm(points[:, :3], axis=1)
    az = np.arctan2(points[:, 1], points[:, 0])
    el = np.arctan2(points[:, 2], np.maximum(np.linalg.norm(points[:, :2], axis=1), 1e-6))
    vr = points[:, 4]
    cos_az = np.maximum(np.abs(np.cos(az)), 1e-3) * np.sign(np.cos(az))
    cos_el = np.maximum(np.abs(np.cos(el)), 1e-3) * np.sign(np.cos(el))
    vx = vr / cos_az / cos_el
    vy = vr * np.cos(el) * np.sin(az)
    relative_z = r * np.sin(el)
    mask = np.ones(len(points), dtype=bool)
    mask &= (points[:, 5] >= p.get("snr_min_threshold", -np.inf)) & (points[:, 5] <= p.get("snr_max_threshold", np.inf))
    mask &= (vx >= p.get("vx_min_threshold", -np.inf)) & (vx <= p.get("vx_max_threshold", np.inf))
    mask &= (vy >= p.get("vy_min_threshold", -np.inf)) & (vy <= p.get("vy_max_threshold", np.inf))
    mask &= (vr >= p.get("radial_v_min_threshold", -np.inf)) & (vr <= p.get("radial_v_max_threshold", np.inf))
    mask &= (r >= p.get("range_min_threshold", -np.inf)) & (r <= p.get("range_max_threshold", np.inf))
    mask &= (points[:, 3] >= p.get("rcs_min_threshold", -np.inf)) & (points[:, 3] <= p.get("rcs_max_threshold", np.inf))
    mask &= (relative_z >= p.get("elevation_min_threshold", -np.inf)) & (relative_z <= p.get("elevation_max_threshold", np.inf))
    mask &= (points[:, 6] >= p.get("rss_min_threshold", -np.inf)) & (points[:, 6] <= p.get("rss_max_threshold", np.inf))
    return points[mask]


def build_one_run(
    cfg: rc.RunConfig,
    run: rc.DatasetRun,
    frame_start: int,
    dry_run: bool,
) -> tuple[list[str], list[str], list[str], dict[str, int | float]]:
    rc.log_section(f"run: {run.name}")
    raw = load_or_build_raw_cache(cfg, run, dry_run=dry_run)
    ego = rc.PoseTimeline.from_arrays(raw["ego_t"], raw["ego_xyz"], raw["ego_quat_xyzw"])
    opponent = rc.PoseTimeline.from_arrays(raw["opponent_t"], raw["opponent_xyz"], raw["opponent_quat_xyzw"])
    T_cog_radar = np.asarray(raw["T_cog_radar"], dtype=np.float64)

    rc.log_kv("ego_samples", f"{len(ego.t)} [{ego.t_start:.3f} .. {ego.t_end:.3f}]")
    rc.log_kv("opponent_samples", f"{len(opponent.t)} [{opponent.t_start:.3f} .. {opponent.t_end:.3f}]")
    rc.log_kv("radar_frames_cached", len(raw["radar_timestamps"]))
    rc.log_kv("decode_error", int(raw["decode_error"][0]))
    rc.log_kv("radar_extrinsic_cog_from_radar_xyz", T_cog_radar[:3, 3].round(4).tolist())
    rc.log_kv("opponent_time_offset_seconds", f"{run.opponent_time_offset_seconds:+.3f}")

    t_lo = max(ego.t_start, opponent.t_start - run.opponent_time_offset_seconds)
    t_hi = min(ego.t_end, opponent.t_end - run.opponent_time_offset_seconds)
    duration = t_hi - t_lo
    if duration <= 0:
        raise RuntimeError(f"{run.name}: no overlapping ego/opponent time range after time offset")

    train_ids, val_ids, test_ids = [], [], []
    buffer: deque[RadarFrame] = deque(maxlen=max(1, cfg.sweep_count))
    stats: dict[str, int | float] = {
        "seen": 0,
        "out": 0,
        "label": 0,
        "empty_low_point_train": 0,
        "empty_low_point_val": 0,
        "test_unlabeled": 0,
        "guard": 0,
        "outside_time": 0,
        "empty": 0,
        "low_points": 0,
        "no_pose_time": 0,
        "decode_error": 0,
        "min_points": 10**9,
        "max_points": 0,
        "sum_points": 0,
    }

    stats["seen"] = int(raw["seen"][0])
    stats["decode_error"] = int(raw["decode_error"][0])
    radar_timestamps = np.asarray(raw["radar_timestamps"], dtype=np.float64)
    radar_points = raw["radar_points"]
    positive_counts = {"train": 0, "val": 0}
    negative_counts = {"train": 0, "val": 0}
    build_start = time.time()
    for radar_index, (radar_time, cached_points) in enumerate(zip(radar_timestamps, radar_points)):
        if radar_index % max(1, cfg.stride) != 0:
            continue
        if cfg.max_frames > 0 and int(stats["out"]) >= cfg.max_frames:
            break

        raw_points = apply_ars548_characteristic_filter(np.asarray(cached_points, dtype=np.float32), cfg)
        if radar_time < t_lo or radar_time > t_hi:
            stats["outside_time"] += 1
            continue

        split = rc.split_assignment(radar_time - t_lo, duration, cfg)
        if not split:
            stats["guard"] += 1
            continue

        T_world_radar = rc.ego_cog_to_radar_pose(ego.pose_at(float(radar_time)), T_cog_radar)
        current = RadarFrame(t=float(radar_time), points=raw_points, T_world_radar=T_world_radar)
        buffer.append(current)
        points = make_points_for_current(current, buffer, cfg)
        if len(points) == 0:
            stats["empty"] += 1
            continue

        label = None
        box = rc.opponent_box_at_radar(
            float(radar_time), ego, opponent, run.opponent_time_offset_seconds, T_cog_radar, cfg.box_lwh
        )
        if box is not None and split != "test":
            center, yaw, velocity = box
            points_in_gt = rc.count_points_in_box(points, center, yaw, cfg.box_lwh)
            if points_in_gt >= cfg.min_points_in_gt:
                label = label_line(center, yaw, velocity, cfg)
                stats["label"] += 1
            else:
                stats["low_points"] += 1
        elif split == "test":
            stats["test_unlabeled"] += 1
        else:
            stats["no_pose_time"] += 1
        if split != "test" and label is None:
            if not keep_empty_low_point_frame(split, cfg, positive_counts, negative_counts):
                continue
            negative_counts[split] += 1
            stats[f"empty_low_point_{split}"] += 1
        elif split != "test" and label is not None:
            positive_counts[split] += 1

        frame_id = f"{frame_start + int(stats['out']):06d}"
        if not dry_run:
            write_frame(cfg.data_root, frame_id, points, label)
        (train_ids if split == "train" else val_ids if split == "val" else test_ids).append(frame_id)
        stats["out"] += 1
        stats["min_points"] = min(int(stats["min_points"]), len(points))
        stats["max_points"] = max(int(stats["max_points"]), len(points))
        stats["sum_points"] += len(points)
        if int(stats["out"]) % 1000 == 0:
            rc.log(
                "  progress "
                f"out={stats['out']} label={stats['label']} low_points={stats['low_points']} "
                f"empty_low_point_train={stats['empty_low_point_train']} "
                f"empty_low_point_val={stats['empty_low_point_val']} "
                f"elapsed={time.time() - build_start:.1f}s"
            )
    if int(stats["out"]) == 0:
        stats["min_points"] = 0
    return train_ids, val_ids, test_ids, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    rc.add_run_config_arg(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_cfg = rc.load_run_config(args.run_config)
    return args


def main() -> None:
    args = parse_args()
    cfg: rc.RunConfig = args.run_cfg
    start = time.time()

    rc.log_section("config")
    rc.log_kv("data_root", cfg.data_root)
    rc.log_kv("sweep_count", cfg.sweep_count)
    rc.log_kv("sweep_horizon_seconds", cfg.sweep_horizon_seconds)
    rc.log_kv("ego_compensated", cfg.ego_compensated)
    rc.log_kv("dataset_mode", cfg.dataset_mode)
    rc.log_kv("empty_train_fraction", cfg.empty_train_fraction)
    rc.log_kv("empty_val_fraction", cfg.empty_val_fraction)
    rc.log_kv("point_features", "x y z rcs radial_velocity snr rss existence_prob time_delta")
    rc.log_kv("box_lwh", cfg.box_lwh.tolist())
    rc.log_kv("class_name", cfg.class_name)
    rc.log_kv("min_points_in_gt", cfg.min_points_in_gt)
    rc.log_kv("radar_filter_enabled", cfg.radar_filter_enabled)
    rc.log_kv("cache_enabled", cfg.cache_enabled)
    rc.log_kv("cache_dir", cfg.cache_dir)

    if not args.dry_run:
        cfg.data_root.mkdir(parents=True, exist_ok=True)
        if cfg.clean_dataset:
            clean_output(cfg.data_root)
        ensure_dirs(cfg.data_root)

    train_ids, val_ids, test_ids = [], [], []
    totals = {
        "seen": 0,
        "out": 0,
        "label": 0,
        "empty_low_point_train": 0,
        "empty_low_point_val": 0,
        "test_unlabeled": 0,
        "guard": 0,
        "outside_time": 0,
        "empty": 0,
        "low_points": 0,
        "no_pose_time": 0,
        "decode_error": 0,
        "min_points": 10**9,
        "max_points": 0,
        "sum_points": 0,
    }
    for run in cfg.runs:
        run_train, run_val, run_test, stats = build_one_run(cfg, run, int(totals["out"]), args.dry_run)
        train_ids.extend(run_train)
        val_ids.extend(run_val)
        test_ids.extend(run_test)
        for key, value in stats.items():
            if key == "min_points":
                totals[key] = min(int(totals[key]), int(value))
            else:
                totals[key] += int(value)

    if totals["out"] == 0:
        totals["min_points"] = 0
    avg_points = float(totals["sum_points"]) / max(1, int(totals["out"]))
    summary = {
        **totals,
        "avg_points": avg_points,
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
        "test_frames": len(test_ids),
        "feature_names": ["x", "y", "z", "rcs", "radial_velocity", "snr", "rss", "existence_prob", "time_delta"],
    }

    if not args.dry_run:
        rc.write_imageset(cfg.data_root / "ImageSets" / "train.txt", train_ids)
        rc.write_imageset(cfg.data_root / "ImageSets" / "val.txt", val_ids)
        rc.write_imageset(cfg.data_root / "ImageSets" / "test.txt", test_ids)
        (cfg.data_root / "build_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=True), encoding="utf-8")

    rc.log_section("done")
    rc.log_kv("frames", f"{totals['out']} ({len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test)")
    rc.log_kv("labels", totals["label"])
    rc.log_kv("empty_low_point_train", totals["empty_low_point_train"])
    rc.log_kv("empty_low_point_val", totals["empty_low_point_val"])
    rc.log_kv("test_unlabeled", totals["test_unlabeled"])
    rc.log_kv("radar_points_per_frame_min_avg_max", f"{totals['min_points']}/{avg_points:.1f}/{totals['max_points']}")
    rc.log_kv("skipped_low_points", totals["low_points"])
    rc.log_kv("skipped_no_pose", totals["no_pose_time"])
    rc.log_kv("skipped_guard", totals["guard"])
    rc.log_kv("skipped_outside_time", totals["outside_time"])
    rc.log_kv("skipped_empty", totals["empty"])
    rc.log_kv("decode_error", totals["decode_error"])
    rc.log_kv("elapsed_seconds", f"{time.time() - start:.1f}")


if __name__ == "__main__":
    main()
