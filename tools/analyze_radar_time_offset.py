#!/usr/bin/env python3
"""Sweep opponent time offset using radar points-in-GT-box as the score."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

import radar_common as rc
from build_pp_radar import make_points_for_current


@dataclass
class RadarFrame:
    t: float
    points: np.ndarray
    T_world_radar: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    rc.add_run_config_arg(parser)
    parser.add_argument("--offset-min", type=float, default=-0.25)
    parser.add_argument("--offset-max", type=float, default=0.05)
    parser.add_argument("--offset-step", type=float, default=0.025)
    parser.add_argument("--max-scored-frames", type=int, default=2000)
    parser.add_argument("--min-points", type=int, default=None)
    args = parser.parse_args()
    args.run_cfg = rc.load_run_config(args.run_config)
    return args


def offset_values(args: argparse.Namespace) -> np.ndarray:
    n = int(round((args.offset_max - args.offset_min) / args.offset_step)) + 1
    return args.offset_min + np.arange(n, dtype=np.float64) * args.offset_step


def summarize(counts: list[int], local_x: list[float]) -> tuple[int, float, float, float, int]:
    if not counts:
        return 0, 0.0, 0.0, 0.0, 0
    arr = np.asarray(counts, dtype=np.float64)
    lx = np.asarray(local_x, dtype=np.float64)
    return int(len(arr)), float(arr.mean()), float(np.median(arr)), float(lx.mean()) if len(lx) else 0.0, int((arr >= 3).sum())


def score_offsets(cfg: rc.RunConfig, args: argparse.Namespace) -> None:
    if len(cfg.runs) != 1:
        raise ValueError("This analysis script expects one configured run")
    run = cfg.runs[0]
    offsets = offset_values(args)
    min_points = cfg.min_points_in_gt if args.min_points is None else int(args.min_points)

    ego = rc.load_odom_timeline(run.ego_odom, run.ego_odom_topic)
    opponent = rc.load_odom_timeline(run.opp_odom, run.opp_odom_topic)
    T_ram_radar = rc.load_static_transform(run.radar, rc.RAM_FRAME, rc.RADAR_FRAME)
    if T_ram_radar is None:
        print(f"WARNING: no /tf_static {rc.RAM_FRAME}->{rc.RADAR_FRAME}; using identity radar extrinsic")
        T_ram_radar = np.eye(4, dtype=np.float64)
    T_cog_radar = rc.T_COG_RAM @ T_ram_radar

    scores = {float(offset): [] for offset in offsets}
    local_x = {float(offset): [] for offset in offsets}
    buffer: deque[RadarFrame] = deque(maxlen=max(1, cfg.sweep_count))
    seen = 0
    scored_frames = 0

    print("[config]")
    print(f"  current offset: {run.opponent_time_offset_seconds:+.3f}s")
    print(f"  sweep: {offsets[0]:+.3f}s .. {offsets[-1]:+.3f}s step {args.offset_step:.3f}s")
    print(f"  sweeps: {cfg.sweep_count}, horizon: {cfg.sweep_horizon_seconds}s, ego_compensated: {cfg.ego_compensated}")
    print(f"  min points threshold: {min_points}")
    print(f"  radar extrinsic COG<-radar translation: {T_cog_radar[:3, 3].round(4).tolist()}")

    for mcap_path in rc.mcap_paths(run.radar, "radar"):
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=[run.radar_topic]):
                seen += 1
                if (seen - 1) % max(1, cfg.stride) != 0:
                    continue
                radar_time = rc.stamp_seconds(msg)
                if radar_time < ego.t_start or radar_time > ego.t_end:
                    continue
                raw_points = rc.parse_ars548_detections(msg)
                T_world_radar = rc.ego_cog_to_radar_pose(ego.pose_at(radar_time), T_cog_radar)
                current = RadarFrame(t=radar_time, points=raw_points, T_world_radar=T_world_radar)
                buffer.append(current)
                points = make_points_for_current(current, buffer, cfg)
                if len(points) == 0:
                    continue

                any_valid = False
                for offset in offsets:
                    offset = float(offset)
                    box = rc.opponent_box_at_radar(radar_time, ego, opponent, offset, T_cog_radar, cfg.box_lwh)
                    if box is None:
                        continue
                    center, yaw, _ = box
                    mask = rc.box_mask(points, center, yaw, cfg.box_lwh)
                    count = int(mask.sum())
                    scores[offset].append(count)
                    if count > 0:
                        local = rc.local_points(points[mask], center, yaw)
                        local_x[offset].append(float(local[:, 0].mean()))
                    any_valid = True
                if any_valid:
                    scored_frames += 1
                if scored_frames >= args.max_scored_frames:
                    break
        if scored_frames >= args.max_scored_frames:
            break

    print("\n[offset sweep]")
    print("offset_s  frames  mean_pts  median_pts  ge_threshold  mean_local_x_m")
    best = None
    for offset in offsets:
        offset = float(offset)
        n, mean_count, median_count, mean_lx, ge = summarize(scores[offset], local_x[offset])
        print(f"{offset:+.3f}  {n:6d}  {mean_count:8.3f}  {median_count:10.3f}  {ge:12d}  {mean_lx:+14.3f}")
        key = (ge, mean_count)
        if best is None or key > best[0]:
            best = (key, offset)
    if best is not None:
        print(f"\nrecommended_offset_seconds: {best[1]:+.3f}")
        print("Use this as inputs.runs[0].opponent_time_offset_seconds, then rebuild data.")


def main() -> None:
    args = parse_args()
    score_offsets(args.run_cfg, args)


if __name__ == "__main__":
    main()
