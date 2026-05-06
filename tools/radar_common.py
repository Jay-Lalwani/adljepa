#!/usr/bin/env python3
"""Shared Putnam Park radar dataset helpers.

The generated detector frame is the current front-radar frame.  Ego odometry in
the UVA bags is map -> center_of_gravity, matching the existing PointPillars
pipeline conventions.
"""
from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

TOOLS_DIR = Path(__file__).resolve().parent
REPO_DIR = TOOLS_DIR.parent

COG_IN_RAM = np.array([1.3206, -0.030188, 0.23598], dtype=np.float64)
RAM_Z_ABOVE_GROUND = 0.198
RADAR_FRAME = "radar_front"
RAM_FRAME = "rear_axle_middle_ground"
COG_FRAME = "center_of_gravity"
DEFAULT_CLASS_NAME = "Car"
DEFAULT_BOX_LWH = np.array([4.876, 1.930, 1.55], dtype=np.float64)

T_COG_RAM = np.eye(4, dtype=np.float64)
T_COG_RAM[:3, 3] = -COG_IN_RAM


@dataclass(frozen=True)
class DatasetRun:
    name: str
    ego_odom: list[Path]
    radar: list[Path]
    opp_odom: list[Path]
    ego_odom_topic: str
    radar_topic: str
    opp_odom_topic: str
    opponent_time_offset_seconds: float


@dataclass(frozen=True)
class RunConfig:
    path: Path
    raw: dict[str, Any]
    run_name: str
    repo_dir: Path
    log_dir: Path
    data_root: Path
    container_image: Path
    container_binds: list[Path]
    runs: list[DatasetRun]
    model_cfg: Path
    checkpoint: Path | None
    pretrained_model: Path | None
    ssl_cfg: Path
    ssl_checkpoint: Path | None
    clean_dataset: bool
    max_frames: int
    stride: int
    chunk_seconds: float
    guard_seconds: float
    train_fraction: float
    test_fraction: float
    sweep_count: int
    sweep_horizon_seconds: float
    ego_compensated: bool
    dataset_mode: str
    empty_train_fraction: float
    empty_val_fraction: float
    min_points_in_gt: int
    radar_filter_enabled: bool
    radar_filter: dict[str, float]
    cache_enabled: bool
    cache_dir: Path
    box_lwh: np.ndarray
    class_name: str
    batch_size: int
    epochs: int
    workers: int
    eval_batch_size: int
    eval_score_thresh: float
    slurm_partition: str
    slurm_setup_gres: str
    slurm_pipeline_gres: str
    slurm_setup_cpus: int
    slurm_pipeline_cpus: int
    slurm_setup_mem: str
    slurm_pipeline_mem: str
    slurm_setup_time: str
    slurm_pipeline_time: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing required config section: {name}")
    return value


def _require(section: dict[str, Any], key: str, dotted: str) -> Any:
    value = section.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {dotted}")
    return value


def _path(value: Any, base: Path | None = None) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value)))
    p = Path(raw)
    return p if p.is_absolute() or base is None else base / p


def _path_list(value: Any, dotted: str) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{dotted} must be a non-empty YAML list")
    return [_path(v) for v in value]


def _dataset_runs(inputs: dict[str, Any]) -> list[DatasetRun]:
    raw_runs = _require(inputs, "runs", "inputs.runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("inputs.runs must be a non-empty YAML list")
    runs = []
    default_ego_topic = str(_require(inputs, "ego_odom_topic", "inputs.ego_odom_topic"))
    default_radar_topic = str(_require(inputs, "radar_topic", "inputs.radar_topic"))
    for i, item in enumerate(raw_runs):
        if not isinstance(item, dict):
            raise ValueError(f"inputs.runs[{i}] must be a YAML mapping")
        prefix = f"inputs.runs[{i}]"
        runs.append(
            DatasetRun(
                name=str(_require(item, "name", f"{prefix}.name")),
                ego_odom=_path_list(_require(item, "ego_odom", f"{prefix}.ego_odom"), f"{prefix}.ego_odom"),
                radar=_path_list(_require(item, "radar", f"{prefix}.radar"), f"{prefix}.radar"),
                opp_odom=_path_list(_require(item, "opp_odom", f"{prefix}.opp_odom"), f"{prefix}.opp_odom"),
                ego_odom_topic=str(item.get("ego_odom_topic", default_ego_topic)),
                radar_topic=str(item.get("radar_topic", default_radar_topic)),
                opp_odom_topic=str(_require(item, "opp_odom_topic", f"{prefix}.opp_odom_topic")),
                opponent_time_offset_seconds=float(
                    _require(item, "opponent_time_offset_seconds", f"{prefix}.opponent_time_offset_seconds")
                ),
            )
        )
    return runs


def load_run_config(path: Path | str) -> RunConfig:
    cfg_path = Path(path).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = (Path.cwd() / cfg_path).resolve()
    data = _read_yaml(cfg_path)
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path} must contain a YAML mapping")

    paths = _section(data, "paths")
    container = _section(data, "container")
    inputs = _section(data, "inputs")
    dataset = _section(data, "dataset")
    radar_filter = data.get("radar_filter", {})
    if not isinstance(radar_filter, dict):
        raise ValueError("radar_filter must be a YAML mapping when provided")
    cache_cfg = data.get("cache", {})
    if not isinstance(cache_cfg, dict):
        raise ValueError("cache must be a YAML mapping when provided")
    model = _section(data, "model")
    train = _section(data, "train")
    evaluate = _section(data, "evaluate")
    slurm = _section(data, "slurm")

    repo_dir = _path(_require(paths, "repo_dir", "paths.repo_dir"))
    box_lwh = np.asarray(dataset.get("box_lwh", DEFAULT_BOX_LWH), dtype=np.float64)
    if box_lwh.shape != (3,):
        raise ValueError("dataset.box_lwh must contain [length, width, height]")
    dataset_mode = str(dataset.get("mode", "supervised"))
    if dataset_mode not in {"supervised", "ssl"}:
        raise ValueError("dataset.mode must be one of: supervised, ssl")

    return RunConfig(
        path=cfg_path,
        raw=data,
        run_name=str(_require(data, "run_name", "run_name")),
        repo_dir=repo_dir,
        log_dir=_path(_require(paths, "log_dir", "paths.log_dir")),
        data_root=_path(_require(paths, "data_root", "paths.data_root"), repo_dir),
        container_image=_path(_require(container, "image", "container.image"), repo_dir),
        container_binds=_path_list(_require(container, "binds", "container.binds"), "container.binds"),
        runs=_dataset_runs(inputs),
        model_cfg=_path(_require(model, "cfg_file", "model.cfg_file"), repo_dir),
        checkpoint=_path(model["checkpoint"]) if model.get("checkpoint") else None,
        pretrained_model=_path(model["pretrained_model"]) if model.get("pretrained_model") else None,
        ssl_cfg=_path(_require(model, "ssl_cfg_file", "model.ssl_cfg_file"), repo_dir),
        ssl_checkpoint=_path(model["ssl_checkpoint"]) if model.get("ssl_checkpoint") else None,
        clean_dataset=bool(_require(dataset, "clean", "dataset.clean")),
        max_frames=int(_require(dataset, "max_frames", "dataset.max_frames")),
        stride=int(_require(dataset, "stride", "dataset.stride")),
        chunk_seconds=float(_require(dataset, "chunk_seconds", "dataset.chunk_seconds")),
        guard_seconds=float(_require(dataset, "guard_seconds", "dataset.guard_seconds")),
        train_fraction=float(_require(dataset, "train_fraction", "dataset.train_fraction")),
        test_fraction=float(_require(dataset, "test_fraction", "dataset.test_fraction")),
        sweep_count=int(_require(dataset, "sweep_count", "dataset.sweep_count")),
        sweep_horizon_seconds=float(_require(dataset, "sweep_horizon_seconds", "dataset.sweep_horizon_seconds")),
        ego_compensated=bool(_require(dataset, "ego_compensated", "dataset.ego_compensated")),
        dataset_mode=dataset_mode,
        empty_train_fraction=float(dataset.get("empty_train_fraction", 0.0)),
        empty_val_fraction=float(dataset.get("empty_val_fraction", 0.0)),
        min_points_in_gt=int(dataset.get("min_points_in_gt", 3)),
        radar_filter_enabled=bool(radar_filter.get("enabled", False)),
        radar_filter={str(k): float(v) for k, v in radar_filter.items() if k != "enabled"},
        cache_enabled=bool(cache_cfg.get("enabled", True)),
        cache_dir=_path(cache_cfg.get("dir", "cache"), _path(_require(paths, "data_root", "paths.data_root"), repo_dir)),
        box_lwh=box_lwh,
        class_name=str(dataset.get("class_name", DEFAULT_CLASS_NAME)),
        batch_size=int(_require(train, "batch_size", "train.batch_size")),
        epochs=int(_require(train, "epochs", "train.epochs")),
        workers=int(_require(train, "workers", "train.workers")),
        eval_batch_size=int(_require(evaluate, "batch_size", "evaluate.batch_size")),
        eval_score_thresh=float(_require(evaluate, "score_thresh", "evaluate.score_thresh")),
        slurm_partition=str(_require(slurm, "partition", "slurm.partition")),
        slurm_setup_gres=str(_require(slurm, "setup_gres", "slurm.setup_gres")),
        slurm_pipeline_gres=str(_require(slurm, "pipeline_gres", "slurm.pipeline_gres")),
        slurm_setup_cpus=int(_require(slurm, "setup_cpus", "slurm.setup_cpus")),
        slurm_pipeline_cpus=int(_require(slurm, "pipeline_cpus", "slurm.pipeline_cpus")),
        slurm_setup_mem=str(_require(slurm, "setup_mem", "slurm.setup_mem")),
        slurm_pipeline_mem=str(_require(slurm, "pipeline_mem", "slurm.pipeline_mem")),
        slurm_setup_time=str(_require(slurm, "setup_time", "slurm.setup_time")),
        slurm_pipeline_time=str(_require(slurm, "pipeline_time", "slurm.pipeline_time")),
    )


def add_run_config_arg(parser: Any) -> None:
    parser.add_argument("--run-config", type=Path, required=True, help="Radar pipeline YAML config")


def log(message: str = "") -> None:
    print(message, flush=True)


def log_section(title: str) -> None:
    log(f"\n[{title}]")


def log_kv(key: str, value: Any, *, indent: int = 2) -> None:
    log(f"{' ' * indent}{key}: {value}")


def log_kvs(title: str, items: list[tuple[str, Any]]) -> None:
    log_section(title)
    for key, value in items:
        log_kv(key, value)


def format_cmd(cmd: list[str]) -> str:
    return shlex.join(str(part) for part in cmd)


def pack_object_array(items: list[Any]) -> np.ndarray:
    out = np.empty(len(items), dtype=object)
    out[:] = items
    return out


def cache_path(cfg: RunConfig, name: str) -> Path:
    return cfg.cache_dir / f"{name}.npz"


def source_fingerprint(paths: list[Path]) -> np.ndarray:
    rows = []
    for path in paths:
        for mcap_path in mcap_paths([path], "cache source"):
            stat = mcap_path.stat()
            rows.append((str(mcap_path), str(stat.st_size), str(stat.st_mtime_ns)))
    return np.asarray(sorted(rows), dtype=object)


def _same_fingerprint(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and all(tuple(a) == tuple(b) for a, b in zip(left.tolist(), right.tolist()))


def load_npz_cache(path: Path, *, expected_fingerprint: np.ndarray, expected_version: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as data:
            version = int(np.asarray(data["cache_version"]).reshape(-1)[0])
            fingerprint = np.asarray(data["source_fingerprint"], dtype=object)
            if version != expected_version or not _same_fingerprint(fingerprint, expected_fingerprint):
                log_kvs("cache stale", [("path", path), ("version", version), ("expected_version", expected_version)])
                return None
            log_kvs("cache hit", [("path", path)])
            return {key: data[key] for key in data.files if key not in {"cache_version", "source_fingerprint"}}
    except Exception as exc:
        log_kvs("cache ignored", [("path", path), ("reason", exc)])
        return None


def write_npz_cache(path: Path, payload: dict[str, Any], *, source_fingerprint: np.ndarray, version: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            cache_version=np.asarray([version], dtype=np.int64),
            source_fingerprint=source_fingerprint,
            **payload,
        )
    tmp_path.replace(path)


@dataclass
class PoseTimeline:
    t: np.ndarray
    xyz: np.ndarray
    quat_xyzw: np.ndarray
    slerp: Any

    @classmethod
    def from_arrays(cls, t, xyz, quat_xyzw):
        from scipy.spatial.transform import Rotation, Slerp

        order = np.argsort(t)
        t, xyz, quat_xyzw = t[order], xyz[order], quat_xyzw[order]
        keep = np.concatenate(([True], np.diff(t) > 1e-9))
        t, xyz, quat_xyzw = t[keep], xyz[keep], quat_xyzw[keep]
        return cls(t=t, xyz=xyz, quat_xyzw=quat_xyzw, slerp=Slerp(t, Rotation.from_quat(quat_xyzw)))

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    def pose_at(self, t_query: float | np.ndarray) -> np.ndarray:
        single = bool(np.isscalar(t_query))
        tq = np.atleast_1d(np.asarray(t_query, np.float64))
        tq = np.clip(tq, self.t[0], self.t[-1])
        T = np.zeros((len(tq), 4, 4), dtype=np.float64)
        T[:, :3, :3] = self.slerp(tq).as_matrix()
        T[:, 0, 3] = np.interp(tq, self.t, self.xyz[:, 0])
        T[:, 1, 3] = np.interp(tq, self.t, self.xyz[:, 1])
        T[:, 2, 3] = np.interp(tq, self.t, self.xyz[:, 2])
        T[:, 3, 3] = 1.0
        return T[0] if single else T

    def velocity_world_at(self, t_query: float, dt: float = 0.05) -> np.ndarray:
        t0 = max(self.t_start, t_query - dt)
        t1 = min(self.t_end, t_query + dt)
        if t1 <= t0:
            return np.zeros(3, dtype=np.float64)
        p0 = self.pose_at(t0)[:3, 3]
        p1 = self.pose_at(t1)[:3, 3]
        return (p1 - p0) / (t1 - t0)


def stamp_seconds(ros_msg) -> float:
    stamp = getattr(getattr(ros_msg, "header", None), "stamp", None)
    if stamp is None:
        raise ValueError("message has no header.stamp")
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def mcap_paths(paths: list[Path], label: str) -> list[Path]:
    out = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out.extend(sorted(path.glob("*.mcap")))
        else:
            raise FileNotFoundError(f"No {label} MCAP at {path}")
    if not out:
        raise FileNotFoundError(f"No {label} MCAP paths configured")
    return out


def _read_odometry(mcap_path: Path, topic: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    ts, xyz, quat = [], [], []
    with open(mcap_path, "rb") as f:
        for _, _, _, odom in make_reader(f, decoder_factories=[DecoderFactory()]).iter_decoded_messages(
            topics=[topic]
        ):
            t = stamp_seconds(odom)
            p, q = odom.pose.pose.position, odom.pose.pose.orientation
            ts.append(t)
            xyz.append([p.x, p.y, p.z])
            quat.append([q.x, q.y, q.z, q.w])
    if not ts:
        raise ValueError(f"No {topic} in {mcap_path}")
    return np.asarray(ts, np.float64), np.asarray(xyz, np.float64), np.asarray(quat, np.float64)


def load_odom_timeline(paths: list[Path], topic: str) -> PoseTimeline:
    t_all, x_all, q_all = [], [], []
    for path in mcap_paths(paths, "odometry"):
        t, x, q = _read_odometry(path, topic)
        t_all.append(t)
        x_all.append(x)
        q_all.append(q)
    return PoseTimeline.from_arrays(np.concatenate(t_all), np.vstack(x_all), np.vstack(q_all))


def transform_to_matrix(transform) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    t = transform.translation
    q = transform.rotation
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def load_static_transform(paths: list[Path], parent: str = RAM_FRAME, child: str = RADAR_FRAME) -> Optional[np.ndarray]:
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    parent = parent.lstrip("/")
    child = child.lstrip("/")
    for path in mcap_paths(paths, "radar/tf"):
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=["/tf_static"]):
                for tf in msg.transforms:
                    tf_parent = str(tf.header.frame_id).lstrip("/")
                    tf_child = str(tf.child_frame_id).lstrip("/")
                    if tf_parent == parent and tf_child == child:
                        return transform_to_matrix(tf.transform)
                    if tf_parent == child and tf_child == parent:
                        return tf_inv(transform_to_matrix(tf.transform))
    return None


def tf_inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def yaw_from_R(R: np.ndarray) -> float:
    return float(math.atan2(R[1, 0], R[0, 0]))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def split_assignment(t_rel: float, total: float, cfg: RunConfig) -> str:
    if t_rel >= total * (1.0 - cfg.test_fraction):
        return "test"
    inside_chunk = t_rel - int(t_rel // cfg.chunk_seconds) * cfg.chunk_seconds
    train_end = cfg.train_fraction * cfg.chunk_seconds
    val_start = train_end + cfg.guard_seconds
    if inside_chunk < train_end - cfg.guard_seconds * 0.5:
        return "train"
    if inside_chunk >= val_start:
        return "val"
    return ""


def ego_cog_to_radar_pose(T_world_cog: np.ndarray, T_cog_radar: np.ndarray) -> np.ndarray:
    return T_world_cog @ T_cog_radar


def opponent_box_at_radar(
    radar_time: float,
    ego: PoseTimeline,
    opponent: PoseTimeline,
    opponent_time_offset: float,
    T_cog_radar: np.ndarray,
    box_lwh: np.ndarray,
) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
    opp_time = radar_time + opponent_time_offset
    if radar_time < ego.t_start or radar_time > ego.t_end:
        return None
    if opp_time < opponent.t_start or opp_time > opponent.t_end:
        return None

    T_world_radar = ego_cog_to_radar_pose(ego.pose_at(radar_time), T_cog_radar)
    T_radar_world = tf_inv(T_world_radar)
    T_world_opp_cog = opponent.pose_at(opp_time)

    center_in_opp_cog = np.r_[np.array([box_lwh[0] * 0.5, 0.0, 0.0]) - COG_IN_RAM, 1.0]
    center = (T_radar_world @ (T_world_opp_cog @ center_in_opp_cog))[:3]

    opp_ground_in_ram = np.array([box_lwh[0] * 0.5, 0.0, -RAM_Z_ABOVE_GROUND], dtype=np.float64)
    bottom = (T_radar_world @ (T_world_opp_cog @ np.r_[opp_ground_in_ram - COG_IN_RAM, 1.0]))[:3][2]
    center[2] = bottom + box_lwh[2] * 0.5

    yaw = wrap_angle(yaw_from_R(T_world_opp_cog[:3, :3]) - yaw_from_R(T_world_radar[:3, :3]))
    v_world = opponent.velocity_world_at(opp_time)
    v_radar = T_radar_world[:3, :3] @ v_world
    return center.astype(np.float64), yaw, v_radar.astype(np.float64)


def local_points(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    d = points[:, :3].astype(np.float64) - center
    cy, sy = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return d @ rot.T


def box_mask(points: np.ndarray, center: np.ndarray, yaw: float, box_lwh: np.ndarray) -> np.ndarray:
    local = local_points(points, center, yaw)
    half = np.asarray(box_lwh, dtype=np.float64) * 0.5
    return (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (np.abs(local[:, 2]) <= half[2])
    )


def count_points_in_box(points: np.ndarray, center: np.ndarray, yaw: float, box_lwh: np.ndarray) -> int:
    return int(box_mask(points, center, yaw, box_lwh).sum())


def parse_ars548_detections(msg) -> np.ndarray:
    """Return Nx8 [x,y,z,rcs,radial_velocity,snr,rss,existence_prob]."""
    points = []
    max_u16 = 65535.0
    max_u8 = 255.0
    min_rss = -177.8
    max_range = 327.675
    max_snr = 63.75
    min_rcs, max_rcs = -100.0, 129.0
    min_rv, max_rv = -200.0, 127.675
    max_ambgt = 3000.0

    for det in msg.detections:
        range_m = max_range * float(det.range) / max_u16
        # Match the current cavauto RadarPoint conversion.  This should be
        # revisited if Continental/ZF scaling docs are added to the repo.
        az = ((math.pi * math.pi + math.pi / 2.0) * (float(det.azimuth_angle) / max_u16)) - math.pi / 2.0
        el = ((math.pi * math.pi + math.pi / 2.0) * (float(det.elevation_angle) / max_u16)) - math.pi / 2.0
        dx = math.sin(math.pi / 2.0 - el) * math.cos(az)
        dy = math.sin(math.pi / 2.0 - el) * math.sin(az)
        dz = math.cos(math.pi / 2.0 - el)
        radial_velocity = ((max_rv - min_rv) * float(det.radial_velocity) / max_u16) + min_rv
        rcs = ((max_rcs - min_rcs) * float(det.radar_cross_section) / max_u16) + min_rcs
        snr = max_snr * float(det.signal_noise_ratio) / max_u8
        _ambgt = max_ambgt * float(det.ambgt_id) / max_u16
        rss = (-min_rss * float(det.received_signal_strength) / max_u8) + min_rss
        points.append(
            [
                range_m * dx,
                range_m * dy,
                range_m * dz,
                rcs,
                radial_velocity,
                snr,
                rss,
                float(det.existence_prob),
            ]
        )
    return np.asarray(points, dtype=np.float32).reshape(-1, 8)


def transform_radar_points(
    points: np.ndarray,
    T_world_radar_i: np.ndarray,
    T_world_radar_t: np.ndarray,
    ego_compensated: bool,
) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    out = points.copy()
    if not ego_compensated:
        return out
    pts_h = np.column_stack((out[:, :3].astype(np.float64), np.ones(len(out), dtype=np.float64)))
    T_rt_ri = tf_inv(T_world_radar_t) @ T_world_radar_i
    out[:, :3] = (T_rt_ri @ pts_h.T).T[:, :3].astype(np.float32)
    return out


def write_imageset(path: Path, frame_ids: list[str]) -> None:
    text = "\n".join(frame_ids)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def check_container_requirements(cfg: RunConfig, *, require_image: bool, require_apptainer: bool = True) -> None:
    if require_apptainer and shutil.which("apptainer") is None:
        raise FileNotFoundError("apptainer is required but was not found on PATH")
    for bind in cfg.container_binds:
        if not bind.exists():
            raise FileNotFoundError(f"Configured Apptainer bind path does not exist: {bind}")
    if not cfg.repo_dir.is_dir():
        raise NotADirectoryError(cfg.repo_dir)
    if require_image and not cfg.container_image.is_file():
        raise FileNotFoundError(f"Configured Apptainer image does not exist: {cfg.container_image}")


def bash_quote(value: Path | str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def rel_to(path: Path, root: Path) -> str:
    path = path.resolve()
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def cfg_arg(cfg_path: Path, repo_dir: Path) -> str:
    return rel_to(cfg_path, repo_dir / "tools")


def apptainer_exec_cmd(cfg: RunConfig, command: list[str], *, writable_tmpfs: bool = False) -> list[str]:
    cmd = ["apptainer", "exec", "--nv", "--cleanenv"]
    if writable_tmpfs:
        cmd.append("--writable-tmpfs")
    for bind in cfg.container_binds:
        cmd.extend(["--bind", f"{bind}:{bind}"])
    cmd.extend(
        [
            "--env",
            f"PYTHONPATH={cfg.repo_dir / '.deps' / 'python'}:{cfg.repo_dir}",
            "--env",
            "PYTHONNOUSERSITE=1",
            "--env",
            f"PIPELINE_APPTAINER_IMAGE={cfg.container_image.resolve()}",
            str(cfg.container_image),
            *command,
        ]
    )
    return cmd


def inside_pipeline_container(cfg: RunConfig) -> bool:
    expected = str(cfg.container_image.resolve())
    return any(
        os.environ.get(name) == expected
        for name in ("PIPELINE_APPTAINER_IMAGE", "APPTAINER_CONTAINER", "SINGULARITY_CONTAINER")
    )


def job_config_path(cfg: RunConfig) -> Path:
    try:
        return cfg.repo_dir / cfg.path.relative_to(REPO_DIR)
    except ValueError:
        return cfg.path


class Runner:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(self, cmd: list[str], cwd: Path) -> None:
        log_kvs("command", [("cwd", cwd), ("cmd", format_cmd(cmd))])
        if self.dry_run:
            log_kv("dry_run", True)
            return
        subprocess.run(cmd, cwd=str(cwd), check=True)

    def capture_to_file(self, cmd: list[str], cwd: Path, out_file: Path) -> None:
        log_kvs("command", [("cwd", cwd), ("cmd", format_cmd(cmd)), ("stdout", out_file)])
        if self.dry_run:
            log_kv("dry_run", True)
            return
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=str(cwd), check=True, stdout=f)


def submit(cfg: RunConfig, mode: str, dry_run: bool, stages: str | None = None) -> None:
    if not dry_run:
        check_container_requirements(cfg, require_image=True, require_apptainer=False)
    is_setup = mode == "setup"
    job = "radar_setup" if is_setup else f"radar_{cfg.run_name}"
    gres = cfg.slurm_setup_gres if is_setup else cfg.slurm_pipeline_gres
    cpus = cfg.slurm_setup_cpus if is_setup else cfg.slurm_pipeline_cpus
    mem = cfg.slurm_setup_mem if is_setup else cfg.slurm_pipeline_mem
    walltime = cfg.slurm_setup_time if is_setup else cfg.slurm_pipeline_time
    cfg_for_job = job_config_path(cfg)
    if is_setup:
        inner = (
            "module purge && module load apptainer && command -v apptainer >/dev/null && "
            f"cd {bash_quote(cfg.repo_dir / 'tools')} && "
            f"python3 -u radar_pipeline.py --run-config {bash_quote(cfg_for_job)} --mode setup"
        )
    else:
        container_cmd = apptainer_exec_cmd(
            cfg,
            [
                "python3",
                "-u",
                str(cfg.repo_dir / "tools" / "radar_pipeline.py"),
                "--run-config",
                str(cfg_for_job),
                "--mode",
                "run",
            ],
        )
        if stages:
            container_cmd.extend(["--stages", stages])
        inner = (
            "module purge && module load apptainer && "
            f"cd {bash_quote(cfg.repo_dir / 'tools')} && {shlex.join(container_cmd)}"
        )
    cmd = [
        "sbatch",
        "-p",
        cfg.slurm_partition,
        "--gres",
        gres,
        "-c",
        str(cpus),
        "--mem",
        mem,
        "-t",
        walltime,
        "-J",
        job,
        "-o",
        str(cfg.log_dir / f"{job}-%A.out"),
        "-e",
        str(cfg.log_dir / f"{job}-%A.err"),
        "--export=NONE",
        "--wrap",
        f"bash -c {bash_quote(inner)}",
    ]
    log_kvs(
        "submit",
        [
            ("job", job),
            ("partition", cfg.slurm_partition),
            ("gres", gres),
            ("cpus", cpus),
            ("mem", mem),
            ("time", walltime),
            ("stdout", cfg.log_dir / f"{job}-%A.out"),
            ("stderr", cfg.log_dir / f"{job}-%A.err"),
            ("cmd", format_cmd(cmd)),
        ],
    )
    if not dry_run:
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True)


def parse_pipeline_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_run_config_arg(parser)
    parser.add_argument("--mode", choices=("run", "setup", "submit-run", "submit-setup"), required=True)
    parser.add_argument("--stages", choices=("all", "data", "smoke", "visualize", "train", "ssl", "transfer", "eval"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.run_cfg = load_run_config(args.run_config)
    return args
