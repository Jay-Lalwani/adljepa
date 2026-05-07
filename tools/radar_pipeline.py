#!/usr/bin/env python3
"""Config-driven radar data, pretraining, training, and evaluation pipeline."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import radar_common as rc


INFO_FILES = (
    "custom_radar_infos_train.pkl",
    "custom_radar_infos_val.pkl",
    "custom_radar_infos_test.pkl",
    "custom_radar_dbinfos_train.pkl",
)

PIPELINE_STAGES = {"all", "data", "smoke", "visualize", "train", "ssl", "transfer", "eval"}


def stage_enabled(stages: str, stage: str) -> bool:
    return stages in {"all", stage}


def remove_infos(data_root: Path) -> None:
    for name in INFO_FILES:
        target = data_root / name
        if target.exists():
            target.unlink()
    gt_db = data_root / "gt_database"
    if gt_db.is_dir():
        import shutil

        shutil.rmtree(gt_db)


def output_dir(cfg: rc.RunConfig, cfg_file: Path, extra_tag: str) -> Path:
    cfg_rel = rc.cfg_arg(cfg_file, cfg.repo_dir)
    parts = cfg_rel.split("/")
    exp_group = Path(*parts[1:-1]) if len(parts) > 2 else Path()
    return cfg.repo_dir / "output" / exp_group / cfg_file.stem / extra_tag


def latest_checkpoint(cfg: rc.RunConfig, cfg_file: Path, extra_tag: str) -> Path:
    ckpt_dir = output_dir(cfg, cfg_file, extra_tag) / "ckpt"
    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return ckpts[-1]


def setup_env(cfg: rc.RunConfig, dry_run: bool) -> None:
    if not dry_run:
        rc.check_container_requirements(cfg, require_image=True)
    script = """
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
cd {repo_dir}
python3 -m pip install --upgrade --target {deps_dir} spconv-cu113==2.3.6 cumm-cu113==0.4.11
python3 setup.py develop
python3 setup.py build_ext --inplace
python3 - <<'PY'
import torch
import pcdet
from pcdet.ops.iou3d_nms import iou3d_nms_cuda
from spconv.utils import Point2VoxelCPU3d
import spconv.pytorch as spconv
import mcap
import mcap_ros2
print(f"torch: {{torch.__version__}}")
print(f"torch cuda: {{torch.version.cuda}}")
print(f"cuda available: {{torch.cuda.is_available()}}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside the Apptainer container")
features = torch.randn(8, 4, device="cuda")
indices = torch.tensor(
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0],
     [0, 1, 1, 1], [0, 2, 2, 2], [0, 3, 3, 3], [0, 4, 4, 4]],
    dtype=torch.int32,
    device="cuda",
)
x = spconv.SparseConvTensor(features, indices, spatial_shape=[5, 5, 5], batch_size=1)
conv = spconv.SubMConv3d(4, 8, 3, padding=1, bias=False, indice_key="setup_probe").cuda()
conv(x)
print("pcdet/spconv cuda/mcap: OK")
PY
""".format(repo_dir=rc.bash_quote(cfg.repo_dir), deps_dir=rc.bash_quote(cfg.repo_dir / ".deps" / "python"))
    cmd = rc.apptainer_exec_cmd(cfg, ["bash", "-c", script], writable_tmpfs=True)
    rc.log_kvs("command", [("cmd", rc.format_cmd(cmd))])
    if not dry_run:
        subprocess.run(cmd, check=True)


def reexec_in_container(cfg: rc.RunConfig, args: argparse.Namespace) -> None:
    if rc.inside_pipeline_container(cfg):
        return
    cmd = rc.apptainer_exec_cmd(
        cfg,
        [
            "python3",
            str(cfg.repo_dir / "tools" / "radar_pipeline.py"),
            "--run-config",
            str(rc.job_config_path(cfg)),
            "--mode",
            "run",
            *(["--dry-run"] if args.dry_run else []),
        ],
    )
    rc.log_kvs("container reexec", [("cmd", rc.format_cmd(cmd))])
    if args.dry_run:
        return
    rc.check_container_requirements(cfg, require_image=True)
    os.execvp(cmd[0], cmd)


def pipeline(cfg: rc.RunConfig, args: argparse.Namespace) -> None:
    reexec_in_container(cfg, args)
    stages = str(args.stages or cfg.raw["pipeline"]["stages"])
    if stages not in PIPELINE_STAGES:
        raise ValueError(f"pipeline.stages must be one of: {', '.join(sorted(PIPELINE_STAGES))}")

    runner = rc.Runner(dry_run=args.dry_run)
    model_cfg_arg = rc.cfg_arg(cfg.model_cfg, cfg.repo_dir)
    ssl_cfg_arg = rc.cfg_arg(cfg.ssl_cfg, cfg.repo_dir)
    configured_pretrained = cfg.pretrained_model

    rc.log_kvs(
        "radar pipeline",
        [
            ("run", cfg.run_name),
            ("stages", stages),
            ("config", cfg.path),
            ("data_root", cfg.data_root),
            ("model_cfg", cfg.model_cfg),
            ("ssl_cfg", cfg.ssl_cfg),
        ],
    )

    if stage_enabled(stages, "data"):
        runner.run([sys.executable, "build_pp_radar.py", "--run-config", str(cfg.path)], cfg.repo_dir / "tools")
        if not args.dry_run:
            remove_infos(cfg.data_root)
        runner.run([sys.executable, "create_custom_radar_infos.py", "--run-config", str(cfg.path)], cfg.repo_dir / "tools")

    if stage_enabled(stages, "smoke"):
        runner.run(
            [sys.executable, "smoke_custom_radar_dataset.py", "--run-config", str(cfg.path), "--split", "train"],
            cfg.repo_dir / "tools",
        )

    if stage_enabled(stages, "visualize"):
        viz_cfg = cfg.raw.get("visualize", {})
        if not isinstance(viz_cfg, dict):
            raise ValueError("visualize must be a YAML mapping")
        visualize_cmd = [
            sys.executable,
            "visualize_custom_radar_frame.py",
            "--run-config",
            str(cfg.path),
            "--split",
            str(viz_cfg.get("split", "train")),
            "--start-index",
            str(viz_cfg.get("start_index", 0)),
            "--count",
            str(viz_cfg.get("count", 12)),
            "--stride",
            str(viz_cfg.get("stride", 250)),
        ]
        out_path = viz_cfg.get("out") or viz_cfg.get("out_path") or viz_cfg.get("out_dir")
        if out_path:
            visualize_cmd.extend(["--out", str(Path(out_path).expanduser())])
        if "dt" in viz_cfg:
            visualize_cmd.extend(["--dt", str(viz_cfg["dt"])])
        if viz_cfg.get("cfg_file"):
            visualize_cmd.extend(["--cfg-file", str(Path(viz_cfg["cfg_file"]).expanduser())])
        if viz_cfg.get("ckpt"):
            visualize_cmd.extend(["--ckpt", str(Path(viz_cfg["ckpt"]).expanduser())])
        if "score_thresh" in viz_cfg:
            visualize_cmd.extend(["--score-thresh", str(viz_cfg["score_thresh"])])
        runner.run(visualize_cmd, cfg.repo_dir / "tools")

    if stage_enabled(stages, "ssl"):
        runner.run(
            [
                sys.executable,
                "-u",
                "ssl_pretrain.py",
                "--cfg_file",
                ssl_cfg_arg,
                "--batch_size",
                str(cfg.batch_size),
                "--epochs",
                str(cfg.raw.get("ssl", {}).get("epochs", 30)),
                "--workers",
                str(cfg.workers),
                "--extra_tag",
                cfg.run_name,
                "--set",
                "DATA_CONFIG.DATA_PATH",
                str(cfg.data_root),
            ],
            cfg.repo_dir / "tools",
        )

    transfer_out = cfg.repo_dir / "output" / "custom_radar" / f"{cfg.run_name}_jepa_encoder_for_centerpoint.pth"
    if stage_enabled(stages, "transfer"):
        ssl_ckpt = cfg.ssl_checkpoint if cfg.ssl_checkpoint is not None else latest_checkpoint(cfg, cfg.ssl_cfg, cfg.run_name)
        runner.run(
            [
                sys.executable,
                "transfer_jepa_encoder_to_centerpoint.py",
                "--src",
                str(ssl_ckpt),
                "--dst",
                str(transfer_out),
            ],
            cfg.repo_dir / "tools",
        )
        configured_pretrained = transfer_out

    if stage_enabled(stages, "train"):
        train_cmd = [
            sys.executable,
            "-u",
            "train.py",
            "--cfg_file",
            model_cfg_arg,
            "--batch_size",
            str(cfg.batch_size),
            "--epochs",
            str(cfg.epochs),
            "--workers",
            str(cfg.workers),
            "--extra_tag",
            cfg.run_name,
        ]
        if configured_pretrained:
            train_cmd.extend(["--pretrained_model", str(configured_pretrained)])
        train_cmd.extend(["--set", "DATA_CONFIG.DATA_PATH", str(cfg.data_root)])
        runner.run(train_cmd, cfg.repo_dir / "tools")

    if stages == "eval":
        if cfg.checkpoint is None:
            raise ValueError("model.checkpoint is required when pipeline.stages is eval")
        runner.run(
            [
                sys.executable,
                "test.py",
                "--cfg_file",
                model_cfg_arg,
                "--ckpt",
                str(cfg.checkpoint),
                "--batch_size",
                str(cfg.eval_batch_size),
                "--workers",
                str(cfg.workers),
                "--set",
                "DATA_CONFIG.DATA_PATH",
                str(cfg.data_root),
            ],
            cfg.repo_dir / "tools",
        )


def main() -> None:
    args = rc.parse_pipeline_args()
    cfg: rc.RunConfig = args.run_cfg
    if args.mode == "submit-setup":
        rc.submit(cfg, "setup", args.dry_run, stages=args.stages)
    elif args.mode == "submit-run":
        rc.submit(cfg, "run", args.dry_run, stages=args.stages)
    elif args.mode == "setup":
        setup_env(cfg, args.dry_run)
    elif args.mode == "run":
        pipeline(cfg, args)


if __name__ == "__main__":
    main()
