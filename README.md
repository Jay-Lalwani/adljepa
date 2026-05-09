# Radar JEPA

This fork adapts AD-L-JEPA, originally built for LiDAR self-supervised pretraining, to radar-only object detection. It builds OpenPCDet-style datasets from ARS548 radar MCAP logs, trains a supervised radar CenterPoint baseline, pretrains a radar JEPA encoder on unlabeled radar frames, transfers that encoder into CenterPoint, and evaluates both detectors on the same validation split.

The short technical summary and final results are in [tools/RADAR_JEPA_REPORT.md](tools/RADAR_JEPA_REPORT.md).

## What This Repo Adds

- A Putnam Park radar dataset builder for front ARS548 MCAP data.
- Five-sweep radar stacking with ego-motion compensation.
- Radar labels with 3D boxes, heading, and Cartesian velocity.
- Controlled empty-label frames for cases where GT exists but radar has too few points on the object.
- Smoke tests and Foxglove MCAP visualization for points, GT boxes, and predictions.
- Custom radar evaluation metrics: BEV AP, precision/recall/F1, center error, velocity error, yaw error, and false positives per frame.
- JEPA encoder transfer from radar SSL pretraining into a standard CenterPoint detector.

## Main Configs

- Supervised training/eval/visualization: `tools/radar_run_config.yaml`
- JEPA pretraining/transfer: `tools/radar_run_config_jepa_pretrain.yaml`
- CenterPoint model: `tools/cfgs/custom_radar_models/radar_centerpoint_front_ars548.yaml`
- AD-L-JEPA model: `tools/cfgs/custom_radar_models/radar_ad_l_jepa_front_ars548.yaml`

The configs are written for the UVA HPC environment and use Apptainer plus Slurm. Update the paths in the YAML files if running elsewhere.

## Standard Workflow

Run commands from `tools/`.

```bash
cd /p/cavalier/jay/radar-jepa/tools
```

Build the supervised radar dataset:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages data
```

Check the dataset:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages smoke
```

Visualize points, labels, and predictions:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages visualize
```

Train the supervised CenterPoint baseline:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages train
```

Pretrain AD-L-JEPA on unlabeled radar:

```bash
python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages ssl
```

Transfer the JEPA encoder into CenterPoint format:

```bash
python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages transfer
```

Fine-tune CenterPoint from JEPA weights by setting `model.pretrained_model` in `radar_run_config.yaml`, then run:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages train
```

Evaluate a checkpoint by setting `model.checkpoint` in `radar_run_config.yaml`, then run:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages eval
```

Use `--dry-run` before submitting if you want to inspect the generated Slurm command:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages train --dry-run
```

## Original Project

This work is based on AD-L-JEPA: *Self-Supervised Representation Learning with Joint Embedding Predictive Architecture for Automotive LiDAR Object Detection*.

```bibtex
@misc{zhu2025adljepa,
  title={Self-Supervised Representation Learning with Joint Embedding Predictive Architecture for Automotive LiDAR Object Detection},
  author={Haoran Zhu and Zhenyuan Dong and Kristi Topollai and Beiyao Sha and Anna Choromanska},
  year={2025},
  eprint={2501.04969},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2501.04969}
}
```
