# Radar JEPA Experiment Runbook

This repo clone is configured for the first front-radar experiment:

- Data: `/p/cavalier/data/raw/PP/2026_04_28/run2`
- Ego odom: `/vehicle/uva_odometry`
- Opponent GT: `/p/cavalier/data/raw/PP/2026_04_28/run2/purdue_gt_from_raw`
- Radar: `/radar_front/ars548_process/detections`
- Supervised dataset: `/p/cavalier/jay/radar-jepa/data/custom_radar_front_ars548_neg20`
- SSL dataset: `/p/cavalier/jay/radar-jepa/data/custom_radar_front_ars548_ssl_all`

## Slurm Workflow

From `/p/cavalier/jay/radar-jepa/tools`:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-setup
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run
```

Use `pipeline.stages` or the CLI `--stages` override to select one stage: `data`, `smoke`, `visualize`, `train`, `ssl`, `transfer`, or `eval`.
The `all` stage runs the normal sequence end-to-end, but for this experiment the safer workflow is to run each milestone explicitly.

Useful dry-runs:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-setup --dry-run
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --dry-run
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode run --stages data --dry-run
```

## Milestones

1. Build supervised data with controlled empty-frame negatives:
   ```bash
   python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages data
   ```

2. Build SSL data with all valid radar frames:
   ```bash
   python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages data
   ```

3. Smoke check after data generation:
   ```bash
   python3 smoke_custom_radar_dataset.py --run-config radar_run_config.yaml --split train
   ```

4. Write a Foxglove MCAP for 3D spot checks:
   ```bash
   python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages visualize

   # Or render one frame directly:
   python3 visualize_custom_radar_frame.py --run-config radar_run_config.yaml --frame-id 000000 --out /p/cavalier/jay/logs/radar_frame_000000.mcap
   ```

   The MCAP contains `/radar/points` from `data_root/points/*.bin` and `/radar/gt_opp` from `data_root/labels/*.txt`. Motion compensation is controlled by `dataset.ego_compensated`; rebuild the dataset with that flag changed to visualize the uncompensated version.

5. Train supervised baseline:
   ```bash
   python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages train
   ```

6. Train AD-L-JEPA on the SSL-all dataset:
   ```bash
   python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages ssl
   ```

7. Transfer JEPA encoder:
   Run `--stages transfer` and either set `model.ssl_checkpoint` or let the pipeline use the latest SSL checkpoint for this run.

8. Fine-tune CenterPoint with transferred weights:
   Run `--stages train` with `model.pretrained_model` set to the transferred checkpoint.

## Empty Frames

Supervised data uses `dataset.mode: supervised` and keeps labeled positive frames plus a bounded empty-frame fraction. The current setting is `empty_train_fraction: 0.20` and `empty_val_fraction: 0.20`, meaning empty frames are capped at roughly 20% of the final train/val split, not 20% of all possible empty radar frames.

SSL data uses `dataset.mode: ssl`, which keeps every valid radar frame with nonzero points, including frames with empty labels. AD-L-JEPA ignores box supervision, so these empty labels are only there to satisfy the standard OpenPCDet dataset contract.

`ALLOW_EMPTY_GT: True` is enabled only in configs that are allowed to consume empty label files. The dataset still rejects samples with zero points or zero voxels during training.

## Required Ablations

Use git branches or commits for ablations rather than adding permanent run configs. Vary only these dataset fields:

- `1sweep`: `sweep_count: 1`, `ego_compensated: true`
- `5sweep_no_egocomp`: `sweep_count: 5`, `ego_compensated: false`
- `5sweep_egocomp`: `sweep_count: 5`, `ego_compensated: true`

Use unique `run_name` and `data_root` for each ablation so generated datasets and outputs do not overwrite each other.

## Baseline Result So Far

The positive-only epoch-80 CenterPoint checkpoint evaluated on 593 val frames after fixing the custom BEV IoU helper:

- BEV AP@0.50: `0.7747`
- Recall@0.50: `0.8550`
- Precision@0.50: `0.4725`
- Center error mean/median/p90: `0.322 / 0.258 / 0.635 m`
- Velocity L2 error mean/median/p90: `0.805 / 0.402 / 2.163 m/s`
- False positives/frame: `0.954`

For train/val, `dataset.min_points_in_gt` applies the same basic quality gate as the LiDAR dataset builder: frames are skipped unless the generated GT box contains at least that many radar points. Rebuild the dataset after changing this value.

## Registry Scope

This AD-L-JEPA clone contains registry references and `.pyc` files for Voxel-MAE, but not the matching `.py` source files. The radar experiment uses the source-backed CenterPoint and AD-L-JEPA paths only, so the model registries intentionally expose those available implementations rather than importing missing Voxel-MAE modules.
