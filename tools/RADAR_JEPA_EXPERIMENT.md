# Radar JEPA Experiment

## Goal

Test whether radar-domain AD-L-JEPA pretraining improves a strong radar-only CenterPoint detector on sparse/noisy 5-sweep front ARS548 radar.

The primary comparison is simple: train the same CenterPoint detector twice on the same supervised radar dataset, once from random initialization and once from a JEPA-pretrained 3D backbone. Success is measured on the same validation split by BEV AP, precision/recall/F1, center error, velocity error, yaw error, false positives per frame, and eventually inference latency.

## Canonical Inputs

- Raw run: `/p/cavalier/data/raw/PP/2026_04_28/run2`
- Ego odom topic: `/vehicle/uva_odometry`
- Opponent GT: `/p/cavalier/data/raw/PP/2026_04_28/run2/purdue_gt_from_raw`
- Radar topic: `/radar_front/ars548_process/detections`
- Supervised dataset: `/p/cavalier/jay/radar-jepa/data/custom_radar_front_ars548_neg20`
- SSL dataset: `/p/cavalier/jay/radar-jepa/data/custom_radar_front_ars548_ssl_all`
- Feature vector: `[x, y, z, rcs, radial_velocity, snr, rss, existence_prob, time_delta]`

The supervised dataset keeps positive frames plus a capped set of low-point negative frames. In this project, an "empty" supervised label file means the opponent GT pose exists, but the generated radar frame has fewer than `dataset.min_points_in_gt` radar points inside the GT box. It does not mean missing GT. The current builds have `no_pose_time: 0`.

The SSL dataset keeps all valid radar frames with nonzero points. Box labels are irrelevant for JEPA pretraining.

## Architecture

**Baseline CenterPoint**

Radar points are ego-motion compensated into the current radar frame, voxelized, encoded by `MeanVFE`, processed by `VoxelBackBone8x`, compressed to BEV, passed through `BaseBEVBackbone`, and decoded by `CenterHead` into box center, dimensions, yaw, score, and Cartesian velocity. All trainable weights start from random initialization and are trained with supervised labels.

**JEPA-Initialized CenterPoint**

AD-L-JEPA pretraining uses the same radar voxel input but no labels. It masks sparse BEV/voxel regions, runs a context encoder, predicts target-encoder embeddings with a small predictor, and optimizes embedding prediction with EMA target updates. This learns radar structure without reconstructing raw points.

For detection, only `backbone_3d.encoder.*` is transferred into the CenterPoint `backbone_3d.*`. The target encoder, predictor, mask token, and empty token are pretraining-only and are not deployed. After transfer, inference architecture is the same as the baseline CenterPoint; the controlled difference is the 3D backbone initialization.

## Completed Runs

| Run | Dataset | Checkpoint | BEV AP@0.50 | Precision | Recall | F1 | FP/frame | Center mean | Velocity mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical positive-only baseline | `custom_radar_front_ars548` | `.../pp_20260428_purdue_front_ars548_5sweep/ckpt/checkpoint_epoch_80.pth` | `0.7747` | `0.4725` | `0.8550` | `0.6086` | `0.9545` | `0.3217 m` | `0.8052 m/s` |
| Current supervised baseline | `custom_radar_front_ars548_neg20` | `.../pp_20260428_purdue_front_ars548_5sweep_neg20/ckpt/checkpoint_epoch_80.pth` | `0.7854` | `0.7266` | `0.8516` | `0.7842` | `0.2564` | `0.3380 m` | `0.7176 m/s` |
| AD-L-JEPA SSL pretrain, beta1 | `custom_radar_front_ars548_ssl_all` | `.../pp_20260428_purdue_front_ars548_5sweep_jepa_pretrain_ssl_all/ckpt/checkpoint_epoch_30.pth` | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| JEPA fine-tune, beta1 pretrain | `custom_radar_front_ars548_neg20` | `.../pp_20260428_purdue_front_ars548_5sweep_neg20_jepa_ft/ckpt/checkpoint_epoch_80.pth` | `0.7751` | `0.7094` | `0.8398` | `0.7691` | `0.2753` | `0.3296 m` | `0.7236 m/s` |
| AD-L-JEPA SSL pretrain, beta10 | `custom_radar_front_ars548_ssl_all` | `.../pp_20260428_purdue_front_ars548_5sweep_jepa_pretrain_ssl_all_beta10/ckpt/checkpoint_epoch_30.pth` | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| JEPA fine-tune, beta10 pretrain | `custom_radar_front_ars548_neg20` | `.../pp_20260428_purdue_front_ars548_5sweep_neg20_jepa_beta10_ft/ckpt/checkpoint_epoch_80.pth` | `0.7682` | `0.7660` | `0.8499` | `0.8058` | `0.2078` | `0.3218 m` | `0.7406 m/s` |

The current supervised baseline is the reference model. Adding low-point empty labels kept recall essentially flat while reducing false positives by roughly 73% versus the positive-only run. The beta10 JEPA run produced the best precision, F1, false-positive rate, and center error, while BEV AP and velocity error remained better in the supervised baseline.

Detailed current baseline metrics:

- Validation frames: `741`
- GT boxes: `593`
- Predictions: `695`
- True positives / false positives / false negatives at BEV IoU 0.50: `505 / 190 / 88`
- Center error mean / median / p90: `0.3380 / 0.2785 / 0.6330 m`
- Velocity L2 error mean / median / p90: `0.7176 / 0.4461 / 1.5718 m/s`
- Yaw error mean / median / p90: `0.0285 / 0.0189 / 0.0674 rad`

## Dataset Summaries

**Supervised neg20**

- Train / val / test frames: `6670 / 741 / 2067`
- Low-point empty train / val frames kept: `1334 / 148`
- Labeled positive frames: `5929`
- Low-point frames available before capping: `30394`
- Average points/frame: `4114.4`

**SSL all**

- Train / val / test frames: `32407 / 3916 / 2067`
- Low-point train / val frames kept: `27071 / 3323`
- Average points/frame: `4580.2`

## Next Steps

1. Transfer the improved JEPA encoder.

   ```bash
   cd /p/cavalier/jay/radar-jepa/tools
   python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages transfer
   ```

   Expected transferred checkpoint:

   ```text
   /p/cavalier/jay/radar-jepa/output/custom_radar/pp_20260428_purdue_front_ars548_5sweep_jepa_pretrain_ssl_all_beta10_jepa_encoder_for_centerpoint.pth
   ```

2. Fine-tune CenterPoint from the transferred JEPA encoder on the supervised neg20 dataset.

   In `tools/radar_run_config.yaml`, set:

   ```yaml
   run_name: pp_20260428_purdue_front_ars548_5sweep_neg20_jepa_beta10_ft
   paths:
     data_root: /p/cavalier/jay/radar-jepa/data/custom_radar_front_ars548_neg20
   model:
     pretrained_model: /p/cavalier/jay/radar-jepa/output/custom_radar/pp_20260428_purdue_front_ars548_5sweep_jepa_pretrain_ssl_all_beta10_jepa_encoder_for_centerpoint.pth
   pipeline:
     stages: train
   ```

   Then submit:

   ```bash
   cd /p/cavalier/jay/radar-jepa/tools
   python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run
   ```

3. Evaluate the JEPA fine-tuned detector on the same neg20 validation split.

   After training, set:

   ```yaml
   model:
     checkpoint: /p/cavalier/jay/radar-jepa/output/custom_radar_models/radar_centerpoint_front_ars548/pp_20260428_purdue_front_ars548_5sweep_neg20_jepa_beta10_ft/ckpt/checkpoint_epoch_80.pth
   pipeline:
     stages: eval
   ```

   Then submit:

   ```bash
   cd /p/cavalier/jay/radar-jepa/tools
   python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run
   ```

4. Visualize JEPA predictions with the same checkpoint.

   Set `pipeline.stages: visualize`, keep `model.checkpoint` pointed at the JEPA fine-tuned checkpoint, and submit the pipeline. The MCAP overlays `/radar/points`, `/radar/gt_opp`, and model predictions.

5. Fill the final comparison table.

   Compare only runs evaluated on `custom_radar_front_ars548_neg20`:

   - Random-init CenterPoint epoch 80: current baseline.
   - JEPA beta1 full fine-tune epoch 80: completed negative-transfer result.
   - JEPA beta10 full fine-tune epoch 80: best current JEPA result.
   - Optional label-efficiency runs if full-label JEPA remains neutral or negative.

## Pretraining Diagnostics

The beta1 SSL run was not obviously broken, but it was weak for the radar setting:

- Non-empty target cosine loss improved from about `0.36` in epoch 1 to `0.16` in epoch 30.
- Empty target cosine loss stayed near `0.42`, so the model did not substantially improve empty-region prediction.
- Non-empty prediction variance ended near `0.00327`, below the nominal `(1/16)^2 = 0.00391` threshold, so variance regularization was still active at the end.
- The SSL dataset is one Putnam Park run, which is useful but not very diverse. JEPA benefits most when unlabeled pretraining adds scene diversity beyond the labeled set.

The cleanest next attempt is `BETA: 10.0`, following the repo's Waymo config. If that still does not improve full-label detection, the next useful JEPA tests are label-efficiency and larger/diverse SSL data, not architectural changes.

## Commands

Run a dry-run before submitting Slurm jobs:

```bash
cd /p/cavalier/jay/radar-jepa/tools
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --dry-run
```

Common stage overrides:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages visualize
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages eval
python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages transfer
```

## Interpretation Rules

- Do not evaluate the raw JEPA pretraining checkpoint as a detector; it has no trained detection head.
- Do not compare the JEPA fine-tune against the old positive-only baseline as the main result; the correct baseline is the neg20 supervised CenterPoint.
- Keep the supervised dataset, detector config, train schedule, and validation split identical between baseline and JEPA fine-tune. The intended difference is only backbone initialization.
- If JEPA does not beat the neg20 baseline, the clean conclusion is that AD-L-JEPA pretraining did not improve this detector under the current data and protocol.

## Registry Scope

This AD-L-JEPA clone contains registry references and `.pyc` files for Voxel-MAE, but not the matching `.py` source files. The radar experiment uses the source-backed CenterPoint and AD-L-JEPA paths only, so the model registries intentionally expose those available implementations rather than importing missing modules.
