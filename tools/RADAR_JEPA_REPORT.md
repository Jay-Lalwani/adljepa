# Radar AD-L-JEPA Experiment Report

## Summary

This repository extends AD-L-JEPA, originally developed for self-supervised LiDAR representation learning, to a radar-only object detection setting. The experiment evaluates whether radar-domain JEPA pretraining can improve a CenterPoint detector on sparse and noisy multi-sweep ARS548 radar data.

The final controlled comparison used the same supervised radar dataset, detector architecture, training schedule, and validation split for both models. The baseline was a randomly initialized radar-only CenterPoint model. The JEPA variant used the same CenterPoint detector, but initialized its sparse 3D backbone from a radar-domain AD-L-JEPA pretraining run before supervised fine-tuning.

The strongest JEPA run improved precision, F1, false positives per frame, and center localization while preserving recall. BEV AP and velocity error did not improve, so the result should be interpreted as a useful radar false-positive/localization improvement rather than a uniform improvement across every metric.

## Research Basis

JEPA learns representations by predicting latent embeddings of missing or masked input regions from visible context, rather than reconstructing raw inputs or constructing contrastive pairs. AD-L-JEPA applies this idea to autonomous-driving LiDAR by masking BEV regions, predicting target encoder embeddings, using a lightweight spatial predictor, applying variance regularization to avoid collapse, and updating the target encoder with an exponential moving average.

This project applies the same core idea to automotive radar. Radar differs from LiDAR because radar point clouds are substantially sparser, noisier, and more affected by multipath, Doppler ambiguity, and intermittent visibility. The working hypothesis was that a JEPA pretext task could encourage the sparse 3D encoder to learn more stable radar scene structure before supervised detector training.

## Data Pipeline

The dataset was generated from Putnam Park ARS548 front radar MCAP data with synchronized ego odometry and opponent ground truth. Each training sample contains a 5-sweep radar stack in the current radar frame. Older sweeps are ego-motion compensated using odometry and radar extrinsics, and each point receives a normalized time offset.

Point features are:

```text
[x, y, z, rcs, radial_velocity, snr, rss, existence_prob, time_delta]
```

Labels use a single `Car` class with boxes and velocity:

```text
[x, y, z, length, width, height, heading, vx, vy, class_name]
```

The supervised dataset includes positive frames and a controlled number of empty-label frames. In this context, an empty supervised frame means the opponent pose exists but fewer than `min_points_in_gt` radar points fall inside the GT box. This was important because radar often cannot see the opponent even when the GT object exists, and training only on visible positives produced many ghost false positives.

The SSL dataset keeps all valid radar frames with nonzero points. Labels are not used during JEPA pretraining.

## Model Architectures

**Baseline CenterPoint**

The baseline uses the standard OpenPCDet CenterPoint path:

```text
multi-sweep radar points
-> voxelization
-> MeanVFE
-> VoxelBackBone8x
-> HeightCompression
-> BaseBEVBackbone
-> CenterHead
-> box, heading, score, vx/vy
```

All weights are randomly initialized and trained with supervised labels.

**JEPA-Pretrained CenterPoint**

The JEPA model uses the same detector at inference time. The only controlled difference is initialization of the sparse 3D backbone.

Pretraining uses:

```text
multi-sweep radar voxels
-> context encoder
-> BEV masking with empty and non-empty regions
-> spatial predictor
-> target encoder embeddings
-> cosine embedding loss + variance regularization + EMA target update
```

After SSL pretraining, only `backbone_3d.encoder.*` is transferred into CenterPoint as `backbone_3d.*`. The predictor, target encoder, mask token, and empty token are pretraining-only and are not used at detection time.

## Experiment Stages

1. Built 5-sweep ego-compensated radar datasets from MCAP data.
2. Added visualization via MCAP/Foxglove to inspect radar points, GT boxes, and predictions.
3. Trained a positive-only radar CenterPoint baseline.
4. Added low-point empty-label frames to reduce ghost detections.
5. Trained the supervised neg20 CenterPoint baseline.
6. Pretrained AD-L-JEPA on all valid unlabeled radar frames.
7. Transferred the JEPA context encoder into CenterPoint.
8. Fine-tuned the JEPA-initialized detector on the exact same supervised neg20 dataset.
9. Evaluated all models on the same validation split with custom radar metrics.

## Final Results

All final comparisons below use the same `custom_radar_front_ars548_neg20` validation split with 741 frames and 593 GT boxes.

| Model | BEV AP@0.50 | Precision | Recall | F1 | FP/frame | Center mean | Velocity mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised CenterPoint baseline | `0.7854` | `0.7266` | `0.8516` | `0.7842` | `0.2564` | `0.3380 m` | `0.7176 m/s` |
| JEPA beta1 pretrain + fine-tune | `0.7751` | `0.7094` | `0.8398` | `0.7691` | `0.2753` | `0.3296 m` | `0.7236 m/s` |
| JEPA beta10 pretrain + fine-tune | `0.7682` | `0.7660` | `0.8499` | `0.8058` | `0.2078` | `0.3218 m` | `0.7406 m/s` |

Relative to the supervised baseline, the best JEPA run:

- Increased precision from `0.7266` to `0.7660`.
- Increased F1 from `0.7842` to `0.8058`.
- Reduced false positives per frame from `0.2564` to `0.2078`.
- Improved mean center error from `0.3380 m` to `0.3218 m`.
- Preserved recall at nearly the same level: `0.8516` vs `0.8499`.
- Did not improve BEV AP or velocity error.

## Interpretation

The most plausible explanation is that radar JEPA pretraining improved the sparse 3D backbone's representation of stable radar scene structure and empty/non-empty BEV context. This made the downstream detector more conservative and spatially cleaner, reducing ghost false positives while preserving recall. That aligns with the JEPA objective: it does not explicitly denoise radar points, but it can bias the encoder toward embeddings that better distinguish plausible object-supporting radar structure from background or multipath-like clutter.

The lack of BEV AP and velocity improvement is also informative. The supervised baseline already had strong full-label training on the same target distribution, and the SSL pretraining data came from a single primary run. JEPA gains are expected to be strongest with more unlabeled diversity or lower label availability. Future work should therefore evaluate label efficiency and expand SSL pretraining to additional radar sessions before adding architectural complexity.

## Usage

Run commands from `/p/cavalier/jay/radar-jepa/tools`.

Build supervised radar data:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages data
```

Train the supervised CenterPoint baseline:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages train
```

Pretrain radar AD-L-JEPA:

```bash
python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages ssl
```

Transfer the JEPA encoder:

```bash
python3 radar_pipeline.py --run-config radar_run_config_jepa_pretrain.yaml --mode submit-run --stages transfer
```

Fine-tune CenterPoint from JEPA weights by setting `model.pretrained_model` in `radar_run_config.yaml`, then running:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages train
```

Evaluate a checkpoint by setting `model.checkpoint` and running:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages eval
```

Visualize data or predictions:

```bash
python3 radar_pipeline.py --run-config radar_run_config.yaml --mode submit-run --stages visualize
```
