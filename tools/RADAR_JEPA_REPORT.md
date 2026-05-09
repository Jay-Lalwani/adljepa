# Radar JEPA Summary

We tested whether a JEPA-style self-supervised encoder can improve radar-only object detection. We built on AD-L-JEPA, which was originally proposed for LiDAR: instead of reconstructing masked points or making contrastive pairs, it learns by predicting masked BEV embeddings from visible context. Our contribution was applying that idea to sparse, noisy automotive radar and wiring it into a full detection pipeline.

The data pipeline starts from raw ARS548 front-radar MCAP logs. We synchronize ego odometry and opponent ground truth, stack five radar sweeps, ego-motion compensate older sweeps into the current radar frame, and write OpenPCDet-compatible point and label files. Each radar point contains position, radar return features, radial velocity, and time offset. Labels contain 3D box position, size, heading, and Cartesian velocity. A practical fix in dataset construction was adding controlled empty-label frames, which made the supervised baseline much less prone to ghost detections.

The final experiment used the Putnam Park `2026_04_28/run2` Purdue opponent data with the front ARS548 radar. The supervised train/validation split contained `7,411` frames: `5,929` frames with radar-supported opponent labels and `1,482` intentionally empty-label frames. JEPA pretraining used the larger radar-only split of `36,323` train/validation frames from the same run, including frames where the opponent was not visible to radar.

The baseline model is radar-only CenterPoint: voxelized radar points go through `MeanVFE`, `VoxelBackBone8x`, `HeightCompression`, `BaseBEVBackbone`, and `CenterHead` to predict boxes, heading, score, and velocity. The JEPA model uses the same detector at inference time. The only architectural difference is initialization: before supervised training, the sparse 3D backbone is pretrained with AD-L-JEPA on unlabeled radar frames. During pretraining, a context encoder predicts target encoder BEV embeddings for masked empty and non-empty regions, using cosine embedding loss, variance regularization, and EMA target updates. After pretraining, only the context encoder weights are transferred into CenterPoint; the JEPA predictor, target encoder, mask token, and empty token are discarded.

The controlled comparison used the same supervised dataset, detector config, training schedule, and validation split. The strongest JEPA run improved the most important radar behavior: it produced fewer false positives while keeping recall essentially unchanged. It did not improve every metric, especially BEV AP and velocity error, so we can preliminarily conclude that radar JEPA helped false-positive suppression and localization quality in this setting, not that it universally improved detection.

| Model | BEV AP@0.50 | Precision | Recall | F1 | FP/frame | Center mean | Velocity mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised CenterPoint | `0.7854` | `0.7266` | `0.8516` | `0.7842` | `0.2564` | `0.3380 m` | `0.7176 m/s` |
| JEPA + fine-tune | `0.7682` | `0.7660` | `0.8499` | `0.8058` | `0.2078` | `0.3218 m` | `0.7406 m/s` |

Relative to the supervised baseline, the best JEPA run raised precision from `0.7266` to `0.7660`, raised F1 from `0.7842` to `0.8058`, reduced false positives per frame from `0.2564` to `0.2078`, and improved mean center error from `0.3380 m` to `0.3218 m`, while recall stayed nearly the same (`0.8516` vs `0.8499`). A plausible explanation is that the radar JEPA pretext task taught the sparse backbone a cleaner notion of radar-supported versus empty BEV structure, which made the final detector less likely to hallucinate objects from clutter or multipath-like returns.

In the future, we plan to incorporate a much larger corpus of data for JEPA pretraining to construct a more effective latent representation of the noisy radar data, and thus better measure its effectiveness (currently used ~40,000 frames). 
