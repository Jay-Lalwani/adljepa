from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from . import custom_radar_utils as radar_utils


class CustomRadarDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / "ImageSets" / f"{self.split}.txt"
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []
        self.custom_radar_infos = []
        self.include_custom_radar_data(self.mode)

    def include_custom_radar_data(self, mode):
        infos = []
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, "rb") as f:
                infos.extend(pickle.load(f))
        self.custom_radar_infos.extend(infos)
        if self.logger is not None:
            self.logger.info("Total samples for CustomRadarDataset: %d" % len(infos))

    @property
    def num_point_features(self) -> int:
        return len(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / "ImageSets" / f"{self.split}.txt"
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else []

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.custom_radar_infos) * self.total_epochs
        return len(self.custom_radar_infos)

    def get_lidar(self, idx):
        point_file = self.root_path / "points" / f"{idx}.bin"
        assert point_file.exists(), point_file
        points = np.fromfile(str(point_file), dtype=np.float32)
        points = points.reshape(-1, self.num_point_features)
        return self.normalize_features(points)

    def normalize_features(self, points):
        norm_cfg = self.dataset_cfg.get("FEATURE_NORMALIZATION", None)
        if norm_cfg is None or len(points) == 0:
            return points
        points = points.copy()
        feature_names = list(self.dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)
        for name, cfg in norm_cfg.items():
            if name not in feature_names:
                continue
            col = feature_names.index(name)
            if "clip" in cfg:
                points[:, col] = np.clip(points[:, col], float(cfg.clip[0]), float(cfg.clip[1]))
            mean = float(cfg.get("mean", 0.0))
            std = max(float(cfg.get("std", 1.0)), 1e-6)
            points[:, col] = (points[:, col] - mean) / std
        return points

    def get_label(self, idx):
        return radar_utils.read_label(self.root_path / "labels" / f"{idx}.txt")

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        import torch

        def process_single_scene(sample_idx):
            print("%s sample_idx: %s" % (self.split, sample_idx))
            points = self.get_lidar(sample_idx)
            info = {
                "point_cloud": {"num_features": self.num_point_features, "lidar_idx": sample_idx},
                "frame_id": sample_idx,
            }
            if has_label:
                gt_boxes, gt_names = self.get_label(sample_idx)
                annotations = {
                    "name": gt_names,
                    "difficulty": np.zeros(len(gt_names), dtype=np.int32),
                    "bbox": np.zeros((len(gt_names), 4), dtype=np.float32),
                    "score": np.ones(len(gt_names), dtype=np.float32),
                    "gt_boxes_lidar": gt_boxes,
                    "index": np.arange(len(gt_names), dtype=np.int32),
                }
                if count_inside_pts and len(gt_boxes) > 0:
                    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                        torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes[:, :7])
                    ).numpy()
                    annotations["num_points_in_gt"] = point_indices.sum(axis=1).astype(np.int32)
                else:
                    annotations["num_points_in_gt"] = np.zeros(len(gt_names), dtype=np.int32)
                info["annos"] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            return list(executor.map(process_single_scene, sample_id_list))

    def create_groundtruth_database(self, info_path=None, used_classes=None, split="train"):
        import torch

        database_save_path = Path(self.root_path) / ("gt_database" if split == "train" else f"gt_database_{split}")
        db_info_save_path = Path(self.root_path) / f"custom_radar_dbinfos_{split}.pkl"
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, "rb") as f:
            infos = pickle.load(f)

        for k, info in enumerate(infos):
            print("gt_database sample: %d/%d" % (k + 1, len(infos)))
            sample_idx = info["point_cloud"]["lidar_idx"]
            points = self.get_lidar(sample_idx)
            annos = info.get("annos", {})
            names = annos.get("name", np.asarray([]))
            gt_boxes = annos.get("gt_boxes_lidar", np.zeros((0, 9), dtype=np.float32))
            if len(gt_boxes) == 0:
                continue
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes[:, :7])
            ).numpy()
            for i in range(gt_boxes.shape[0]):
                filename = f"{sample_idx}_{names[i]}_{i}.bin"
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, "w") as f:
                    gt_points.tofile(f)
                if used_classes is None or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))
                    db_info = {
                        "name": names[i],
                        "path": db_path,
                        "image_idx": sample_idx,
                        "gt_idx": i,
                        "box3d_lidar": gt_boxes[i],
                        "num_points_in_gt": gt_points.shape[0],
                        "difficulty": 0,
                        "bbox": np.zeros(4, dtype=np.float32),
                        "score": 1.0,
                    }
                    all_db_infos.setdefault(names[i], []).append(db_info)
        for k, v in all_db_infos.items():
            print("Database %s: %d" % (k, len(v)))
        with open(db_info_save_path, "wb") as f:
            pickle.dump(all_db_infos, f)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_radar_infos)
        info = copy.deepcopy(self.custom_radar_infos[index])
        sample_idx = info["point_cloud"]["lidar_idx"]
        points = self.get_lidar(sample_idx)
        input_dict = {"points": points, "frame_id": sample_idx}
        if "annos" in info:
            annos = info["annos"]
            input_dict.update({"gt_names": annos["name"], "gt_boxes": annos["gt_boxes_lidar"].astype(np.float32)})
        return self.prepare_data(data_dict=input_dict)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        def template(n):
            return {
                "name": np.zeros(n),
                "score": np.zeros(n),
                "boxes_lidar": np.zeros((n, 9), dtype=np.float32),
                "pred_labels": np.zeros(n, dtype=np.int32),
            }

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = template(pred_scores.shape[0])
            if pred_scores.shape[0] > 0:
                pred_dict["name"] = np.array(class_names)[pred_labels - 1]
                pred_dict["score"] = pred_scores
                pred_dict["boxes_lidar"][:, : pred_boxes.shape[1]] = pred_boxes
                pred_dict["pred_labels"] = pred_labels
            pred_dict["frame_id"] = batch_dict["frame_id"][index]
            annos.append(pred_dict)
            if output_path is not None:
                output_path.mkdir(parents=True, exist_ok=True)
                with open(output_path / f"{pred_dict['frame_id']}.txt", "w") as f:
                    for name, score, box in zip(pred_dict["name"], pred_dict["score"], pred_dict["boxes_lidar"]):
                        f.write(
                            f"{name} {score:.6f} "
                            + " ".join(f"{x:.6f}" for x in box[:9])
                            + "\n"
                        )
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if not self.custom_radar_infos or "annos" not in self.custom_radar_infos[0]:
            return None, {}

        gt_annos = [copy.deepcopy(info["annos"]) for info in self.custom_radar_infos]
        result = evaluate_custom_radar(gt_annos, det_annos, class_names)
        lines = ["CustomRadar evaluation:"]
        for key in sorted(result):
            lines.append(f"{key}: {result[key]:.6f}")
        return "\n".join(lines), result


def evaluate_custom_radar(gt_annos, det_annos, class_names, iou_thresh=0.5):
    total_gt = 0
    total_pred = 0
    tp = 0
    fp = 0
    center_errors = []
    vel_errors = []
    yaw_errors = []
    all_scores = []
    all_tp = []

    for gt, det in zip(gt_annos, det_annos):
        gt_boxes = gt.get("gt_boxes_lidar", np.zeros((0, 9), dtype=np.float32))
        gt_names = gt.get("name", np.asarray([]))
        keep_gt = common_utils.keep_arrays_by_name(gt_names, class_names)
        gt_boxes = gt_boxes[keep_gt]
        pred_boxes = det.get("boxes_lidar", np.zeros((0, 9), dtype=np.float32))
        pred_names = det.get("name", np.asarray([]))
        pred_scores = det.get("score", np.zeros(len(pred_boxes), dtype=np.float32))
        keep_pred = common_utils.keep_arrays_by_name(pred_names, class_names)
        pred_boxes = pred_boxes[keep_pred]
        pred_scores = pred_scores[keep_pred]
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        matched = np.zeros(len(gt_boxes), dtype=bool)
        ious = radar_utils.boxes_iou_bev_numpy(pred_boxes[:, :7], gt_boxes[:, :7]) if len(pred_boxes) and len(gt_boxes) else np.zeros((len(pred_boxes), len(gt_boxes)))
        for i, score in enumerate(pred_scores):
            all_scores.append(float(score))
            if len(gt_boxes) == 0:
                fp += 1
                all_tp.append(0)
                continue
            j = int(np.argmax(ious[i]))
            if ious[i, j] >= iou_thresh and not matched[j]:
                matched[j] = True
                tp += 1
                all_tp.append(1)
                center_errors.append(float(np.linalg.norm(pred_boxes[i, :2] - gt_boxes[j, :2])))
                if pred_boxes.shape[1] >= 9 and gt_boxes.shape[1] >= 9:
                    vel_errors.append(float(np.linalg.norm(pred_boxes[i, 7:9] - gt_boxes[j, 7:9])))
                yaw_errors.append(float(abs(((pred_boxes[i, 6] - gt_boxes[j, 6] + np.pi) % (2 * np.pi)) - np.pi)))
            else:
                fp += 1
                all_tp.append(0)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, total_gt)
    f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)
    fp_per_frame = fp / max(1, len(gt_annos))
    ap = approximate_ap(np.asarray(all_scores), np.asarray(all_tp), total_gt)
    fn = total_gt - tp
    return {
        "bev_ap_50": ap,
        "precision_50": precision,
        "recall_50": recall,
        "f1_50": f1,
        "center_error_mean_m": _mean(center_errors),
        "center_error_median_m": _percentile(center_errors, 50),
        "center_error_p90_m": _percentile(center_errors, 90),
        "velocity_l2_error_mean_mps": _mean(vel_errors),
        "velocity_l2_error_median_mps": _percentile(vel_errors, 50),
        "velocity_l2_error_p90_mps": _percentile(vel_errors, 90),
        "yaw_error_mean_rad": _mean(yaw_errors),
        "yaw_error_median_rad": _percentile(yaw_errors, 50),
        "yaw_error_p90_rad": _percentile(yaw_errors, 90),
        "false_positives_per_frame": fp_per_frame,
        "true_positives": float(tp),
        "false_positives": float(fp),
        "false_negatives": float(fn),
        "num_gt": float(total_gt),
        "num_predictions": float(total_pred),
        "num_frames": float(len(gt_annos)),
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def approximate_ap(scores: np.ndarray, tp_flags: np.ndarray, total_gt: int) -> float:
    if len(scores) == 0 or total_gt == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_sorted = tp_flags[order].astype(np.float64)
    fp_sorted = 1.0 - tp_sorted
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recall = tp_cum / max(1, total_gt)
    precision = tp_cum / np.maximum(1.0, tp_cum + fp_cum)
    ap = 0.0
    for thresh in np.linspace(0.0, 1.0, 11):
        vals = precision[recall >= thresh]
        ap += (vals.max() if vals.size else 0.0) / 11.0
    return float(ap)
