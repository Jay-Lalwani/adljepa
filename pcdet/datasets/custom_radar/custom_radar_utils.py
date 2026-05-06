from __future__ import annotations

from pathlib import Path

import numpy as np


def read_label(label_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read labels as [x,y,z,dx,dy,dz,heading,vx,vy,class_name]."""
    boxes, names = [], []
    if not label_path.exists():
        return np.zeros((0, 9), dtype=np.float32), np.asarray([], dtype=str)
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 10:
            raise ValueError(f"{label_path}: expected 10 fields, got {len(parts)} in {line!r}")
        boxes.append([float(x) for x in parts[:9]])
        names.append(parts[9])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 9), np.asarray(names)


def boxes_to_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """Return oriented BEV corners in counter-clockwise order for polygon clipping."""
    if len(boxes) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    centers = boxes[:, :2]
    dx = boxes[:, 3] * 0.5
    dy = boxes[:, 4] * 0.5
    local = np.stack(
        [
            np.stack([dx, -dy], axis=1),
            np.stack([dx, dy], axis=1),
            np.stack([-dx, dy], axis=1),
            np.stack([-dx, -dy], axis=1),
        ],
        axis=1,
    )
    c = np.cos(boxes[:, 6])
    s = np.sin(boxes[:, 6])
    rot = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)], axis=1)
    return local @ np.transpose(rot, (0, 2, 1)) + centers[:, None, :]


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _inside(p: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) >= (
        edge_end[1] - edge_start[1]
    ) * (p[0] - edge_start[0])


def _intersection(s: np.ndarray, e: np.ndarray, cp1: np.ndarray, cp2: np.ndarray) -> np.ndarray:
    dc = cp1 - cp2
    dp = s - e
    n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-8:
        return e
    return np.array([(n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom])


def convex_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject.copy()
    cp1 = clip[-1]
    for cp2 in clip:
        input_list = output
        output = []
        if len(input_list) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        s = input_list[-1]
        for e in input_list:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    output.append(_intersection(s, e, cp1, cp2))
                output.append(e)
            elif _inside(s, cp1, cp2):
                output.append(_intersection(s, e, cp1, cp2))
            s = e
        output = np.asarray(output, dtype=np.float32)
        cp1 = cp2
    return output


def boxes_iou_bev_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    corners_a = boxes_to_corners_bev(a)
    corners_b = boxes_to_corners_bev(b)
    out = np.zeros((len(a), len(b)), dtype=np.float32)
    areas_a = np.asarray([polygon_area(p) for p in corners_a], dtype=np.float32)
    areas_b = np.asarray([polygon_area(p) for p in corners_b], dtype=np.float32)
    for i, pa in enumerate(corners_a):
        for j, pb in enumerate(corners_b):
            inter = polygon_area(convex_intersection(pa, pb))
            union = areas_a[i] + areas_b[j] - inter
            out[i, j] = 0.0 if union <= 0 else inter / union
    return out
