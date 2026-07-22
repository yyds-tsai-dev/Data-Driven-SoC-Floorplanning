from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RetrievalFeatures:
    global_vector: np.ndarray
    node_matrix: np.ndarray
    block_count: int

    def __post_init__(self) -> None:
        if self.global_vector.shape != (24,):
            raise ValueError("global retrieval feature must have shape [24]")
        if self.node_matrix.shape != (self.block_count, 16):
            raise ValueError("node retrieval features must have shape [N,16]")
        if not np.isfinite(self.global_vector).all() or not np.isfinite(self.node_matrix).all():
            raise ValueError("retrieval features must be finite")


def _summary(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(values.mean()),
        float(values.std()),
        float(np.quantile(values, 0.25)),
        float(np.quantile(values, 0.75)),
    ]


def extract_retrieval_features(
    area, b2b, p2b, pins, constraints, target_positions
) -> RetrievalFeatures:
    area = np.asarray(area, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    n = len(area)
    degree = np.zeros(n, dtype=np.float64)
    weighted_degree = np.zeros(n, dtype=np.float64)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for edge in np.asarray(b2b):
        if edge[0] < 0:
            continue
        i, j, weight = int(edge[0]), int(edge[1]), float(edge[2])
        degree[[i, j]] += 1.0
        weighted_degree[[i, j]] += max(weight, 0.0)
        adjacency[i, j] += max(weight, 0.0)
        adjacency[j, i] += max(weight, 0.0)
    pin_degree = np.zeros(n, dtype=np.float64)
    for edge in np.asarray(p2b):
        if edge[0] >= 0 and 0 <= int(edge[1]) < n:
            pin_degree[int(edge[1])] += 1.0
    safe_area = np.maximum(area, 1e-9)
    area_share = safe_area / safe_area.sum()
    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    mib = constraints[:, 2] if constraints.shape[1] > 2 else np.zeros(n)
    cluster = constraints[:, 3] if constraints.shape[1] > 3 else np.zeros(n)
    boundary = constraints[:, 4].astype(int) if constraints.shape[1] > 4 else np.zeros(n, dtype=int)
    neighbor_denominator = np.maximum((adjacency > 0).sum(axis=1), 1.0)
    neighbor_log_area = (adjacency > 0) @ np.log(safe_area) / neighbor_denominator
    neighbor_degree = (adjacency > 0) @ degree / neighbor_denominator
    global_values = (
        [float(n), float(np.log1p(safe_area.sum()))]
        + _summary(np.log(safe_area))
        + _summary(degree)
        + _summary(weighted_degree)
        + _summary(pin_degree)
        + [
            float(fixed.mean()),
            float(preplaced.mean()),
            float((mib > 0).mean()),
            float((cluster > 0).mean()),
            float((boundary > 0).mean()),
            float(np.unique(cluster[cluster > 0]).size),
        ]
    )
    global_vector = np.asarray(global_values[:24], dtype=np.float32)
    node_matrix = np.stack(
        [
            np.log(safe_area),
            area_share,
            degree,
            weighted_degree,
            pin_degree,
            fixed,
            preplaced,
            mib > 0,
            cluster > 0,
            (boundary & 1) != 0,
            (boundary & 2) != 0,
            (boundary & 4) != 0,
            (boundary & 8) != 0,
            np.asarray(target_positions)[:, 0] >= 0,
            neighbor_log_area,
            neighbor_degree,
        ],
        axis=1,
    ).astype(np.float32)
    return RetrievalFeatures(global_vector, node_matrix, n)
