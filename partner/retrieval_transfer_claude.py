"""Coherent geometry transfer from a retrieved floorplan."""

from __future__ import annotations

import numpy as np


_D4_TRANSFORMS = {"identity", "mirror_x", "mirror_y", "transpose"}


def apply_d4_rectangles(rectangles, transform):
    """Apply a supported transform to every rectangle in one layout."""
    result = np.asarray(rectangles, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] != 4 or len(result) == 0:
        raise ValueError("rectangles must have shape [N, 4] with N > 0")
    if not np.isfinite(result).all():
        raise ValueError("rectangles must be finite")
    if transform not in _D4_TRANSFORMS:
        raise ValueError(f"unsupported D4 transform: {transform}")
    if transform == "identity":
        return result

    x0 = result[:, 0].min()
    y0 = result[:, 1].min()
    x1 = (result[:, 0] + result[:, 2]).max()
    y1 = (result[:, 1] + result[:, 3]).max()
    if transform == "mirror_x":
        result[:, 0] = x0 + x1 - (result[:, 0] + result[:, 2])
    elif transform == "mirror_y":
        result[:, 1] = y0 + y1 - (result[:, 1] + result[:, 3])
    elif transform == "transpose":
        result = result[:, [1, 0, 3, 2]]
    return result


def remap_boundary_node_features(node_features, transform):
    """Map left/right/top/bottom features consistently with a layout transform."""
    result = np.asarray(node_features, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[1] <= 12:
        raise ValueError("node features must contain boundary columns 9 through 12")
    if not np.isfinite(result).all():
        raise ValueError("node features must be finite")
    if transform not in _D4_TRANSFORMS:
        raise ValueError(f"unsupported D4 transform: {transform}")
    left, right, top, bottom = [result[:, column].copy() for column in (9, 10, 11, 12)]
    if transform == "mirror_x":
        result[:, 9], result[:, 10] = right, left
    elif transform == "mirror_y":
        result[:, 11], result[:, 12] = bottom, top
    elif transform == "transpose":
        result[:, 9], result[:, 10], result[:, 11], result[:, 12] = bottom, top, right, left
    return result


def transfer_layout(
    source_fp_xywh, target_to_source, target_area, constraints, target_positions, transform
) -> np.ndarray:
    """Transfer one complete source layout into target block order and scale."""
    source_fp = np.asarray(source_fp_xywh, dtype=np.float64)
    target_area = np.asarray(target_area, dtype=np.float64)
    constraints = np.asarray(constraints, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    target_to_source = np.asarray(target_to_source)
    n = len(target_area)
    if source_fp.ndim != 2 or source_fp.shape[1] != 4 or not np.isfinite(source_fp).all():
        raise ValueError("source floorplan must be finite [N, 4] rectangles")
    if source_fp.shape[0] != n:
        raise ValueError("first retrieval version requires equal source and target block counts")
    if target_to_source.shape != (n,) or not np.issubdtype(target_to_source.dtype, np.integer):
        raise ValueError("target-to-source assignment must be an integer vector of target length")
    if np.any(target_to_source < 0) or np.any(target_to_source >= len(source_fp)):
        raise ValueError("target-to-source assignment contains an invalid source index")
    if np.unique(target_to_source).size != n:
        raise ValueError("target-to-source assignment must reorder every source block exactly once")
    if constraints.ndim != 2 or constraints.shape[0] != n or constraints.shape[1] < 2:
        raise ValueError("constraints must include fixed and preplaced flags for every target")
    if target_positions.shape != (n, 4):
        raise ValueError("target positions must have shape [N, 4]")
    if not np.isfinite(target_area).all() or np.any(target_area <= 0):
        raise ValueError("target areas must be finite and positive")
    if not np.isfinite(constraints).all() or not np.isfinite(target_positions).all():
        raise ValueError("target constraints and positions must be finite")

    source = apply_d4_rectangles(source_fp[target_to_source], transform)
    if np.any(source[:, 2:] <= 0):
        raise ValueError("source rectangles must have positive sizes")
    source_scale = np.sqrt(np.maximum((source[:, 2] * source[:, 3]).sum(), 1.0))
    xy = source[:, :2] / source_scale
    log_aspect = np.clip(np.log(source[:, 2] / np.maximum(source[:, 3], 1e-9)), -3.0, 3.0)
    target_scale = np.sqrt(np.maximum(target_area.sum(), 1.0))
    aspect = np.exp(log_aspect)
    width = np.sqrt(target_area * aspect)
    height = np.sqrt(target_area / aspect)
    result = np.column_stack([xy * target_scale, width, height])

    fixed = constraints[:, 0] != 0
    preplaced = constraints[:, 1] != 0
    has_wh = (target_positions[:, 2] > 0) & (target_positions[:, 3] > 0)
    shape_known = (fixed | preplaced) & has_wh
    result[shape_known, 2:4] = target_positions[shape_known, 2:4]
    has_xy = preplaced & (target_positions[:, 0] >= 0) & (target_positions[:, 1] >= 0)
    result[has_xy, :2] = target_positions[has_xy, :2]
    if not np.isfinite(result).all() or np.any(result[:, 2:] <= 0):
        raise ValueError("transferred rectangles must be finite and positive")
    return result
