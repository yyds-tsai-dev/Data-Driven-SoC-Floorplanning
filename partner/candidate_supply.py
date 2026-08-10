from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateBatch:
    source: str
    predictions: list[np.ndarray]
    generation_s: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("candidate source must be non-empty")
        if self.generation_s < 0:
            raise ValueError("generation_s must be non-negative")
        for prediction in self.predictions:
            if prediction.ndim != 2 or prediction.shape[1] != 4:
                raise ValueError("candidate prediction must have shape [N,4]")
            if not np.isfinite(prediction).all():
                raise ValueError("candidate prediction must be finite")


def allocate_quotas(
    total: int,
    requested: Mapping[str, int],
    enabled: Sequence[str],
) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    names = tuple(name for name in enabled if requested.get(name, 0) > 0)
    if not names or total == 0:
        return {}
    weights = {name: int(requested[name]) for name in names}
    weight_sum = sum(weights.values())
    raw = {name: total * weights[name] / weight_sum for name in names}
    result = {name: int(raw[name]) for name in names}
    left = total - sum(result.values())
    order = sorted(names, key=lambda name: (-(raw[name] - result[name]), names.index(name)))
    for name in order[:left]:
        result[name] += 1
    return result


def _hpwl_proxy(prediction: np.ndarray, b2b: np.ndarray) -> float:
    n = prediction.shape[0]
    centers_x = prediction[:, 0] + 0.5 * prediction[:, 2]
    centers_y = prediction[:, 1] + 0.5 * prediction[:, 3]
    i, j = np.nonzero(np.triu(b2b[:n, :n], 1))
    if len(i) == 0:
        return 0.0
    return float((b2b[i, j] * (
        np.abs(centers_x[i] - centers_x[j])
        + np.abs(centers_y[i] - centers_y[j])
    )).sum())


def _overlap_fraction(prediction: np.ndarray, area_targets: np.ndarray) -> float:
    n = prediction.shape[0]
    x0, y0 = prediction[:, 0], prediction[:, 1]
    x1 = x0 + prediction[:, 2]
    y1 = y0 + prediction[:, 3]
    overlap_x = np.maximum(
        0.0, np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    )
    overlap_y = np.maximum(
        0.0, np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    )
    overlap = overlap_x * overlap_y
    overlap[np.diag_indices(n)] = 0.0
    return float(overlap.sum()) / (2.0 * max(float(area_targets[:n].sum()), 1e-9))


def _bbox_area(prediction: np.ndarray) -> float:
    x0, y0 = prediction[:, 0], prediction[:, 1]
    x1 = x0 + prediction[:, 2]
    y1 = y0 + prediction[:, 3]
    return float((x1.max() - x0.min()) * (y1.max() - y0.min()))


def rank_predictions(
    predictions: Sequence[np.ndarray],
    area_targets: np.ndarray,
    b2b: np.ndarray,
    constraint_penalties: Sequence[float] | None = None,
    violation_weight: float = 0.0,
    align_ref: tuple[float, float] | None = None,
) -> list[int]:
    """Rank raw model predictions for the prescreen.

    `align_ref=(area_ref, n_soft)` selects the PARTNER_PROXY_ALIGN form: the
    official cost's own functional shape,

        (1 + 0.5 * (hpwl_gap + area_gap + 5 * overlap)) * exp(2 * V / n_soft)

    instead of the historical additive `hpwl/hpwl_ref + 5*overlap + w*pen`.
    Two things change, both of them alignment defects of the old form:

      * V is repriced.  In the official cost a unit of relative violation is
        worth FOUR gap units (`exp(2 .)` vs `alpha = 0.5`); the additive form
        priced `violation_weight` (0.5) against a hpwl coefficient of 1.0,
        i.e. HALF a gap unit -- 8x too cheap, and in units that were not
        violation counts at all.  `constraint_penalties` must therefore be
        supplied in ESTIMATED VIOLATION COUNTS when `align_ref` is set.
      * `area_gap` enters at all.  The official cost weights the bbox gap
        exactly like the hpwl gap and the additive form omitted it entirely.

    `align_ref=None` (the default, and every pre-flag caller) is byte-for-byte
    the historical ranking."""
    if not predictions:
        return []
    hpwl = [_hpwl_proxy(prediction, b2b) for prediction in predictions]
    hpwl_ref = max(min(hpwl), 1e-9)
    if constraint_penalties is None:
        penalties = [0.0] * len(predictions)
    else:
        penalties = constraint_penalties
    if len(penalties) != len(predictions):
        raise ValueError("constraint penalty count must match predictions")
    if align_ref is not None:
        area_ref, n_soft = align_ref
        area_ref = max(float(area_ref), 1e-9)
        n_soft = max(float(n_soft), 1.0)
        score = [
            (1.0 + 0.5 * (hpwl[k] / hpwl_ref - 1.0
                          + max(0.0, _bbox_area(predictions[k]) / area_ref - 1.0)
                          + 5.0 * _overlap_fraction(predictions[k], area_targets)))
            * math.exp(2.0 * max(0.0, float(penalties[k])) / n_soft)
            for k in range(len(predictions))
        ]
        return sorted(range(len(predictions)), key=lambda k: (score[k], k))
    score = [
        hpwl[k] / hpwl_ref
        + 5.0 * _overlap_fraction(predictions[k], area_targets)
        + violation_weight * float(penalties[k])
        for k in range(len(predictions))
    ]
    return sorted(range(len(predictions)), key=lambda k: (score[k], k))
