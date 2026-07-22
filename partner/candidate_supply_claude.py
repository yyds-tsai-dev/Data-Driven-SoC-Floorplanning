from __future__ import annotations

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
