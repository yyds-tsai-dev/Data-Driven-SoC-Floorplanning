"""One-step split/merge large neighborhood for the partner column solver.

The production SA relocates one unit at a time.  This module evaluates a
completed repartition directly: split one vertical unit list into two adjacent
columns, or merge two adjacent columns into one.  It is deliberately unaware
of geometry; `_ColumnOptimizer._evaluate` remains the single layout and cost
authority.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence


Columns = list[list[int]]


@dataclass(frozen=True)
class ColumnLNSResult:
    cols: Columns
    base_cost: float
    best_cost: float
    candidate_count: int
    elapsed_s: float

    @property
    def improved(self) -> bool:
        return self.best_cost < self.base_cost - 1e-9


def normalize_columns(
    cols: Sequence[Sequence[int]],
    *,
    min_columns: int = 2,
) -> Columns:
    """Copy `cols`, discard zero-width empty slots, and retain a legal floor."""
    out = [list(col) for col in cols if col]
    while len(out) < min_columns:
        out.append([])
    return out


def _key(cols: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(col) for col in cols)


def _forced_edges_hold(
    cols: Sequence[Sequence[int]],
    forces: Sequence[Optional[str]],
) -> bool:
    last = len(cols) - 1
    for ci, col in enumerate(cols):
        for unit in col:
            force = forces[unit] if unit < len(forces) else None
            if force == "L" and ci != 0:
                return False
            if force == "R" and ci != last:
                return False
    return True


def iter_split_merge_neighbors(
    cols: Sequence[Sequence[int]],
    forces: Sequence[Optional[str]],
    *,
    min_columns: int = 2,
    max_columns: int = 18,
) -> Iterator[tuple[str, Columns]]:
    """Yield unique one-step neighbors without mutating `cols`.

    Only active (non-empty) columns define the representation.  Empty slots
    have zero width in the production packer and are therefore canonicalized
    away before the neighborhood is built.
    """
    base = [list(col) for col in cols if col]
    seen: set[tuple[tuple[int, ...], ...]] = set()

    def emit(kind: str, cand: Columns):
        if not (min_columns <= len(cand) <= max_columns):
            return None
        if any(not col for col in cand):
            return None
        if not _forced_edges_hold(cand, forces):
            return None
        key = _key(cand)
        if key == _key(base) or key in seen:
            return None
        seen.add(key)
        return kind, cand

    if len(base) < max_columns:
        for ci, col in enumerate(base):
            for cut in range(1, len(col)):
                left, right = col[:cut], col[cut:]
                for first, second in ((left, right), (right, left)):
                    cand = [list(c) for c in base[:ci]]
                    cand.extend((list(first), list(second)))
                    cand.extend(list(c) for c in base[ci + 1:])
                    item = emit("split", cand)
                    if item is not None:
                        yield item

    if len(base) > min_columns:
        for ci in range(len(base) - 1):
            left, right = base[ci], base[ci + 1]
            for merged in (left + right, right + left):
                cand = [list(c) for c in base[:ci]]
                cand.append(list(merged))
                cand.extend(list(c) for c in base[ci + 2:])
                item = emit("merge", cand)
                if item is not None:
                    yield item


def best_split_merge_neighbor(
    optimizer,
    cols: Sequence[Sequence[int]],
    forces: Sequence[Optional[str]],
    *,
    min_columns: int = 2,
    max_columns: int = 18,
) -> ColumnLNSResult:
    """Return the strict best one-step neighbor under the optimizer proxy."""
    start = time.perf_counter()
    base = normalize_columns(cols, min_columns=min_columns)
    base_cost, _ = optimizer._evaluate(base)
    best_cost = float(base_cost)
    best_cols = [list(col) for col in base]
    count = 0

    for _kind, cand in iter_split_merge_neighbors(
        base,
        forces,
        min_columns=min_columns,
        max_columns=max_columns,
    ):
        cost, _ = optimizer._evaluate(cand)
        count += 1
        if cost < best_cost - 1e-9:
            best_cost = float(cost)
            best_cols = [list(col) for col in cand]

    return ColumnLNSResult(
        cols=best_cols,
        base_cost=float(base_cost),
        best_cost=best_cost,
        candidate_count=count,
        elapsed_s=time.perf_counter() - start,
    )
