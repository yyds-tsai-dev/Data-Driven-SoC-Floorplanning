"""`PARTNER_FLOW_WARM=1` -> pay Flow's first-call CUDA/JIT cost inside
`__init__` (via `_warm_flow_sampler`) instead of the first real `solve()`
case.  Ported from the handover build.

What must hold:
  * flag off (default) -> `_warm_flow_sampler` is a no-op: `flow_warm_succeeded`
    stays False and `_sample_flow_preds` is never invoked, regardless of
    whether a flow model is loaded.
  * flag on, but no flow model / no flow slots -> still a no-op (guarded).
  * flag on, flow model present, PARTNER_FLOW_SLOTS>0 -> `_sample_flow_preds`
    IS invoked with the synthetic warm-up instance, and
    `flow_warm_succeeded` is set True on success.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partner"))

import contest_optimizer as co  # noqa: E402

_ENV = ("PARTNER_FLOW_WARM", "PARTNER_FLOW_WARM_N", "PARTNER_FLOW_SLOTS",
        "PARTNER_NREF")


def _clean(monkeypatch):
    for v in _ENV:
        monkeypatch.delenv(v, raising=False)


def _opt():
    opt = co.MyOptimizer.__new__(co.MyOptimizer)
    opt.verbose = False
    opt.flow_model = None
    return opt


def test_flag_off_is_a_noop_even_with_a_flow_model(monkeypatch):
    _clean(monkeypatch)
    opt = _opt()
    opt.flow_model = object()          # truthy stand-in

    called = {}

    def spy(*a, **k):
        called["hit"] = True
        raise AssertionError("should not be called")

    opt._sample_flow_preds = spy
    opt._warm_flow_sampler()
    assert opt.flow_warm_succeeded is False
    assert "hit" not in called


def test_flag_on_without_flow_model_is_a_noop(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PARTNER_FLOW_WARM", "1")
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "6")
    opt = _opt()
    assert opt.flow_model is None
    opt._sample_flow_preds = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not be called"))
    opt._warm_flow_sampler()
    assert opt.flow_warm_succeeded is False


def test_flag_on_without_flow_slots_is_a_noop(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PARTNER_FLOW_WARM", "1")
    opt = _opt()
    opt.flow_model = object()
    opt._sample_flow_preds = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("should not be called"))
    opt._warm_flow_sampler()          # PARTNER_FLOW_SLOTS unset -> default 0
    assert opt.flow_warm_succeeded is False


def test_flag_on_with_model_and_slots_invokes_sampler(monkeypatch):
    _clean(monkeypatch)
    monkeypatch.setenv("PARTNER_FLOW_WARM", "1")
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "6")
    monkeypatch.setenv("PARTNER_FLOW_WARM_N", "21")
    opt = _opt()
    opt.flow_model = object()

    captured = {}

    def spy(n, at, cons, tpos, b2b, p2b, pins, samples):
        captured["n"] = n
        captured["samples"] = samples
        return [object()] * samples

    opt._sample_flow_preds = spy
    opt._warm_flow_sampler()
    assert captured.get("n") == 21
    assert opt.flow_warm_succeeded is True
