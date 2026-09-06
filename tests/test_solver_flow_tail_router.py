"""Block-count-routed second flow prior (`FLOW_CKPT_TAIL`).

Root cause context: docs/experiments/2026-08-21-post-beta-p0-execution.md
Sec.17t -- a tail-tilted fine-tune raises direct-channel candidate supply at
n>=90 and collapses it at n=76-89, so the two priors are routed by block
count (a reusable instance statistic, never a case id).

Off path: FLOW_CKPT_TAIL unset or PARTNER_FLOW_TAIL_MIN_N<=0 -> the loader
returns before touching anything, `_flow_route` returns None on every call,
and `_sample_flow_preds` never rebinds `self.flow_model`.
"""

import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src" / "solver", ROOT / "FloorSet",
           ROOT / "FloorSet" / "iccad2026contest"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import contest_optimizer as CO  # noqa: E402


TAIL_ENV = ("FLOW_CKPT_TAIL", "PARTNER_FLOW_TAIL_MIN_N", "PARTNER_FLOW_SLOTS")


def _stub(primary="PRIMARY", tail=None, min_n=0, verbose=False):
    """A MyOptimizer shell with only the router's own state populated.

    `object.__new__` skips __init__ on purpose: the router must be testable
    without loading two 107.5M-parameter checkpoints or spawning the pool.
    """
    o = object.__new__(CO.MyOptimizer)
    o.verbose = verbose
    o.flow_model = primary
    o.flow_cfg = "PRIMARY_CFG"
    o.flow_tail_model = tail
    o.flow_tail_cfg = "TAIL_CFG" if tail is not None else None
    o.flow_tail_min_n = min_n
    return o


def _clear(monkeypatch):
    for name in TAIL_ENV:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------- off path
def test_loader_off_when_env_unset(monkeypatch):
    _clear(monkeypatch)
    o = _stub()
    o._load_flow_tail_model()
    assert o.flow_tail_model is None
    assert o.flow_tail_cfg is None
    assert o.flow_tail_min_n == 0


def test_loader_off_when_min_n_not_set_even_with_a_path(monkeypatch, tmp_path):
    _clear(monkeypatch)
    ck = tmp_path / "tail.pt"
    ck.write_bytes(b"not-a-real-checkpoint")
    monkeypatch.setenv("FLOW_CKPT_TAIL", str(ck))
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "10")
    o = _stub()
    o._load_flow_tail_model()          # MIN_N unset -> 0 -> never loads
    assert o.flow_tail_model is None
    assert o.flow_tail_min_n == 0


def test_loader_off_when_slots_zero(monkeypatch, tmp_path):
    _clear(monkeypatch)
    ck = tmp_path / "tail.pt"
    ck.write_bytes(b"x")
    monkeypatch.setenv("FLOW_CKPT_TAIL", str(ck))
    monkeypatch.setenv("PARTNER_FLOW_TAIL_MIN_N", "95")
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "0")
    o = _stub()
    o._load_flow_tail_model()
    assert o.flow_tail_model is None
    assert o.flow_tail_min_n == 0


def test_missing_tail_file_leaves_primary_only(monkeypatch, tmp_path):
    _clear(monkeypatch)
    monkeypatch.setenv("FLOW_CKPT_TAIL", str(tmp_path / "does_not_exist.pt"))
    monkeypatch.setenv("PARTNER_FLOW_TAIL_MIN_N", "95")
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "10")
    o = _stub()
    o._load_flow_tail_model()
    assert o.flow_tail_model is None
    assert o.flow_tail_min_n == 0
    assert o.flow_model == "PRIMARY"        # primary untouched


def test_corrupt_tail_checkpoint_is_contained(monkeypatch, tmp_path, capsys):
    """A file that exists but is not a checkpoint must disable the router,
    not the flow channel."""
    _clear(monkeypatch)
    ck = tmp_path / "tail.pt"
    ck.write_bytes(b"definitely not a torch checkpoint")
    monkeypatch.setenv("FLOW_CKPT_TAIL", str(ck))
    monkeypatch.setenv("PARTNER_FLOW_TAIL_MIN_N", "95")
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "10")
    o = _stub()
    o._load_flow_tail_model()               # must not raise
    assert o.flow_tail_model is None
    assert o.flow_tail_min_n == 0
    assert o.flow_model == "PRIMARY"


def test_loader_off_when_primary_flow_model_absent(monkeypatch, tmp_path):
    """No primary flow channel -> no router (the tail prior is a routing
    partner of the primary, not a standalone channel)."""
    _clear(monkeypatch)
    ck = tmp_path / "tail.pt"
    ck.write_bytes(b"x")
    monkeypatch.setenv("FLOW_CKPT_TAIL", str(ck))
    monkeypatch.setenv("PARTNER_FLOW_TAIL_MIN_N", "95")
    monkeypatch.setenv("PARTNER_FLOW_SLOTS", "10")
    o = _stub(primary=None)
    o._load_flow_tail_model()
    assert o.flow_tail_model is None


def test_route_is_none_when_router_off():
    o = _stub()                              # no tail model, min_n 0
    for n in (21, 75, 76, 95, 105, 120):
        assert o._flow_route(n) is None


# --------------------------------------------------------------- on path
@pytest.mark.parametrize("n,expect_tail", [
    (21, False), (75, False), (89, False),
    (94, False),                # MIN_N - 1
    (95, True),                 # == MIN_N
    (96, True), (110, True), (120, True),
])
def test_route_boundary_is_inclusive_at_min_n(n, expect_tail):
    o = _stub(tail="TAIL", min_n=95)
    got = o._flow_route(n)
    if expect_tail:
        assert got == ("TAIL", "TAIL_CFG")
    else:
        assert got is None


def test_route_is_not_reentrant():
    """While the tail prior is installed the router must decline, so
    `_sample_flow_preds` cannot recurse into itself."""
    o = _stub(tail="TAIL", min_n=95)
    o.flow_model = "TAIL"                   # as the rebind leaves it
    assert o._flow_route(120) is None


def test_sample_flow_preds_installs_and_restores_the_tail_model():
    """The router rebinds (flow_model, flow_cfg) for exactly one call and
    restores them afterwards, including on an exception."""
    seen = []

    o = _stub(tail="TAIL", min_n=95)
    real = CO.MyOptimizer._sample_flow_preds

    def fake(self, n, *a, **kw):
        route = self._flow_route(n)
        if route is not None:
            return real(self, n, *a, **kw)      # exercises the rebind block
        seen.append((n, self.flow_model, self.flow_cfg))
        return ["PRED"]

    o._sample_flow_preds = types.MethodType(fake, o)
    # below MIN_N -> primary, no rebind
    assert o._sample_flow_preds(80, None, None, None, None, None, None, 3) \
        == ["PRED"]
    # at/above MIN_N -> tail installed for the inner call
    assert o._sample_flow_preds(110, None, None, None, None, None, None, 3) \
        == ["PRED"]
    assert seen == [(80, "PRIMARY", "PRIMARY_CFG"),
                    (110, "TAIL", "TAIL_CFG")]
    # restored
    assert (o.flow_model, o.flow_cfg) == ("PRIMARY", "PRIMARY_CFG")


def test_rebind_is_restored_when_the_inner_call_raises():
    o = _stub(tail="TAIL", min_n=95)
    real = CO.MyOptimizer._sample_flow_preds

    def fake(self, n, *a, **kw):
        if self._flow_route(n) is not None:
            return real(self, n, *a, **kw)
        raise RuntimeError("sampler blew up")

    o._sample_flow_preds = types.MethodType(fake, o)
    with pytest.raises(RuntimeError):
        o._sample_flow_preds(110, None, None, None, None, None, None, 3)
    assert (o.flow_model, o.flow_cfg) == ("PRIMARY", "PRIMARY_CFG")
