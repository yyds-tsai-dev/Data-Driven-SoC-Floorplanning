"""Unit coverage for the final post-pass feature router and stage receipt."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "partner"))

import postpass_router as pr  # noqa: E402
import stage_receipt as sr  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in ("PARTNER_ROUTE_MIN_N", "PARTNER_ROUTE_MAX_N",
                "PARTNER_ROUTE_RESERVE_FRAC", "PARTNER_ROUTE_RESERVE_MAX",
                "PARTNER_ROUTE_RESERVE_MIN", "PARTNER_ROUTE_MIN_SLICE",
                "PARTNER_ROUTE_BOX_TOPOLOGY", "PARTNER_ROUTE_BOX_VKILL",
                "PARTNER_ROUTE_BOX_POLISH", "PARTNER_ROUTE_CARVE",
                "PARTNER_ROUTE_HEADROOM_S",
                "PARTNER_ROUTE_VKILL_ALWAYS", "PARTNER_ROUTE_POLISH_ALWAYS",
                "PARTNER_STAGE_RECEIPT"):
        monkeypatch.delenv(key, raising=False)
    assert "PARTNER_STAGE_RECEIPT" not in os.environ


# ---------------------------------------------------------------------------
# value band
# ---------------------------------------------------------------------------

def test_out_of_band_reserves_nothing_and_allows_nothing():
    r = pr.PostPassRouter(80, 0.5)
    assert r.in_band is False
    assert r.reserve == 0.0
    for stage in (pr.TOPOLOGY, pr.VKILL, pr.POLISH):
        ok, reason = r.allow(stage)
        assert ok is False
        assert reason == "out_of_value_band"
        assert r.budget_for(stage) == 0.0


def test_in_band_reserves_a_fraction_of_the_case_budget():
    small = pr.PostPassRouter(95, 0.2335)
    large = pr.PostPassRouter(120, 1.22)
    # A fixed reserve would tax the low-weight small case hardest; a reserve
    # tied to the case budget does not.  This ordering is the design's point.
    assert small.reserve < large.reserve
    assert small.reserve == pytest.approx(min(0.28, max(0.06, 0.35 * 0.2335)))
    assert large.reserve == pytest.approx(min(0.28, max(0.06, 0.35 * 1.22)))


def test_reserve_is_capped_and_floored(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_RESERVE_MAX", "0.10")
    monkeypatch.setenv("PARTNER_ROUTE_RESERVE_MIN", "0.03")
    assert pr.PostPassRouter(120, 100.0).reserve == pytest.approx(0.10)
    assert pr.PostPassRouter(120, 0.0).reserve == pytest.approx(0.03)


def test_band_edges_are_configurable(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_MIN_N", "100")
    monkeypatch.setenv("PARTNER_ROUTE_MAX_N", "110")
    assert pr.PostPassRouter(99, 1.0).in_band is False
    assert pr.PostPassRouter(100, 1.0).in_band is True
    assert pr.PostPassRouter(110, 1.0).in_band is True
    assert pr.PostPassRouter(111, 1.0).in_band is False


# ---------------------------------------------------------------------------
# reserve accounting — the property that keeps a carve from becoming an append
# ---------------------------------------------------------------------------

def test_spending_the_reserve_closes_later_stages(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_BOX_TOPOLOGY", "0.06")
    r = pr.PostPassRouter(120, 1.22)
    assert r.allow(pr.TOPOLOGY)[0] is True
    r.spend(r.reserve)
    assert r.remaining == 0.0
    for stage in (pr.TOPOLOGY, pr.VKILL):
        ok, reason = r.allow(stage, soft_violations=5, layout_dirty=True)
        assert ok is False
        assert reason == "reserve_exhausted"


def test_carve_is_off_by_default(monkeypatch):
    # Appending makes the runtime cost exactly measurable from the receipt;
    # carving hides it inside an SA whose full-run spread (~0.03 on Shadow v3)
    # is larger than the effect under test.
    r = pr.PostPassRouter(120, 1.22)
    assert r.carve is False
    assert r.sa_reserve == 0.0
    monkeypatch.setenv("PARTNER_ROUTE_CARVE", "1")
    r2 = pr.PostPassRouter(120, 1.22)
    assert r2.carve is True
    assert r2.sa_reserve == pytest.approx(r2.reserve)


def test_disabled_stage_is_reported_as_such(monkeypatch):
    # The second polish ships disabled: a same-call receipt recorded it
    # running 16 times, spending 1.0 s, and changing zero layouts.
    r = pr.PostPassRouter(120, 1.22)
    assert r.allow(pr.POLISH, layout_dirty=True) == (False, "stage_disabled")
    monkeypatch.setenv("PARTNER_ROUTE_BOX_POLISH", "0.1")
    assert pr.PostPassRouter(120, 1.22).allow(
        pr.POLISH, layout_dirty=True)[0] is True


def test_stage_box_is_a_cap_not_a_share_of_the_reserve(monkeypatch):
    # A stage that finishes early must NOT hand its slack to a later stage:
    # the first receipt caught exactly that, with the second polish absorbing
    # the topology pass's leftovers and spending them for no change at all.
    monkeypatch.setenv("PARTNER_ROUTE_BOX_TOPOLOGY", "0.06")
    r = pr.PostPassRouter(120, 1.22)
    assert r.reserve > 0.06 + 0.20
    assert r.budget_for(pr.TOPOLOGY) == pytest.approx(0.06)
    assert r.budget_for(pr.VKILL) == pytest.approx(0.20)
    r.spend(0.001)
    assert r.budget_for(pr.TOPOLOGY) == pytest.approx(0.06)


def test_reserve_still_caps_the_whole_post_pass_tail():
    r = pr.PostPassRouter(120, 1.22)
    r.spend(r.reserve - 0.01)
    assert r.budget_for(pr.VKILL) == pytest.approx(0.01)


def test_stage_budget_never_exceeds_remaining_reserve():
    r = pr.PostPassRouter(120, 1.22)
    r.spend(r.reserve * 0.9)
    for stage in (pr.TOPOLOGY, pr.VKILL, pr.POLISH):
        assert r.budget_for(stage) <= r.remaining + 1e-12


def test_overspend_cannot_drive_the_reserve_negative():
    r = pr.PostPassRouter(120, 1.22)
    r.spend(10.0)
    assert r.remaining == 0.0


# ---------------------------------------------------------------------------
# stage preconditions
# ---------------------------------------------------------------------------

def test_vkill_skipped_without_soft_violations():
    r = pr.PostPassRouter(120, 1.22)
    ok, reason = r.allow(pr.VKILL, soft_violations=0)
    assert (ok, reason) == (False, "no_soft_violations")
    assert r.allow(pr.VKILL, soft_violations=1)[0] is True
    # Unknown violation count must not suppress the pass.
    assert r.allow(pr.VKILL, soft_violations=None)[0] is True


def test_vkill_precondition_can_be_overridden(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_VKILL_ALWAYS", "1")
    r = pr.PostPassRouter(120, 1.22)
    assert r.allow(pr.VKILL, soft_violations=0)[0] is True


def test_second_polish_skipped_when_layout_unchanged_and_first_converged(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_BOX_POLISH", "0.1")
    r = pr.PostPassRouter(120, 1.22)
    ok, reason = r.allow(pr.POLISH, layout_dirty=False,
                         first_polish_truncated=False)
    assert (ok, reason) == (False, "layout_unchanged_since_first_polish")


def test_second_polish_runs_when_layout_changed_or_first_truncated(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_BOX_POLISH", "0.1")
    r = pr.PostPassRouter(120, 1.22)
    assert r.allow(pr.POLISH, layout_dirty=True)[0] is True
    assert r.allow(pr.POLISH, layout_dirty=False,
                   first_polish_truncated=True)[0] is True
    # Unknown dirtiness must not suppress the pass.
    assert r.allow(pr.POLISH, layout_dirty=None)[0] is True


def test_topology_has_no_content_precondition(monkeypatch):
    monkeypatch.setenv("PARTNER_ROUTE_BOX_TOPOLOGY", "0.06")
    r = pr.PostPassRouter(120, 1.22)
    assert r.allow(pr.TOPOLOGY, soft_violations=0, layout_dirty=False)[0] is True


def test_only_violation_repair_is_on_by_default():
    # Shipped policy, from three exactly-paired same-call reps on Shadow v3:
    # violation repair nets -0.00493, discrete topology nets -0.00024 (a wash),
    # the second polish changed zero layouts and nets +0.00460.
    r = pr.PostPassRouter(120, 1.22)
    assert r.allow(pr.VKILL, soft_violations=3)[0] is True
    assert r.allow(pr.TOPOLOGY)[1] == "stage_disabled"
    assert r.allow(pr.POLISH, layout_dirty=True)[1] == "stage_disabled"


def test_runtime_headroom_gate_closes_stages_past_the_flat_region():
    # max(0.7, R^0.3) is flat for R <= 0.3006, so time spent while the case is
    # still inside that region is exactly free; past it every millisecond is
    # charged and no measured post-pass gain covers the charge.
    import time
    r = pr.PostPassRouter(120, 1.22, start=time.time() - 5.0)
    ok, reason = r.allow(pr.VKILL, soft_violations=3)
    assert (ok, reason) == (False, "no_runtime_headroom")
    fresh = pr.PostPassRouter(120, 1.22, start=time.time())
    assert fresh.allow(pr.VKILL, soft_violations=3)[0] is True


def test_headroom_gate_can_be_disabled(monkeypatch):
    import time
    monkeypatch.setenv("PARTNER_ROUTE_HEADROOM_S", "0")
    r = pr.PostPassRouter(120, 1.22, start=time.time() - 5.0)
    assert r.allow(pr.VKILL, soft_violations=3)[0] is True


def test_headroom_gate_does_not_apply_when_carving(monkeypatch):
    # A carving router is time-neutral, so it never faces the runtime trade.
    import time
    monkeypatch.setenv("PARTNER_ROUTE_CARVE", "1")
    r = pr.PostPassRouter(120, 1.22, start=time.time() - 5.0)
    assert r.allow(pr.VKILL, soft_violations=3)[0] is True


def test_case_weight_share_matches_the_evaluator_weighting():
    import math
    assert pr.case_weight_share(120) == pytest.approx(1.0)
    assert pr.case_weight_share(108) == pytest.approx(math.exp(-1.0))
    assert pr.case_weight_share(95) < pr.case_weight_share(120)


# ---------------------------------------------------------------------------
# stage receipt
# ---------------------------------------------------------------------------

def test_receipt_is_inert_when_unarmed(monkeypatch):
    monkeypatch.delenv("PARTNER_STAGE_RECEIPT", raising=False)
    assert sr.enabled() is False
    assert sr.open_case(0, 100) is None


def test_receipt_records_paired_before_after(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTNER_STAGE_RECEIPT", str(tmp_path / "r.json"))
    rec = sr.open_case(7, 100)
    assert rec is not None
    before = [(0.0, 0.0, 1.0, 1.0)]
    after = [(0.0, 0.0, 2.0, 0.5)]
    rec.stage("final_topology", before, after, 0.01, True, "ok")
    rec.stage("final_vkill", after, after, 0.0, False, "no_soft_violations")
    assert rec.stages[0]["changed"] is True
    assert rec.stages[1]["changed"] is False
    assert rec.stages[0]["before"] == [[0.0, 0.0, 1.0, 1.0]]
    assert rec.stages[1]["ran"] is False


def test_receipt_flush_round_trips(monkeypatch, tmp_path):
    path = tmp_path / "r.json"
    monkeypatch.setenv("PARTNER_STAGE_RECEIPT", str(path))
    sr._RECORDS.clear()
    rec = sr.open_case(0, 21)
    rec.stage("final_topology", [(0.0, 0.0, 1.0, 1.0)],
              [(0.0, 0.0, 1.0, 1.0)], 0.0, False, "out_of_value_band")
    rec.close()
    sr._flush()
    data = json.loads(path.read_text())
    assert len(data["records"]) == 1
    assert data["records"][0]["block_count"] == 21
    sr._RECORDS.clear()


def test_receipt_never_raises_on_bad_layouts(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTNER_STAGE_RECEIPT", str(tmp_path / "r.json"))
    rec = sr.open_case(0, 21)
    rec.stage("final_topology", [(0.0, "nope", 1.0, 1.0)], [], 0.0, True, "ok")
    rec.note(anything=object())
    assert isinstance(rec.stages, list)
    sr._RECORDS.clear()
