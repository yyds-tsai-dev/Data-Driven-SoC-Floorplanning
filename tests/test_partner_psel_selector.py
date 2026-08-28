"""Pool-arbitration calibration knobs (`column_sa_legalizer.psel_*`).

Root cause context: docs/experiments/2026-08-21-post-beta-p0-execution.md
Sec.17r.  The pool selector (`_parallel_solve`) was suspected of letting a
bad flow candidate displace the column champion; the PSEL dump shows the
opposite (the selector's in-pool regret is +0.00009 weighted on the
official tail) and that the dead zone is the only calibration defect it
still carries.  These tests pin the OFF path bit-exact and cover each knob.
"""

import math

import column_sa_legalizer as lg


PSEL_ENV = ("PARTNER_PSEL_FIX", "PARTNER_PSEL_FIX_G", "PARTNER_PSEL_FIX_DZ",
            "PARTNER_PSEL_G", "PARTNER_PSEL_DZ")


def _clear(monkeypatch):
    for name in PSEL_ENV:
        monkeypatch.delenv(name, raising=False)


def _score(o):
    """Same shape as the selector's proxy; the candidates carry it directly."""
    return o[1]


def test_off_path_is_the_shipped_constants(monkeypatch):
    """With every knob unset the helper reproduces the hard-coded pair, and
    the caller's `if _div != 1.0` guard means `hp_ref` is never touched."""
    _clear(monkeypatch)
    div, dz = lg.psel_selector_params()
    assert div == 1.0
    assert dz == 0.985


def test_legacy_psel_fix_semantics_unchanged(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PARTNER_PSEL_FIX", "1")
    div, dz = lg.psel_selector_params()
    assert div == 1.0 + 0.29
    assert dz == 1.0
    monkeypatch.setenv("PARTNER_PSEL_FIX_G", "0.5")
    monkeypatch.setenv("PARTNER_PSEL_FIX_DZ", "0.99")
    div, dz = lg.psel_selector_params()
    assert div == 1.5
    assert dz == 0.99


def test_dz_knob_is_independent_of_hp_ref(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PARTNER_PSEL_DZ", "1.0")
    div, dz = lg.psel_selector_params()
    assert div == 1.0          # hp_ref untouched
    assert dz == 1.0


def test_g_knob_is_independent_of_the_dead_zone(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PARTNER_PSEL_G", "0.07")
    div, dz = lg.psel_selector_params()
    assert div == 1.07
    assert dz == 0.985


def test_g_composes_with_psel_fix(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PARTNER_PSEL_FIX", "1")
    monkeypatch.setenv("PARTNER_PSEL_G", "0.1")
    div, _dz = lg.psel_selector_params()
    assert math.isclose(div, 1.29 * 1.1)


def test_malformed_env_falls_back_to_shipped(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setenv("PARTNER_PSEL_DZ", "not-a-number")
    monkeypatch.setenv("PARTNER_PSEL_G", "")
    div, dz = lg.psel_selector_params()
    assert (div, dz) == (1.0, 0.985)


def test_pick_reproduces_the_shipped_expression():
    """`psel_pick` is a verbatim extraction: check it against the original
    inline expression on a grid of (col, dir, dz)."""
    def shipped(best_col, best_dir, score, dz):
        if best_dir is not None and (
                best_col is None or score(best_dir) < dz * score(best_col)):
            return best_dir, True
        return best_col, False

    cands = [None] + [("c", v) for v in (0.9, 1.0, 1.0001, 1.2, 2.0)]
    for bc in cands:
        for bd in cands:
            for dz in (0.985, 1.0, 1.05):
                if bc is None and bd is None:
                    continue
                assert lg.psel_pick(bc, bd, _score, dz) == \
                    shipped(bc, bd, _score, dz)


def test_dead_zone_blocks_a_marginal_direct_win():
    """The v6 tid 66 shape: the direct candidate is proxy-better AND
    true-cost better, but inside the 1.5% dead zone, so the column champion
    ships.  dz=1.0 flips it."""
    col = ("col", 1.3200)
    dr = ("dir", 1.3090)          # 0.8% better -> inside the dead zone
    assert lg.psel_pick(col, dr, _score, 0.985) == (col, False)
    assert lg.psel_pick(col, dr, _score, 1.0) == (dr, True)


def test_empty_direct_channel_ships_the_column_champion():
    """The measured failure mode of Sec.17r: when the flow channel yields no
    refined candidate the column champion ships unconditionally, however
    bad it is -- no dead zone or hp_ref calibration can change that."""
    col = ("col", 3.05)
    assert lg.psel_pick(col, None, _score, 0.985) == (col, False)
    assert lg.psel_pick(col, None, _score, 1.0) == (col, False)
