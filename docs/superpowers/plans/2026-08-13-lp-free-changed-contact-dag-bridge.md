# LP-Free Changed-Contact Separation-DAG Grouping Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off, LP-free changed-contact DAG bridge that can create one missing evaluator-semantic grouping contact while preserving all hard constraints and the current baseline on failure.

**Architecture:** Keep all production geometry and acceptance logic in `partner/violation_killer.py`. Model each axis as affine equalities plus separation edges, eliminate/merge equality roots, tighten a reduced DAG by deterministic forward/reverse projections, and reject every inconsistent, pinned, cyclic, bounded, non-finite, or soft-regressing candidate. Integrate only after the existing local bridge; the final `contest_optimizer.py` hook is serialized and owned by the integrator after core review.

**Tech Stack:** Python 3.12, NumPy, pytest, `time.perf_counter`, existing evaluator/scorer and `uv`.

## Global Constraints

- Production code remains in `partner/violation_killer.py`; no new production module.
- The feature is controlled by default-off `PARTNER_GROUP_DAG_BRIDGE`; budget/debug variables are `PARTNER_GROUP_DAG_BRIDGE_BUDGET` and `PARTNER_GROUP_DAG_BRIDGE_DEBUG`.
- Dimensions, areas, preplaced origins, pins, hard legality, and evaluator grouping semantics must remain exact; corner-only touch is invalid and positive shared-edge overlap is required.
- Caps are components `12`, contact choices `4`, reduced edges `8192`, and two projection sweeps; use only a secondary `perf_counter` deadline.
- Acceptance requires strict grouping decrease, strict exact total-V decrease, strict `_Ctx` improvement, per-item/per-group/MIB `_SoftProfile` non-regression, exact dimensions, and `_final_guards_ok`; failures return identity.
- Mechanism must not call `_fix_grouping` or fixed-topology `coord_polish`; one shared scorer is used.
- G0 requires 100/100 feasible, score `<= 1.1412448258795715`, and causal
  mean `<= 0.75 ms/case`. G1 requires three reversed ON/OFF pairs, mean
  ON-minus-OFF `<= -0.002`, and ON average runtime `<= 0.300 s`. Keep default
  off if either fails.
- Track A alone cannot reach the final goal of exact full100 score `1.00 @ <=0.300s`; do not package or review promotion.

## Files and ownership

- Modify `partner/violation_killer.py`: frozen axis/contact/profile dataclasses, DAG solver, contact construction/projection, diagnostics, wrapper.
- Modify `tests/test_partner_group_bridge.py`: primitive, geometry, acceptance, determinism, and mechanism tests.
- Modify `tests/test_partner_tag_compress.py`: flag/order/shared-scorer/diagnostic integration tests.
- Create `scripts/probes/group_dag_bridge_probe.py`: stable G0 replay probe and source-hygiene scan.
- Final hook in `partner/contest_optimizer.py` is serialized and owned by the integrator after the core review; no package/checkpoint files.

### Task 1: Axis DAG primitives

**Files:**
- Modify: `partner/violation_killer.py`
- Test: `tests/test_partner_group_bridge.py`

**Interfaces:**
- Add frozen dataclasses:
```python
@dataclass(frozen=True)
class _AxisEquality: left: int; right: int; delta: float
@dataclass(frozen=True)
class _AxisEdge: before: int; after: int; gap: float
@dataclass(frozen=True)
class _AxisProblem:
    coords: np.ndarray; sizes: np.ndarray; lower: np.ndarray; upper: np.ndarray
    pinned: np.ndarray
    equalities: Tuple[_AxisEquality, ...]
    edges: Tuple[_AxisEdge, ...]
```
- Add `_solve_axis_dag(problem: _AxisProblem) -> Optional[np.ndarray]`.

- [ ] **Step 1: Write failing primitive tests** for inconsistent equality, positive same-root edge, cycle, pin conflict, bounds conflict, exact pin preservation, and deterministic repeated solutions.
```python
from dataclasses import replace


def test_axis_solver_rejects_bad_equality_cycle_pin_and_bounds():
    base = _AxisProblem(np.array([0., 2.]), np.ones(2),
                        np.array([-10., -10.]), np.array([10., 10.]),
                        np.array([False, False]), (), ())
    assert _solve_axis_dag(replace(base, equalities=(
        _AxisEquality(0, 1, 0.), _AxisEquality(1, 0, 1.)))) is None
    assert _solve_axis_dag(replace(base, edges=(_AxisEdge(0, 0, 1.),))) is None

    pinned = replace(base, pinned=np.array([True, True]),
                     equalities=(_AxisEquality(0, 1, 3.),))
    assert _solve_axis_dag(pinned) is None

    cycle = replace(base, edges=(
        _AxisEdge(0, 1, 1.), _AxisEdge(1, 0, 1.),
    ))
    assert _solve_axis_dag(cycle) is None

    bounded = replace(base, lower=np.array([0., 0.]),
                      upper=np.array([0., 1.]),
                      edges=(_AxisEdge(0, 1, 2.),))
    assert _solve_axis_dag(bounded) is None


def test_axis_solver_preserves_pin_satisfies_edges_and_is_repeatable():
    problem = _AxisProblem(
        coords=np.array([0., 7., 11.]), sizes=np.array([2., 2., 1.]),
        lower=np.array([0., 0., 0.]), upper=np.array([0., 20., 20.]),
        pinned=np.array([True, False, False]),
        equalities=(_AxisEquality(1, 2, 4.),),
        edges=(_AxisEdge(0, 1, 3.),),
    )
    first = _solve_axis_dag(problem)
    second = _solve_axis_dag(problem)
    assert first is not None and np.array_equal(first, second)
    assert first[0] == 0.
    assert first[1] >= first[0] + 3.
    assert first[2] == pytest.approx(first[1] + 4., abs=1e-9)
```
- [ ] **Step 2: Run** `uv run pytest tests/test_partner_group_bridge.py -k 'axis_solver' -q`; expected initial FAIL because the solver/tests are absent.
- [ ] **Step 3: Implement weighted union/affine elimination** with this concrete algorithm:
```python
def _solve_axis_dag(problem: _AxisProblem) -> Optional[np.ndarray]:
    n = len(problem.coords)
    if not _axis_problem_shapes_ok(problem, n):
        return None
    parent = np.arange(n, dtype=np.int64)
    offset = np.zeros(n, dtype=np.float64)  # coord[i] = coord[root] + offset[i]

    def find(i: int) -> Tuple[int, float]:
        if parent[i] != i:
            root, delta = find(int(parent[i]))
            offset[i] += delta
            parent[i] = root
        return int(parent[i]), float(offset[i])

    # Equality semantics: coord[right] = coord[left] + delta.
    for eq in problem.equalities:
        ra, da = find(eq.left)
        rb, db = find(eq.right)
        if ra == rb:
            if not np.isclose(db - da, eq.delta, atol=1e-9, rtol=0.0):
                return None
            continue
        parent[rb] = ra
        offset[rb] = da + eq.delta - db

    roots = sorted({find(i)[0] for i in range(n)})
    root_index = {root: k for k, root in enumerate(roots)}
    root_lo = np.full(len(roots), -np.inf)
    root_hi = np.full(len(roots), np.inf)
    targets: List[List[float]] = [[] for _ in roots]
    pin_value: Dict[int, float] = {}
    for i in range(n):
        root, delta = find(i)
        k = root_index[root]
        root_lo[k] = max(root_lo[k], float(problem.lower[i] - delta))
        root_hi[k] = min(root_hi[k], float(problem.upper[i] - delta))
        targets[k].append(float(problem.coords[i] - delta))
        if problem.pinned[i]:
            value = float(problem.coords[i] - delta)
            if k in pin_value and not np.isclose(pin_value[k], value,
                                                 atol=1e-9, rtol=0.0):
                return None
            pin_value[k] = value
    for k, value in pin_value.items():
        root_lo[k] = root_hi[k] = value

    # Collapse parallel root edges by their strongest required gap.
    reduced: Dict[Tuple[int, int], float] = {}
    for edge in problem.edges:
        ra, da = find(edge.before)
        rb, db = find(edge.after)
        gap = float(da + edge.gap - db)
        if ra == rb:
            if gap > 1e-9:
                return None
            continue
        key = (root_index[ra], root_index[rb])
        reduced[key] = max(reduced.get(key, -np.inf), gap)
    order = _topological_order(len(roots), tuple(reduced))
    if order is None:
        return None

    incoming = _incoming_edges(order, reduced)
    outgoing = _outgoing_edges(order, reduced)
    for v in order:
        for u, gap in incoming[v]:
            root_lo[v] = max(root_lo[v], root_lo[u] + gap)
    for u in reversed(order):
        for v, gap in outgoing[u]:
            root_hi[u] = min(root_hi[u], root_hi[v] - gap)
    if np.any(root_lo > root_hi + 1e-9):
        return None

    target = np.array([np.median(values) for values in targets])
    x = np.clip(target, root_lo, root_hi)
    for _ in range(2):
        for v in order:
            if incoming[v]:
                x[v] = max(x[v], max(x[u] + gap for u, gap in incoming[v]))
            x[v] = min(x[v], root_hi[v])
        for u in reversed(order):
            cap = root_hi[u]
            if outgoing[u]:
                cap = min(cap, min(x[v] - gap for v, gap in outgoing[u]))
            x[u] = max(root_lo[u], min(cap, max(x[u], target[u])))

    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        root, delta = find(i)
        result[i] = x[root_index[root]] + delta
    return result if _axis_solution_ok(problem, result) else None
```
Implement `_axis_problem_shapes_ok`, `_topological_order`, edge-list builders,
and `_axis_solution_ok` in the same task. The final predicate rechecks every
original bound, equality, edge, finite value, and pinned coordinate rather
than trusting the reduced representation.
- [ ] **Step 4: Run** the focused axis tests; expected PASS with finite output, all inequalities, exact pins, and repeatability.
- [ ] **Step 5: Refresh the graph** with `graphify update .`; expected success.
  Inspect `git status --short graphify-out` but do not stage unrelated graph
  dirt in this task.
- [ ] **Step 6: Commit** `git add partner/violation_killer.py tests/test_partner_group_bridge.py && git commit -m "feat: add affine separation dag solver"`.

### Task 2: Changed-contact construction and public wrapper

**Files:**
- Modify: `partner/violation_killer.py`
- Test: `tests/test_partner_group_bridge.py`

**Interfaces:**

The task produces these exact call signatures:

```text
@dataclass(frozen=True)
class _ContactChoice:
    group_id: int; a: int; b: int; axis: int
    a_before_b: bool; perp_delta: float
@dataclass(frozen=True)
class _SoftProfile:
    boundary_satisfied: frozenset[Tuple[int, int]]
    grouping_connected: frozenset[Tuple[int, int, int]]
    mib_equal: frozenset[Tuple[int, int, int]]
_soft_profile(opt, P: np.ndarray) -> _SoftProfile
_profile_nonregressing(before: _SoftProfile, after: _SoftProfile) -> bool
_separation_edges(P: np.ndarray, axis: int) -> Tuple[_AxisEdge, ...]
_contact_forest(P: np.ndarray, members: Sequence[int]) -> Tuple[Tuple[int, int, int], ...]
_forest_equalities(P: np.ndarray, forest: Sequence[Tuple[int, int, int]]) -> Tuple[Tuple[_AxisEquality, ...], Tuple[_AxisEquality, ...]]
_make_axis_problem(P: np.ndarray, axis: int, equalities: Sequence[_AxisEquality], edges: Sequence[_AxisEdge], kind: Sequence[int]) -> _AxisProblem
_positive_shared_edge(P: np.ndarray, choice: _ContactChoice) -> bool
_enumerate_contact_choices(opt, P: np.ndarray, max_contacts: int = 4) -> List[_ContactChoice]
def _project_changed_contact(opt, P: np.ndarray, choice: _ContactChoice,
                             budget_s: float = 0.003) -> Optional[np.ndarray]
bridge_grouping_violations_dag(opt, out, budget_s: float = 0.003)
```
- [ ] **Step 1a: Add the exact changed-contact RED fixture and test.** The
  `_coordinated_chain_case` helper must assert its premise before invoking the
  new method: `bridge_grouping_violations` returns identity, the group has two
  components, and `_violations_exact>0`. Then test the new mechanism:
```python
def test_dag_bridge_creates_positive_shared_edge_and_reduces_grouping():
    opt, out = _coordinated_chain_case()
    P = np.asarray(out, float)
    assert vk.bridge_grouping_violations(opt, out, 0.2) is out
    assert len(vk._components(P, np.asarray(opt.cluster_groups[1]))) == 2
    before = vk._grouping_count(opt, P)
    got = vk.bridge_grouping_violations_dag(opt, out, 0.2)
    Q = np.asarray(got, float)
    assert vk._grouping_count(opt, Q) < before
    assert vk._violations_exact(opt, Q) < vk._violations_exact(opt, P)
    assert vk._Ctx(opt, P).score(Q)[0] < vk._Ctx(opt, P).score(P)[0]
    assert np.array_equal(Q[:, 2:], P[:, 2:])
    assert np.array_equal(Q[np.asarray(opt.kind) == 2],
                          P[np.asarray(opt.kind) == 2])
```
- [ ] **Step 1b: Add relation-level regression tests.** Boundary relations are
  `(block, bit)` for every currently satisfied bit in the evaluator bitmask;
  grouping relations are `(group_id,min_block,max_block)` for every pair in
  the same evaluator component; MIB relations use the same triple for every
  equal-shape pair. Non-regression is the three before-set subset checks.
```python
def test_soft_profile_rejects_corner_bit_swap_and_connected_pair_loss():
    opt, before, bit_swap, pair_loss = _soft_profile_regression_cases()
    p0 = vk._soft_profile(opt, before)
    assert (0, 1) in p0.boundary_satisfied
    assert (0, 8) in p0.boundary_satisfied
    assert not vk._profile_nonregressing(
        p0, vk._soft_profile(opt, bit_swap)
    )
    assert (1, 1, 2) in p0.grouping_connected
    assert not vk._profile_nonregressing(
        p0, vk._soft_profile(opt, pair_loss)
    )
```
  Add these separate exact tests for geometry, failure identity, determinism,
  and mechanism exclusion:
```python
import types


def test_positive_shared_edge_rejects_corner_only_touch():
    choice = vk._ContactChoice(1, 0, 1, 0, True, 0.)
    edge = np.asarray([(0., 0., 2., 2.), (2., 1., 2., 2.)])
    corner = np.asarray([(0., 0., 2., 2.), (2., 2., 2., 2.)])
    assert vk._positive_shared_edge(edge, choice)
    assert not vk._positive_shared_edge(corner, choice)


def test_impossible_dag_bridge_is_exact_identity(monkeypatch):
    opt, out = _coordinated_chain_case()
    monkeypatch.setattr(vk, "_solve_axis_dag", lambda problem: None)
    assert vk.bridge_grouping_violations_dag(opt, out, 0.2) is out


def test_dag_bridge_is_repeatable_and_preserves_hard_geometry():
    opt, out = _coordinated_chain_case()
    one = vk.bridge_grouping_violations_dag(opt, list(out), 0.2)
    two = vk.bridge_grouping_violations_dag(opt, list(out), 0.2)
    assert one == two
    P, Q = np.asarray(out), np.asarray(one)
    assert np.array_equal(Q[:, 2:], P[:, 2:])
    assert np.array_equal(Q[np.asarray(opt.kind) == 2],
                          P[np.asarray(opt.kind) == 2])


def test_dag_mechanism_never_calls_legacy_grouping_or_coord_polish(monkeypatch):
    opt, out = _coordinated_chain_case()
    monkeypatch.setattr(vk, "_fix_grouping",
                        lambda *a, **k: pytest.fail("legacy grouping called"))
    fake = types.SimpleNamespace(
        polish_layout=lambda *a, **k: pytest.fail("coord polish called")
    )
    monkeypatch.setitem(sys.modules, "coord_polish", fake)
    got = vk.bridge_grouping_violations_dag(opt, out, 0.2)
    assert vk._grouping_count(opt, np.asarray(got)) < \
        vk._grouping_count(opt, np.asarray(out))
```
- [ ] **Step 2: Run** `uv run pytest tests/test_partner_group_bridge.py -k
  'dag_bridge or contact or soft_profile or mechanism' -q`; expected RED with
  missing DAG/profile symbols, while the fixture premise assertions pass.
- [ ] **Step 3: Implement contact construction and projection** using the following complete control flow:
```python
choices = _enumerate_contact_choices(opt, P, max_contacts=4)
for choice in choices:
    axis, perp = choice.axis, 1 - choice.axis
    before = choice.a if choice.a_before_b else choice.b
    after = choice.b if choice.a_before_b else choice.a
    # Equality semantics: coordinate[right] = coordinate[left] + delta.
    contact_eq = _AxisEquality(before, after, float(P[before, 2 + axis]))
    perp_eq = _AxisEquality(choice.a, choice.b, choice.perp_delta)
    axis_edges = tuple(
        edge for edge in _separation_edges(P, axis)
        if {edge.before, edge.after} != {choice.a, choice.b}
    )
    perp_edges = tuple(
        edge for edge in _separation_edges(P, perp)
        if {edge.before, edge.after} != {choice.a, choice.b}
    )
    forest = _contact_forest(P, opt.cluster_groups[choice.group_id])
    axis_equalities, perp_equalities = _forest_equalities(P, forest)
    axis_equalities += (contact_eq,)
    perp_equalities += (perp_eq,)
    q_axis = _solve_axis_dag(
        _make_axis_problem(P, axis, axis_equalities, axis_edges, opt.kind)
    )
    q_perp = _solve_axis_dag(
        _make_axis_problem(P, perp, perp_equalities, perp_edges, opt.kind)
    )
    if q_axis is not None and q_perp is not None:
        Q = P.copy()
        Q[:, axis] = q_axis
        Q[:, perp] = q_perp
        if _positive_shared_edge(Q, choice):
            return Q
return None
```
The implementation must discover components with `_components`, cap components
at 12 and reduced edges at 8192, and preserve each existing grouping component
as a rigid contact-spanning tree: `_forest_equalities` emits the current x and
y offsets of every forest edge, without closing an equality cycle. Assign each
currently non-overlapping pair to exactly one deterministic separating axis
(largest normalized gap, then x on ties); `_separation_edges(P, axis)` returns
only pairs assigned to that axis. Remove only the selected pair's old
separator, enforce exact contact-axis equality, choose the deterministic
perpendicular delta in `[-size_b+JOIN, size_a-JOIN]`, and return identity on any
exception, deadline, non-finite value, pin conflict, cycle, corner contact, or
guard failure.
- [ ] **Step 4: Implement the public wrapper and acceptance as one scorer**
with explicit profile comparison:
```python
def bridge_grouping_violations_dag(opt, out, budget_s: float = 0.003):
    try:
        budget = float(budget_s)
        if not math.isfinite(budget) or budget <= 0.0:
            return out
        P0 = np.asarray(out, dtype=np.float64)
        if P0.shape != (int(opt.n), 4) or not np.isfinite(P0).all():
            return out
        grouping0 = _grouping_count(opt, P0)
        if grouping0 <= 0:
            return out
        ctx = _Ctx(opt, P0)
        score0, violations0 = ctx.score(P0)
        profile0 = _soft_profile(opt, P0)
        deadline = time.perf_counter() + budget
        for choice in _enumerate_contact_choices(opt, P0, max_contacts=4):
            if time.perf_counter() >= deadline:
                break
            Q = _project_changed_contact(
                opt, P0, choice, max(0.0, deadline - time.perf_counter())
            )
            if Q is None:
                continue
            score1, violations1 = ctx.score(Q)
            accepted = (
                _grouping_count(opt, Q) < grouping0
                and violations1 < violations0
                and score1 < score0 - 1e-12
                and _profile_nonregressing(profile0, _soft_profile(opt, Q))
                and np.array_equal(Q[:, 2:], P0[:, 2:])
                and _positive_shared_edge(Q, choice)
                and _final_guards_ok(
                    opt, P0, Q, list(opt.kind), list(opt.areas)
                )
            )
            if accepted:
                return [tuple(map(float, row)) for row in Q]
        return out
    except Exception:
        return out
```
`_soft_profile` must materialize the relation sets defined by Step 1b, and
`_profile_nonregressing(before, after)` returns true only when all three
before-sets are subsets of their corresponding after-sets. This prevents a
corner-tagged block from exchanging one satisfied bit for another while its
per-block evaluator violation count remains unchanged. Use one `_Ctx` scorer,
stable reason/case/candidate/component/edge/sweep/timing diagnostics, and never
call `_fix_grouping` or `coord_polish`.
- [ ] **Step 5: Run** focused tests; expected PASS, including corner invalidity and pinned-origin exactness.
- [ ] **Step 6: Refresh the graph** with `graphify update .`; inspect but do
  not stage unrelated graph dirt.
- [ ] **Step 7: Commit** `git add partner/violation_killer.py tests/test_partner_group_bridge.py && git commit -m "feat: add changed-contact dag grouping bridge"`.

### Task 3: Serialized hook, flags, and diagnostics

**Files:**
- Modify: `partner/contest_optimizer.py` (integrator-owned, after Task 2 review)
- Test: `tests/test_partner_tag_compress.py`

**Interfaces:**
- Preserve current local bridge as preceding path; when
  `PARTNER_GROUP_DAG_BRIDGE=1`, call
  `bridge_grouping_violations_dag(scorer, current, budget)` after it, even if
  only the DAG flag is set. Treat only the literal string `"1"` as enabled.
- Read `PARTNER_GROUP_DAG_BRIDGE_BUDGET` and `PARTNER_GROUP_DAG_BRIDGE_DEBUG`; use one scorer and warmed `perf_counter` buckets.

- [ ] **Step 1: Write these failing integration tests** for default-off identity,
  local-before-DAG order, DAG-only activation, literal `"0"` disabled, shared
  scorer identity, failure preservation, constructor warm-up under DAG-only,
  environment cleanup for all three DAG variables, and diagnostics containing
  activation/timing/grouping/V/HPWL/bbox/commit fields.
```python
def test_dag_only_runs_local_then_dag_with_one_scorer(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    sentinel, events = object(), []
    monkeypatch.setattr(co, "_ColumnOptimizer", lambda *a, **k: sentinel)
    monkeypatch.setattr(
        "violation_killer.bridge_grouping_violations",
        lambda scorer, value, budget:
            events.append(("local", scorer)) or value,
    )
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr(
        "violation_killer.bridge_grouping_violations_dag",
        lambda scorer, value, budget:
            events.append(("dag", scorer)) or value,
    )
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    assert events == [("local", sentinel), ("dag", sentinel)]

def test_literal_zero_is_disabled_without_scorer(monkeypatch):
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "0")
    monkeypatch.setattr(co, "_ColumnOptimizer",
                        lambda *a, **k: pytest.fail("scorer constructed"))
    out = [(0., 0., 1., 1.)]
    assert _opt()._tag_compress(
        out, torch.ones(1), torch.zeros((1, 5)),
        torch.full((1, 4), -1.), torch.zeros((0, 3)),
        torch.zeros((0, 3)), torch.zeros((0, 2)), None,
    ) is out
```
  Extend `_ENV` with the three DAG variables. Add named tests
  `test_dag_only_warms_dependencies`, `test_dag_failure_preserves_local_result`,
  and `test_dag_debug_reports_self_paired_fields`; the last asserts
  `("ms=","grouping=","V=","hpwl=","bbox=","committed=")` in stderr.
```python
def test_dag_only_warms_dependencies(monkeypatch):
    calls = []
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setattr(tc, "warm_dependencies", lambda: calls.append(1))
    _opt()
    assert calls == [1]


def test_dag_failure_preserves_local_result(monkeypatch):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    local = [(7., 7., 1., 1.)] * n
    monkeypatch.setattr("violation_killer.bridge_grouping_violations",
                        lambda *a: local)
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr(
        "violation_killer.bridge_grouping_violations_dag",
        lambda *a: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert _opt()._tag_compress(
        list(rects), at, cons, tpos, b2b, p2b, pins, None
    ) is local


def test_dag_debug_reports_self_paired_fields(monkeypatch, capsys):
    n, at, cons, tpos, b2b, p2b, pins, rects = _single()
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE", "1")
    monkeypatch.setenv("PARTNER_GROUP_DAG_BRIDGE_DEBUG", "1")
    monkeypatch.setattr("violation_killer._grouping_count", lambda *a: 1)
    monkeypatch.setattr("violation_killer.bridge_grouping_violations_dag",
                        lambda scorer, value, budget: value)
    _opt()._tag_compress(list(rects), at, cons, tpos, b2b, p2b, pins, None)
    err = capsys.readouterr().err
    for field in ("ms=", "grouping=", "V=", "hpwl=", "bbox=", "committed="):
        assert field in err
```
- [ ] **Step 2: Run** `uv run pytest tests/test_partner_tag_compress.py -k
  'dag_bridge or bridge_order or shared_scorer or debug or warms' -q`; expected
  RED because DAG integration symbols/behavior are absent.
- [ ] **Step 3: Add the serialized hook** immediately after the local bridge:
```python
dag_on = os.environ.get("PARTNER_GROUP_DAG_BRIDGE") == "1"
local_on = os.environ.get("PARTNER_GROUP_BRIDGE") == "1" or dag_on
if not (tag_on or local_on):
    return out

# Existing local bridge executes first whenever either grouping flag is on.
if local_on:
    try:
        local_budget = float(os.environ.get(
            "PARTNER_GROUP_BRIDGE_BUDGET", "0.02"
        ))
    except (TypeError, ValueError):
        local_budget = 0.02
    try:
        from violation_killer import bridge_grouping_violations
        current = bridge_grouping_violations(scorer, current, local_budget)
    except Exception:
        pass

if dag_on:
    import numpy as _np
    from violation_killer import (
        _grouping_count,
        bridge_grouping_violations_dag,
    )
    P = _np.asarray([tuple(map(float, row)) for row in current], dtype=float)
    if _grouping_count(scorer, P) > 0:
        try:
            dag_budget = float(os.environ.get(
                "PARTNER_GROUP_DAG_BRIDGE_BUDGET", "0.003"
            ))
        except (TypeError, ValueError):
            dag_budget = 0.003
        before_dag = current
        started = time.perf_counter()
        current = bridge_grouping_violations_dag(
            scorer, current, dag_budget
        )
        if os.environ.get("PARTNER_GROUP_DAG_BRIDGE_DEBUG") == "1":
            _print_dag_bridge_diag(
                scorer, before_dag, current,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
```
Extend `_warm_tag_compress_dependencies` so DAG-only activation warms the same
dependencies outside the case timer. Preserve existing fallback, do no
graph-build work for zero residual grouping, factor the existing local debug
calculation into `_print_dag_bridge_diag` rather than duplicating scorer work,
use one scorer, and do not alter Track B/package paths.
- [ ] **Step 4: Run** targeted tests and `bash scripts/validate.sh`; expected PASS.
- [ ] **Step 5: Refresh the graph** with `graphify update .`; inspect but do
  not stage unrelated graph dirt.
- [ ] **Step 6: Commit** `git add partner/contest_optimizer.py tests/test_partner_tag_compress.py && git commit -m "feat: gate dag bridge after local bridge"`.

### Task 4: G0 stable replay probe and evidence

**Files:**
- Create: `scripts/probes/group_dag_bridge_probe.py`
- Test: `tests/test_partner_group_bridge.py`

- [ ] **Step 1: Write failing CLI/manifest tests** using a two-case fixture and
  monkeypatched scorer. Assert separate case/manifest paths, exact schema keys,
  deterministic reruns, and nonzero exit on baseline hash/score mismatch.
```python
def test_g0_probe_writes_separate_deterministic_case_and_manifest_files(
        tmp_path, monkeypatch):
    probe = _load_probe_module("group_dag_bridge_probe")
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(_two_case_g0_baseline(), sort_keys=True))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256",
                        hashlib.sha256(source.read_bytes()).hexdigest())
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SCORE",
                        _weighted_fixture_score(source))
    out, manifest = tmp_path / "cases.json", tmp_path / "manifest.json"
    args = ["replay", "--input", str(source), "--output", str(out),
            "--manifest", str(manifest)]
    assert probe.main(args, case_loader=_two_case_inputs,
                      bridge_fn=_fixture_bridge,
                      evaluate_fn=_fixture_evaluate) == 0
    first = (out.read_bytes(), manifest.read_bytes())
    assert probe.main(args, case_loader=_two_case_inputs,
                      bridge_fn=_fixture_bridge,
                      evaluate_fn=_fixture_evaluate) == 0
    assert first == (out.read_bytes(), manifest.read_bytes())
    m = json.loads(manifest.read_text())
    assert set(m) == {
        "schema", "baseline", "feasible", "errors", "score_on",
        "score_off", "grouping_delta", "v_delta", "runtime_ms",
        "causal_mean_ms", "accepted", "source_hygiene",
    }
    assert m["schema"] == "group-dag-g0.v1"


def test_g0_probe_fails_closed_on_baseline_hash_or_score_mismatch(
        tmp_path, monkeypatch):
    probe = _load_probe_module("group_dag_bridge_probe")
    source = tmp_path / "baseline.json"
    source.write_text(json.dumps(_two_case_g0_baseline()))
    monkeypatch.setattr(probe, "EXPECTED_BASELINE_SHA256", "0" * 64)
    assert probe.main(["replay", "--input", str(source), "--output",
                       str(tmp_path / "o.json"), "--manifest",
                       str(tmp_path / "m.json")]) != 0
```
- [ ] **Step 2: Run** `uv run pytest tests/test_partner_group_bridge.py -k
  'g0_probe or source_hygiene' -q`; expected FAIL because the probe is absent.
- [ ] **Step 3: Implement the probe** to load
  `artifacts/partner_eval/gbridge_package_full100.json`, verify exact SHA256
  `6e7089b1e0fa4510f5ffe11187232bee488b25da599435f7d369ba329a1349c5` and
  exact weighted `score_off=1.1437448258795715`, load validation inputs only for
  offline G0 scoring, invoke the DAG wrapper directly on the recorded exact
  post-local layouts (never rerun the local bridge), score through
  `scripts.iccad2026_evaluate.evaluate_solution/compute_total_score`, and write
  case JSON plus a separate manifest containing feasibility/errors, ON/OFF
  score, grouping/V deltas, runtime median/p95/max, and diagnostics.
- [ ] **Step 4: Add source-hygiene assertions** scanning only the DAG
  production policy for forbidden test IDs, saved layouts/coordinates, and
  block-count-specific selectors; allow generic evaluator component counting,
  and reject imports/calls to fixed-topology mechanisms. Do not read untracked
  scratchpad data.
- [ ] **Step 5: Run**:
```bash
uv run python scripts/probes/group_dag_bridge_probe.py replay \
  --input artifacts/partner_eval/gbridge_package_full100.json \
  --output artifacts/partner_eval/group_dag_g0.json \
  --manifest artifacts/partner_eval/group_dag_g0.manifest.json
```
Expected: 100 case records and a separate manifest with
`schema="group-dag-g0.v1"`, the baseline path and SHA256, `feasible`, `errors`,
`score_on`, `score_off`, `grouping_delta`, `v_delta`,
`runtime_ms={"mean","median","p95","max"}`, `causal_mean_ms`, `accepted`, and
`source_hygiene`.
- [ ] **Step 6: Review G0:** require 100/100 feasible, zero errors, score
  `<= 1.1412448258795715`, causal mean `<= 0.75 ms/case`, and every accepted
  candidate has strict grouping and exact total-V decrease; otherwise retain
  flag off and record failure.
- [ ] **Step 7: Commit**:
```bash
git add scripts/probes/group_dag_bridge_probe.py tests/test_partner_group_bridge.py
git commit -m "test: add changed-contact dag g0 probe"
```

### Task 5: G1 paired gate and decision record

**Files:**
- Modify: `scripts/probes/group_dag_bridge_probe.py`
- Modify: `tests/test_partner_group_bridge.py`
- Create: `docs/experiments/2026-08-13-lp-free-changed-contact-dag-bridge-gates.md`
  only by the adjudicator after G0; no package files.

- [ ] **Step 1: Write the deterministic adjudicator RED tests.** A run fixture
  contains 100 records with IDs `0..99`, `is_feasible=True`, `error=None`, a
  top-level weighted score, and summary `avg_runtime`. Exact pass and fail
  assertions are:
```python
def test_g1_gate_computes_reversed_pairs_and_is_the_only_verdict(tmp_path):
    probe = _load_probe_module("group_dag_bridge_probe")
    paths = _write_six_g1_runs(
        tmp_path,
        scores=(1.1450, 1.1428, 1.1419, 1.1440, 1.1435, 1.1414),
        runtimes=(.298, .299, .2995, .298, .298, .2999),
    )  # roles: off,on,on,off,off,on
    report = tmp_path / "decision.md"
    result = probe.adjudicate_g1(paths, report)
    assert result["paired_on_minus_off"] == pytest.approx(
        [-.0022, -.0021, -.0021]
    )
    assert result["mean_on_minus_off"] == pytest.approx(-.002133333333)
    assert result["verdict"] == "PROMOTE"
    assert "PROMOTE" in report.read_text()


@pytest.mark.parametrize("mutation", ["missing_id", "error", "infeasible",
                                      "runtime", "quality"])
def test_g1_gate_fails_closed_for_every_binding_gate(tmp_path, mutation):
    probe = _load_probe_module("group_dag_bridge_probe")
    paths = _write_passing_six_g1_runs(tmp_path)
    _mutate_g1_fixture(paths, mutation)
    result = probe.adjudicate_g1(paths, tmp_path / "decision.md")
    assert result["verdict"] == "HOLD"
```
- [ ] **Step 2: Run** `uv run pytest tests/test_partner_group_bridge.py -k
  'g1_gate' -q`; expected RED because `adjudicate_g1` is absent.
- [ ] **Step 3: Implement `adjudicate_g1`.** It accepts the ordered roles
  `off,on,on,off,off,on`; validates exactly 100 unique records/IDs `0..99`,
  zero errors, and all feasible in every arm; computes pair deltas
  `(on1-off1,on2-off2,on3-off3)`; requires their mean `<=-0.002`; requires
  every ON `summary.avg_runtime<=.300`; and writes the sole deterministic
  Markdown `PROMOTE`/`HOLD` decision with artifact SHA256s and all gate values.
  Any parse/schema/order failure emits `HOLD` and returns nonzero from CLI.
- [ ] **Step 4: Define the fixed evaluator inputs** once:
```bash
ROOT="$(git rev-parse --show-toplevel)"
DIRECT="$ROOT/submission/cadc1013/checkpoints/direct_v2_final.pt"
FLOW="$ROOT/submission/cadc1013/checkpoints/flow_matching_v1_final.pt"
EVAL="$ROOT/scripts/iccad2026_evaluate.py"
WRAPPER="$ROOT/scripts/probes/tag_compress_warm_wrapper.py"
cd "$ROOT/FloorSet/iccad2026contest"
```
- [ ] **Step 5: Run the six reversed arms** in the fixed order OFF/ON,
ON/OFF, OFF/ON. The local bridge is enabled in every arm and DAG is the only
toggle:
```bash
env -u PARTNER_GROUP_DAG_BRIDGE -u PARTNER_GROUP_DAG_BRIDGE_DEBUG PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 DIRECT_CKPT="$DIRECT" FLOW_CKPT="$FLOW" uv run python "$EVAL" --data-path ../ --evaluate "$WRAPPER" --output "$ROOT/artifacts/partner_eval/gdag_pair1_off.json"
env -u PARTNER_GROUP_DAG_BRIDGE_DEBUG PARTNER_GROUP_DAG_BRIDGE=1 PARTNER_GROUP_DAG_BRIDGE_BUDGET=0.003 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 DIRECT_CKPT="$DIRECT" FLOW_CKPT="$FLOW" uv run python "$EVAL" --data-path ../ --evaluate "$WRAPPER" --output "$ROOT/artifacts/partner_eval/gdag_pair1_on.json"
env -u PARTNER_GROUP_DAG_BRIDGE_DEBUG PARTNER_GROUP_DAG_BRIDGE=1 PARTNER_GROUP_DAG_BRIDGE_BUDGET=0.003 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 DIRECT_CKPT="$DIRECT" FLOW_CKPT="$FLOW" uv run python "$EVAL" --data-path ../ --evaluate "$WRAPPER" --output "$ROOT/artifacts/partner_eval/gdag_pair2_on.json"
env -u PARTNER_GROUP_DAG_BRIDGE -u PARTNER_GROUP_DAG_BRIDGE_DEBUG PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 DIRECT_CKPT="$DIRECT" FLOW_CKPT="$FLOW" uv run python "$EVAL" --data-path ../ --evaluate "$WRAPPER" --output "$ROOT/artifacts/partner_eval/gdag_pair2_off.json"
env -u PARTNER_GROUP_DAG_BRIDGE -u PARTNER_GROUP_DAG_BRIDGE_DEBUG PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 DIRECT_CKPT="$DIRECT" FLOW_CKPT="$FLOW" uv run python "$EVAL" --data-path ../ --evaluate "$WRAPPER" --output "$ROOT/artifacts/partner_eval/gdag_pair3_off.json"
env -u PARTNER_GROUP_DAG_BRIDGE_DEBUG PARTNER_GROUP_DAG_BRIDGE=1 PARTNER_GROUP_DAG_BRIDGE_BUDGET=0.003 PARTNER_TAG_COMPRESS=1 PARTNER_GROUP_BRIDGE=1 DIRECT_CKPT="$DIRECT" FLOW_CKPT="$FLOW" uv run python "$EVAL" --data-path ../ --evaluate "$WRAPPER" --output "$ROOT/artifacts/partner_eval/gdag_pair3_on.json"
```
Expected: six JSONs with 100 records, no errors, and 100/100 feasibility.
- [ ] **Step 6: Adjudicate exactly once**; this command alone creates the
  decision record and exits nonzero on `HOLD`:
```bash
uv run python "$ROOT/scripts/probes/group_dag_bridge_probe.py" gate-g1 \
  --pair1-off "$ROOT/artifacts/partner_eval/gdag_pair1_off.json" \
  --pair1-on "$ROOT/artifacts/partner_eval/gdag_pair1_on.json" \
  --pair2-on "$ROOT/artifacts/partner_eval/gdag_pair2_on.json" \
  --pair2-off "$ROOT/artifacts/partner_eval/gdag_pair2_off.json" \
  --pair3-off "$ROOT/artifacts/partner_eval/gdag_pair3_off.json" \
  --pair3-on "$ROOT/artifacts/partner_eval/gdag_pair3_on.json" \
  --decision "$ROOT/docs/experiments/2026-08-13-lp-free-changed-contact-dag-bridge-gates.md"
```
  `PROMOTE` enables the default-on rollout only after integration review;
  `HOLD` retains `PARTNER_GROUP_DAG_BRIDGE=0`. Neither path packages.
- [ ] **Step 7: Run** `uv run pytest`, `bash scripts/validate.sh`,
`git diff --check`, and `graphify update .`; expected all checks pass, with
unrelated graph dirt unstaged.
- [ ] **Step 8: Commit the reviewed adjudicator/tests and its decision record**:
```bash
git add scripts/probes/group_dag_bridge_probe.py tests/test_partner_group_bridge.py \
  docs/experiments/2026-08-13-lp-free-changed-contact-dag-bridge-gates.md
git commit -m "docs: record dag bridge gate decision"
```
Evaluator JSONs remain generated evidence and are not staged.

## Self-review checklist

- [x] Spec coverage checked after independent review: solver
  rejection/projection, changed contact, caps, strict acceptance, mechanism
  exclusion, integration ordering/flags, diagnostics, G0/G1 and rollback.
- [x] No placeholders in implementation steps; every code-changing task has
  concrete signatures, test shape, commands, and expected outcomes.
- [x] Type/signature consistency checked; Task 3 consumes the Task 2 wrapper
  exactly.
- [x] Track boundary checked: no Track B files, no package review/promotion,
  and final goal remains full100 exact 1.00 at `<= 0.300 s`.

## Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-lp-free-changed-contact-dag-bridge.md`. Execution must use either subagent-driven-development or executing-plans, with independent review after each task.
