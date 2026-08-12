# Codex Model Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure Codex so GPT-5.6 Sol schedules and integrates Subagent-Driven work, GPT-5.6 Terra performs deep reasoning and review, and GPT-5.6 Luna executes bounded mechanical tasks.

**Architecture:** Add a project-scoped Codex config that pins the root scheduler and defines safe subagent defaults, plus two standalone custom-agent TOML files that pin each specialist's model, reasoning effort, sandbox, and prompt. Replace the short routing paragraph in `AGENTS.md` with the complete opt-in dispatch, ownership, escalation, fallback, and review contracts approved in the design spec.

**Tech Stack:** Codex project configuration (TOML), Markdown repository instructions, Python 3.12 `tomllib` via `uv run`, Codex CLI strict-config validation.

## Global Constraints

- Activate this routing only when the user explicitly selects Subagent-Driven execution.
- Pin the root Scheduler/Integrator to `gpt-5.6-sol` with `xhigh` reasoning.
- Pin the deep reasoner to `gpt-5.6-terra` with `xhigh` reasoning and a read-only sandbox.
- Pin the fast worker to `gpt-5.6-luna` with `low` reasoning and a workspace-write sandbox.
- Allow at most three spawned subagent threads so the primary scheduler remains the fourth active thread.
- Every dispatch must still specify model and reasoning effort explicitly.
- All agents share one worktree; parallel writers must have disjoint file ownership and must preserve unrelated edits.
- Keep solver code, evaluator paths, checkpoints, generated outputs, and the existing Claude agent files unchanged.
- Project configuration is expected to take effect on a newly started or reloaded Codex task, not retroactively in the current task.

---

### Task 1: Add Project-Scoped Codex Model Roles

**Files:**
- Create: `.codex/config.toml`
- Create: `.codex/agents/deep-reasoner.toml`
- Create: `.codex/agents/fast-worker.toml`
- Reference: `docs/superpowers/specs/2026-08-12-codex-model-orchestration-design.md`

**Interfaces:**
- Consumes: Codex's project-scoped configuration loader and standalone custom-agent schema.
- Produces: root configuration for Sol and spawnable `deep-reasoner` and `fast-worker` custom-agent names for the rules in Task 2.

- [ ] **Step 1: Run the configuration contract check before creating the files**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path

required = [
    Path(".codex/config.toml"),
    Path(".codex/agents/deep-reasoner.toml"),
    Path(".codex/agents/fast-worker.toml"),
]
missing = [str(path) for path in required if not path.is_file()]
assert not missing, f"missing Codex configuration: {missing}"
PY
```

Expected: FAIL with all three paths listed as missing.

- [ ] **Step 2: Create the root scheduler configuration**

Create `.codex/config.toml` with exactly:

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "xhigh"

[agents]
enabled = true
max_concurrent_threads_per_session = 3
default_subagent_model = "gpt-5.6-terra"
default_subagent_reasoning_effort = "high"
```

- [ ] **Step 3: Create the Terra deep-reasoner agent**

Create `.codex/agents/deep-reasoner.toml` with exactly:

```toml
name = "deep-reasoner"
description = "GPT-5.6 Terra specialist for bounded architecture analysis, experiment design, integration reasoning, subtle diagnosis, and task-scoped review."
model = "gpt-5.6-terra"
model_reasoning_effort = "xhigh"
sandbox_mode = "read-only"
developer_instructions = """
You are the deep-reasoning specialist for the ICCAD 2026 FloorSet SoC floorplanning project. The GPT-5.6 Sol scheduler gives you bounded problems that require judgment rather than mechanical execution.

Start by restating the decision or failure under investigation. Inspect the relevant repository evidence, identify assumptions and invariants, compare viable options, and return one justified recommendation. For diagnosis, separate confirmed facts from hypotheses and name the smallest experiment that would discriminate between remaining explanations.

Read AGENTS.md, CLAUDE.md, and CONTEXT.md when the task touches solver architecture or policy. Preserve the project rules: hard legality precedes soft refinement; checkpoint promotion requires evaluator evidence; runtime work must clear the budget layer; validation case IDs must never become production policy.

Remain within the assigned scope and read-only sandbox. Do not turn an analysis or review assignment into implementation. Escalate unresolved trade-offs, highest-risk solver policy, and cross-task conflicts to Sol for the final decision.

Return the decision or finding first, followed by evidence with file references, risks or uncertainty, and concrete next steps. For reviews, report only actionable correctness, security, regression, or test-evidence findings and classify their severity.
"""
```

- [ ] **Step 4: Create the Luna fast-worker agent**

Create `.codex/agents/fast-worker.toml` with exactly:

```toml
name = "fast-worker"
description = "GPT-5.6 Luna worker for well-specified implementation, commands and tests, renames, boilerplate, documentation synchronization, and fact gathering."
model = "gpt-5.6-luna"
model_reasoning_effort = "low"
sandbox_mode = "workspace-write"
developer_instructions = """
You are the fast execution worker for the ICCAD 2026 FloorSet SoC floorplanning project. The GPT-5.6 Sol scheduler gives you tasks whose design and acceptance criteria are already decided.

Execute the assigned specification exactly. Stay inside the named files or subsystem ownership, make the smallest defensible change, and stop for scheduler guidance if a material design, behaviour, or scope decision is missing. Do not redesign the task.

You are not alone in the codebase. Preserve unrelated user and agent edits, never revert work you do not own, and adapt to compatible concurrent changes. Do not hand-edit or commit checkpoints, artifacts, wandb data, logs, or other generated evidence unless the task explicitly owns them.

Use apply_patch for file edits. Run Python and tests through uv run, and invoke repository shell scripts with bash. Verify the changed behaviour before reporting completion; do not infer success from an exit command you did not run.

Return a concise handoff containing changed files, the exact verification commands and results, and any blocker or residual risk. Do not claim completion when required verification failed or was skipped.
"""
```

- [ ] **Step 5: Parse and assert the exact configuration values**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import tomllib

root = tomllib.loads(Path(".codex/config.toml").read_text())
deep = tomllib.loads(Path(".codex/agents/deep-reasoner.toml").read_text())
fast = tomllib.loads(Path(".codex/agents/fast-worker.toml").read_text())

assert root["model"] == "gpt-5.6-sol"
assert root["model_reasoning_effort"] == "xhigh"
assert root["agents"] == {
    "enabled": True,
    "max_concurrent_threads_per_session": 3,
    "default_subagent_model": "gpt-5.6-terra",
    "default_subagent_reasoning_effort": "high",
}
assert (deep["name"], deep["model"], deep["model_reasoning_effort"]) == (
    "deep-reasoner",
    "gpt-5.6-terra",
    "xhigh",
)
assert deep["sandbox_mode"] == "read-only"
assert (fast["name"], fast["model"], fast["model_reasoning_effort"]) == (
    "fast-worker",
    "gpt-5.6-luna",
    "low",
)
assert fast["sandbox_mode"] == "workspace-write"
assert all(item["developer_instructions"].strip() for item in (deep, fast))
print("Codex orchestration TOML contracts: PASS")
PY
```

Expected: `Codex orchestration TOML contracts: PASS`.

- [ ] **Step 6: Ask the installed Codex CLI to reject unknown configuration keys**

Run:

```bash
codex --strict-config features list >/dev/null
```

Expected: exit status 0. A stale temporary-directory warning is acceptable; an unknown-key or invalid-value error is not.

- [ ] **Step 7: Commit the project configuration and custom agents**

```bash
git add .codex/config.toml .codex/agents/deep-reasoner.toml .codex/agents/fast-worker.toml
git diff --cached --check
git commit -m "chore: configure codex agent roles"
```

Expected: one focused commit containing exactly the three TOML files.

---

### Task 2: Expand the Repository Orchestration Contract

**Files:**
- Modify: `AGENTS.md:30-39`
- Reference: `CLAUDE.md` section `Model orchestration (Fable 5 scheduler)`
- Reference: `docs/superpowers/specs/2026-08-12-codex-model-orchestration-design.md`

**Interfaces:**
- Consumes: custom-agent names `deep-reasoner` and `fast-worker` from Task 1.
- Produces: the project instruction that activates, routes, reviews, escalates, and falls back for the three-model team.

- [ ] **Step 1: Run the documentation contract check before expanding the section**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path

text = Path("AGENTS.md").read_text()
required = [
    "### Model Orchestration (GPT-5.6)",
    "#### Dispatch contract",
    "#### Execution and review loop",
    "#### Fallbacks",
]
missing = [heading for heading in required if heading not in text]
assert not missing, f"missing orchestration sections: {missing}"
PY
```

Expected: FAIL with all four headings listed as missing.

- [ ] **Step 2: Replace the existing `Subagent Model Routing` section**

Replace `### Subagent Model Routing` and its bullets, stopping before
`### graphify`, with:

```markdown
### Model Orchestration (GPT-5.6)

Activate this policy only when the user explicitly selects Subagent-Driven
execution. Otherwise, keep normal single-agent behaviour and do not spawn
subagents merely because the roles below are available. Project configuration
in `.codex/config.toml` pins the root task to `gpt-5.6-sol` at `xhigh`; a newly
started or reloaded Codex task is required for that setting and the custom-agent
files to take effect.

#### Roles and routing

- **Scheduler/Integrator — `gpt-5.6-sol` (`xhigh`).** The root task owns request
  interpretation, dependency-aware decomposition, routing, integration,
  conflict resolution, final verification, and acceptance. Keep the scheduler
  context focused on requirements and decisions; delegate noisy exploration,
  test output, and mechanical execution when delegation is worthwhile.
- **Deep reasoner — `deep-reasoner`, `gpt-5.6-terra` (`xhigh`).** Use for
  bounded architecture analysis, experiment design, integration reasoning,
  subtle diagnosis, task-scoped review, and decisions that require trade-off
  analysis. Terra returns evidence and a recommendation; Sol retains the final
  decision for highest-risk architecture, solver policy, unresolved ambiguity,
  and whole-branch acceptance. The project role is defined in
  `.codex/agents/deep-reasoner.toml` with a read-only default sandbox.
- **Fast worker — `fast-worker`, `gpt-5.6-luna` (`low`).** Use for
  well-specified, low-ambiguity implementation, commands and tests, renames and
  moves, boilerplate, documentation synchronization, and fact gathering. Luna
  executes decided work and stops at material design ambiguity instead of
  guessing. The project role is defined in `.codex/agents/fast-worker.toml`.

Route decisions, diagnoses, and non-mechanical review to Terra; route already
decided execution to Luna; keep cross-task integration, unresolved trade-offs,
highest-risk work, and final acceptance with Sol. The scheduler may perform a
trivial action directly when delegation overhead would exceed the work.

#### Dispatch contract

Every delegated task must state the objective, why the chosen role is
appropriate, exact file or subsystem ownership, relevant constraints and
already-made decisions, required output, and verification evidence. Always
specify the model and reasoning effort explicitly at dispatch time even when a
custom-agent file also pins them.

All agents share the same worktree. Parallelize only independent tasks with
disjoint write ownership; serialize dependencies and overlapping files. Tell
every writer that it is not alone in the codebase, must preserve unrelated
edits, and must not revert another agent's work.

#### Execution and review loop

1. Sol decomposes the request into bounded, dependency-aware tasks.
2. Terra resolves any undecided architecture or behaviour questions before
   implementation.
3. A fresh Luna implementer receives one task with explicit ownership and
   verification requirements.
4. A separate Terra pass reviews the task requirements and actual diff.
5. Important or Critical findings return for correction and re-review.
6. Sol integrates accepted tasks, resolves cross-task inconsistencies, runs or
   delegates final verification, and performs the broad final whole-branch
   review at high or xhigh reasoning.

Subagent reports are advisory until Sol checks the claimed files, evidence,
and integration state. Never report success based only on a subagent summary.

#### Fallbacks

- If the active interface has not loaded the project custom-agent names, use a
  built-in `default` or `worker` agent with the same explicit model, reasoning
  effort, and role instructions.
- If the active spawn interface rejects `gpt-5.6-luna`, invoke the verified
  fast-worker path through an ephemeral
  `codex exec -m gpt-5.6-luna` file-handoff workflow. Fall back to Terra at
  low or medium effort only when that CLI workflow is unsuitable.
- A failed subagent does not authorize discarding unrelated work or widening
  scope. Retry with a corrected bounded prompt, use the documented fallback,
  or report the blocker.
```

- [ ] **Step 3: Run the documentation contract and role-allocation checks**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path

text = Path("AGENTS.md").read_text()
required = [
    "### Model Orchestration (GPT-5.6)",
    "#### Dispatch contract",
    "#### Execution and review loop",
    "#### Fallbacks",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    ".codex/agents/deep-reasoner.toml",
    ".codex/agents/fast-worker.toml",
    "All agents share the same worktree.",
]
missing = [value for value in required if value not in text]
assert not missing, f"missing orchestration requirements: {missing}"
assert "### Subagent Model Routing" not in text
assert text.index("### Model Orchestration (GPT-5.6)") < text.index("### graphify")
print("AGENTS.md orchestration contract: PASS")
PY
```

Expected: `AGENTS.md orchestration contract: PASS`.

- [ ] **Step 4: Run integrated configuration and diff verification**

Run:

```bash
uv run python - <<'PY'
from pathlib import Path
import tomllib

root = tomllib.loads(Path(".codex/config.toml").read_text())
deep = tomllib.loads(Path(".codex/agents/deep-reasoner.toml").read_text())
fast = tomllib.loads(Path(".codex/agents/fast-worker.toml").read_text())
agents = Path("AGENTS.md").read_text()

assert root["model"] == "gpt-5.6-sol"
assert deep["model"] == "gpt-5.6-terra"
assert fast["model"] == "gpt-5.6-luna"
assert deep["name"] in agents and fast["name"] in agents
assert "Subagent-Driven" in agents
print("Integrated model orchestration: PASS")
PY
codex --strict-config features list >/dev/null
git diff --check
git status --short
```

Expected: both validation commands exit 0, the Python check prints
`Integrated model orchestration: PASS`, and `git status --short` shows only
the intended `AGENTS.md` change plus pre-existing unrelated scratchpad files.
Do not stage the scratchpad files.

- [ ] **Step 5: Commit the repository orchestration rules**

```bash
git add AGENTS.md
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: expand codex model orchestration rules"
```

Expected: the staged file list contains only `AGENTS.md`, and the commit
succeeds. No pytest or evaluator run is required because solver code and the
submission path are unchanged.

## Plan Self-Review

- Task 1 covers the Sol root configuration, Terra and Luna custom agents,
  concurrency limit, explicit model/effort settings, sandbox defaults, and
  current-session reload caveat.
- Task 2 covers opt-in activation, routing, dispatch fields, shared-worktree
  ownership, the fresh implementer/reviewer loop, correction and re-review,
  final Sol integration, and both documented fallbacks.
- Every created file has complete contents, every validation command has an
  expected result, and the two commits have disjoint file ownership.
- No solver, evaluator, checkpoint, generated output, Claude agent, or
  scratchpad file is included.
