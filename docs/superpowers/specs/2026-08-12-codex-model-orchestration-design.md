# Codex Model Orchestration Design

Date: 2026-08-12

## Goal

Define a project-scoped Codex team in which `gpt-5.6-sol` is the root
Scheduler/Integrator, `gpt-5.6-terra` is the deep-reasoning specialist, and
`gpt-5.6-luna` is the fast execution worker.  The design mirrors the useful
separation in `CLAUDE.md` while using Codex-native project configuration and
custom-agent files.

The orchestration applies only when the user explicitly selects
Subagent-Driven execution.  Outside that mode, Codex keeps normal single-agent
behaviour and does not delegate merely because custom agents are available.

## Context and Decision

`CLAUDE.md` already separates scheduling, deep reasoning, and mechanical work.
`AGENTS.md` contains a shorter Codex routing policy, but it does not yet define
the complete task handoff, escalation, ownership, review, or fallback
contracts.  The repository has no project-scoped `.codex/config.toml` or
custom Codex agent definitions.

Three approaches were considered:

1. Add prose to `AGENTS.md` and rely on the user or scheduler to select every
   model manually.  This is flexible but does not enforce the requested model
   allocation.
2. Add `AGENTS.md`, project configuration, and two custom-agent TOML files.
   This makes the requested allocation executable and keeps behavioural rules
   beside their role definitions.  This is the selected design.
3. Add only custom-agent TOML files and omit the orchestration prose.  This
   pins models but leaves routing, escalation, and review behaviour ambiguous.

Codex's project-scoped custom-agent format is used rather than copying the
Claude Markdown frontmatter format.  Each custom agent has a stable name,
description, developer instructions, model, reasoning effort, and sandbox
default.

## Configuration Architecture

### Root scheduler configuration

Create `.codex/config.toml` with:

- `model = "gpt-5.6-sol"`;
- `model_reasoning_effort = "xhigh"`;
- multi-agent execution enabled;
- at most three concurrent subagent threads, leaving the primary scheduler as
  the fourth active thread;
- Terra/high as the safe fallback for a subagent spawn that accidentally omits
  explicit model settings.

The root model is responsible for understanding the request, constructing the
dependency graph, assigning bounded work, integrating results, resolving
conflicts, and verifying the final state.  It should keep noisy exploration,
test output, and mechanical execution out of the scheduler context when
delegation is justified.

Project configuration takes effect when Codex starts or reloads a task in the
repository.  It is not expected to change the model of the already-running
task retroactively.

### Terra deep reasoner

Create `.codex/agents/deep-reasoner.toml` with:

- `name = "deep-reasoner"`;
- `model = "gpt-5.6-terra"`;
- `model_reasoning_effort = "xhigh"`;
- `sandbox_mode = "read-only"` by default.

Terra owns bounded analysis that needs judgment: architecture, solver policy,
experiment design, integration reasoning, subtle diagnosis, risk assessment,
and task-scoped review.  It returns a decision or recommendation, evidence,
assumptions, risks, and concrete next steps.  It does not silently turn an
analysis assignment into an implementation task.

For the highest-risk architecture or solver-policy decisions, unresolved
ambiguity, and the final whole-branch review, Terra reports its analysis to Sol
and Sol makes the final decision.

### Luna fast worker

Create `.codex/agents/fast-worker.toml` with:

- `name = "fast-worker"`;
- `model = "gpt-5.6-luna"`;
- `model_reasoning_effort = "low"`;
- `sandbox_mode = "workspace-write"` by default.

Luna owns well-specified, low-ambiguity execution: applying an already-decided
edit, focused implementation, running commands or tests, renames and moves,
boilerplate, documentation synchronization, and repository fact gathering.
It must preserve unrelated edits, respect its assigned file ownership, and
verify its work before reporting success.

If Luna encounters a choice that can materially change behaviour, scope, or
architecture, it stops at the decision boundary and reports the ambiguity to
Sol instead of guessing.

## Orchestration Rules

Expand `AGENTS.md` from the current short routing list into an explicit model
orchestration policy.

Every delegated task must include:

- the objective and why that role is appropriate;
- exact files or subsystem ownership;
- relevant context, constraints, and already-made decisions;
- required output or handoff format;
- verification commands or evidence expectations;
- explicit model and reasoning effort at dispatch time.

The scheduler uses the following routing rule:

- a decision, design, diagnosis, or non-mechanical review goes to Terra;
- already-decided implementation, test execution, or mechanical maintenance
  goes to Luna;
- cross-task integration, unresolved trade-offs, highest-risk work, and final
  acceptance stay with Sol.

Tasks may run in parallel only when they do not depend on one another and do
not write the same files.  The scheduler must serialize dependent work and
assign explicit ownership because all subagents share the same worktree.

## Execution and Review Loop

For Subagent-Driven implementation:

1. Sol decomposes the request into dependency-aware, bounded tasks.
2. Terra resolves architecture or behaviour questions before implementation
   when those decisions are not already fixed.
3. A fresh Luna implementer receives one task with explicit ownership and
   verification requirements.
4. A separate Terra review checks that task against its requirements and the
   actual diff.
5. Important or Critical findings return to an implementer for correction,
   followed by re-review.
6. Sol integrates all accepted tasks, resolves cross-task inconsistencies,
   runs or delegates final verification, and performs the broad whole-branch
   review at high or xhigh reasoning.

The scheduler may handle a trivial action directly when delegation overhead
would exceed the work.  It must not use the deep reasoner for rote execution
or the fast worker for an unresolved design decision.

## Failure and Fallback Behaviour

- If a custom-agent name is not available in the active Codex session, use the
  built-in `default` or `worker` type with the same explicit model, reasoning
  effort, and role instructions.  A newly started task should load the custom
  agent TOML files.
- If `gpt-5.6-luna` is rejected by the active spawn interface, use the verified
  ephemeral `codex exec -m gpt-5.6-luna` file-handoff path.  Fall back to
  Terra at low or medium effort only when that CLI path is unsuitable.
- A failed subagent does not authorize Sol to discard unrelated work or widen
  scope.  Sol either retries with a corrected bounded prompt, selects the
  documented fallback, or reports the blocker.
- Subagent output is advisory until Sol checks the claimed files, evidence,
  and integration state.

## Verification Strategy

The implementation is documentation and configuration only.  Verification
must therefore check structure and loading rather than run the solver suite:

1. Parse `.codex/config.toml` and both custom-agent files with Python's
   `tomllib` through `uv run`.
2. Run a strict Codex configuration command to detect unsupported keys.
3. Confirm the custom-agent files contain the requested models and reasoning
   efforts.
4. Review the `AGENTS.md` diff for consistency with `CLAUDE.md`, especially
   the Subagent-Driven mode gate and the fresh implementer/reviewer loop.
5. Confirm no unrelated generated files or scratchpad files enter the change.

No evaluator or pytest run is required because no solver code or submission
path changes.

## Non-Goals

- Rewriting the existing Claude agent definitions.
- Changing solver behaviour, runtime policy, checkpoints, or evaluator paths.
- Enabling unconditional proactive delegation for every task.
- Defining additional specialized agents beyond the Terra deep reasoner and
  Luna fast worker.

## Self-Review

- The three roles have non-overlapping primary responsibilities and explicit
  escalation paths.
- Model names and reasoning efforts are fixed in configuration and repeated in
  dispatch rules to prevent accidental inheritance from changing allocation.
- The design accounts for the four-thread environment and shared-worktree
  conflicts.
- The Subagent-Driven opt-in gate is explicit and consistent with the existing
  repository policy.
- Current-session discovery limitations have a bounded fallback without
  changing the intended steady-state configuration.
- There are no placeholders, case-specific solver rules, or changes outside
  model orchestration.
