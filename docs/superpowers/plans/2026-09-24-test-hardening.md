# Test Hardening (Track B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `reconstruction/`'s test suite trustworthy for the outer loop — no environmental dependency, every surviving audit finding pinned by a test that lands with its fix, and the paper's replay ≡ live claim asserted across seeds — while the strict gate stays strict.

**Architecture:** One `conftest.py` fixture removes the `generated/` dependency by pointing `see.prompts.GENERATED` at stub prompts. Six audit findings are each closed by a red-then-green pair: a test named for the claim it protects, built from `see.toy`, `see.synthetic` and `tmp_path`, plus the smallest internal fix (a guard in `DreamRSI.online`, fallback symmetry in `run_episode`, per-cell validation in `LiveQuestion`, a digest-verified `DreamRSI._deploy`, and paper-scale hard caps). A single property test replays live toy episodes on-policy. The ledger and docs land last, together with the workflow's pinned count.

**Tech Stack:** Python ≥ 3.10, pytest, ruff 0.16.8, pyright 1.1.414, pre-commit; `see/` stays standard-library only.

**Spec:** `docs/superpowers/specs/2026-09-24-test-hardening-design.md` (binding authority; the context map behind it is workflow `wf_4e4d9eaa-66a`, findings verified on `main` d788c8f).

## Global Constraints

- Branch `spec/test-hardening`, forked from `main` at d788c8f; the spec is commit 1584ab0. Never push without the owner's explicit go-ahead.
- All commands run from `reconstruction/` inside its venv (`.venv/`, with `reconstruction/.venv/bin` first on PATH); `pip install -e ".[extract,lasso,dev]"` is the documented install. The system interpreter is never used.
- The gate is strict and stays strict: `ruff format --check .`, `ruff check .` (line-length 100, `E,F,I,UP,B,RUF`), `pyright` (basic, 0 errors), `python tools/extract_listings.py --check`, `python -m pytest -q` with ZERO skips, `python tools/check_junit.py report.xml --expect N`. No `xfail`, no `skip`, no `# noqa`, no `# type: ignore`.
- Test count: 47 today → 55 after Task 8 (Task 2 +2, Task 3 +1, Task 4 +1, Task 5 +2, Task 6 +1, Task 7 +1). The count is pinned ONLY in `.github/workflows/ci.yml`; `.claude/CLAUDE.md` and `reconstruction/README.md` stop quoting it (Task 8).
- Frozen surfaces: nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes; no workdir directory (`runs/`, `trace_pool/`, `policy_dev/history/`, `deployed/`, `state.json`) is renamed; `generated/` is never committed; no new `fail_class` string; Task 4 reuses the existing `evaluator_crashed` result flag (`evaluated=False`) and its error text.
- Owner decisions binding every task: fixes land with their tests (no red tests on the branch); validation is the toy loop and `python -m see demo` only, no real-agent runs; an interrupted iteration is refused, never resumed or merged; no product cap on the grid (Listing 2's rule is two per-dimension bounds).
- Test conventions: names state the claim they protect; worlds are built from `see.toy`, `see.synthetic`, `tmp_path`, never from files in the repository; `pytest.raises(match=r"...")` uses raw strings; the property test is one function looping over seeds.
- Every paper-silent choice a fix makes gets a `reconstruction/GAPS.md` §3 row (written in Task 8, one row per Task 2–6).
- Commit messages: plain imperative sentences like the existing history (no `feat:` prefixes), **no `Co-Authored-By` or other agent-attribution trailers** (owner's rule). One commit per task; the pre-commit hooks run on every commit (ruff-check, ruff-format, pyright on `reconstruction/` changes).
- Line numbers cited in tasks are as of d788c8f; locate edits by symbol name if they have moved.
- Deviations recorded in this plan: (a) the suite-runtime target is about five seconds, not the spec's original three — Task 7's three seeds cost ~0.4 s and the suite lands near 4 s after Tasks 2–7; trimming seeds or the tamper test would weaken the pins to save a second of CI, so the spec's §5 and §7.4 were amended to five seconds in the same commit as this plan.

---

### Task 1: Prompt injection fixture — the loop tests never skip

**Files:**
- Create: `reconstruction/tests/conftest.py`
- Modify: `reconstruction/tests/test_loop.py:1-24` (drop the `pytestmark` skip, add `stub_prompts` to the `finished_loop` fixture), `reconstruction/tests/test_loop.py:97` (`test_agent_crash_is_a_failed_attempt_not_a_failed_episode` signature)

**Interfaces:**
- Consumes: nothing (first task on `spec/test-hardening`).
- Produces: `stub_prompts` — a `scope="module"` pytest fixture in `tests/conftest.py`, auto-discovered for every file under `tests/` (`pyproject.toml:25` sets `testpaths = ["tests"]`). It writes stub `exploration_prompt.md` and `policy_improvement_prompt.md` into a `tmp_path_factory`-owned directory and monkeypatches `see.prompts.GENERATED` to it for the fixture's lifetime; `see.prompts._read` stays real. Any test that drives `see.live.LiveQuestion` directly, or calls `see.loop.DreamRSI.online`/`.offline` (both build prompt text unconditionally — `see/live.py:182-189`, `see/loop.py:200-202`), must request `stub_prompts` as its own fixture parameter, not rely on another test in the same module having triggered it first. Tests that only use `see.objective.run_episode`/`see.world.ReplayQuestion` (everything in `tests/test_objective.py`) never touch `see.prompts` and never need it. Tasks 2, 4 and 5 — which add tests to `tests/test_loop.py` that call `online()`/`offline()`/`LiveQuestion` outside `finished_loop` — must add `stub_prompts` to their own new test signatures the same way this task adds it to `test_agent_crash_is_a_failed_attempt_not_a_failed_episode`.

- [ ] **Step 1: Reproduce the red state this task fixes**

No new test is added in this task (spec §8: this is the one task exempted from the red/green pair). The "red" state is the five tests currently skipping without `generated/`. Confirm it from `reconstruction/`, venv active:

```bash
mv generated /tmp/generated.bak
python -m pytest -q tests/test_loop.py -rs | tail -8
mv /tmp/generated.bak generated
```

Expected: `5 skipped`, each reported reason `run tools/extract_listings.py first (needs the paper's prompts)` — the `pytestmark` at `tests/test_loop.py:13-20` firing. (Already verified against this checkout: `generated/` is present today, `pytest --collect-only tests/` reports `47 tests collected`, and `.github/workflows/ci.yml:38` pins `--expect 47`; CLAUDE.md and the track-B context map both record `42 passed, 5 skipped` as the without-`generated/` baseline.)

- [ ] **Step 2: Write `reconstruction/tests/conftest.py`**

```python
"""Shared test fixtures for reconstruction/tests."""

import pytest

from see import prompts

EXPLORATION_PROMPT_STUB = (
    "$node_dir $history_dir $baseline_dir $eval_program $problem_file $direction_guidance\n"
)
POLICY_IMPROVEMENT_PROMPT_STUB = "{method_file} {history_dir} {trace_pool}\n"


@pytest.fixture(scope="module")
def stub_prompts(tmp_path_factory):
    """Point see.prompts.GENERATED at stub Listing 1/2 prompts so loop tests never skip.

    Writes each of exploration_prompt's six $-placeholders and
    policy_improvement_prompt's three brace placeholders exactly once, then
    monkeypatches see.prompts.GENERATED (the only place that name is read,
    see/prompts.py:15) to the stub directory. The scripted toy agents
    (see.toy.ScriptedDiscoveryAgent, see.toy.ScriptedPolicyAgent) never read
    the prompt text they are handed, so the stub content itself is never
    asserted on — only its existence and placeholder syntax matter.

    Module-scoped (not function-scoped tmp_path) so a module-scoped consumer
    like tests/test_loop.py's finished_loop fixture can request it; the
    built-in monkeypatch fixture is function-scoped only, so the patch is
    applied and undone through pytest.MonkeyPatch.context() by hand instead.
    """
    directory = tmp_path_factory.mktemp("generated")
    (directory / "exploration_prompt.md").write_text(EXPLORATION_PROMPT_STUB)
    (directory / "policy_improvement_prompt.md").write_text(POLICY_IMPROVEMENT_PROMPT_STUB)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prompts, "GENERATED", str(directory))
        yield
```

- [ ] **Step 3: Wire `tests/test_loop.py` to the fixture and delete the skip**

Replace lines 1–24 (the imports through the `finished_loop` signature) with:

```python
import json
import os

import pytest

from see.live import LiveQuestion
from see.loader import load_policy
from see.loop import DreamRSI, LoopConfig
from see.objective import run_episode
from see.pool import context_factory, load_pool
from see.toy import ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task


@pytest.fixture(scope="module")
def finished_loop(tmp_path_factory, stub_prompts):
```

This deletes the `pytestmark = pytest.mark.skipif(...)` block (old lines 13-20) outright and adds `stub_prompts` as a dependency of `finished_loop`; the fixture's body (old lines 25-42) is unchanged. `import os` stays — the file's test bodies still use `os.path.exists`, `os.readlink`, `os.path.join` and `os.listdir`.

Replace line 97 (the one test that builds its own `LiveQuestion` instead of using `finished_loop`, and so does not inherit the patch through that fixture) with:

```python
def test_agent_crash_is_a_failed_attempt_not_a_failed_episode(tmp_path, stub_prompts):
```

- [ ] **Step 4: Run the loop tests with `generated/` present**

```bash
python -m pytest -q tests/test_loop.py | tail -1
```

Expected: `5 passed`.

- [ ] **Step 5: Proof step — the loop tests run without `generated/` at all**

```bash
mv generated /tmp/generated.bak
python -m pytest -q --junitxml=/tmp/report.xml | tail -1
python tools/check_junit.py /tmp/report.xml --expect 47
mv /tmp/generated.bak generated
```

Expected: `47 passed` and `junit: 47 tests, no skips` — the same total as with `generated/` present, now independent of it. (Restore `generated/` before continuing even if a step above fails, so later tasks are not blocked.)

- [ ] **Step 6: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```

Expected: `ruff format --check .` prints `31 files already formatted` (30 today plus `tests/conftest.py`); `ruff check .` prints `All checks passed!`; pyright prints `0 errors, 0 warnings, 0 informations`; pytest ends `47 passed`.

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_loop.py
git commit -m "Add a stub-prompt fixture so the loop tests never skip

tests/conftest.py's stub_prompts fixture writes placeholder
exploration_prompt.md and policy_improvement_prompt.md into a
tmp_path_factory directory and monkeypatches see.prompts.GENERATED to it,
so LiveQuestion._run_attempt and DreamRSI.offline never hit the
FileNotFoundError generated/ guard. The module-level skipif in
tests/test_loop.py is gone; finished_loop and
test_agent_crash_is_a_failed_attempt_not_a_failed_episode now request the
fixture directly instead of relying on it being absent. Verified without
generated/ present: 47 passed, 0 skipped."
```

---

### Task 2: Interrupted iteration is refused, never merged

**Files:**
- Modify: `reconstruction/see/loop.py`. Only `DreamRSI.online` changes: the guard at lines 131–133, and the `run_dir = ...` assignment at line 138, which moves up above the guard (lines as of d788c8f). No other symbol in this file is touched. Task 5 edits `_archive`, `offline` and `_deploy`; Task 6 edits `LoopConfig`.
- Modify: `reconstruction/tests/test_loop.py`. Append at the end of the file (the last line is 125 at d788c8f; Task 1's edits shift that number). Existing tests stay in order and keep their numbers. No new imports: `json`, `os`, `pytest`, `DreamRSI`, `LoopConfig`, `ScriptedDiscoveryAgent`, `ScriptedPolicyAgent` and `make_task` are already imported.

**Interfaces:**
- Consumes: from Task 1, the pytest fixture `stub_prompts` in `reconstruction/tests/conftest.py`, which points `see.prompts.GENERATED` at placeholder prompts. Also from Task 1, the removal of the module-level `pytestmark` skip in `tests/test_loop.py`.
  - The second test needs the fixture: `LiveQuestion._run_attempt` calls `exploration_prompt` before every agent call (`see/live.py:182-193`).
  - The first test refuses before any prompt is built once the fix is in. It still requests the fixture so that its red run in Step 2 fails on the assertion and not on a missing `generated/`.
- Produces: nothing a later task calls.
  - `DreamRSI.online(t)` raises `RuntimeError(f"{dir} exists: iteration {t} was interrupted or already ran; delete it, do not merge into it")` when `runs/iterNNNN`, or failing that `trace_pool/iterNNNN`, already exists. It raises before loading the policy, evaluating the baseline, creating any directory or calling any agent. Task 8's `GAPS.md` §3 row can quote this.
  - Module-private helpers appended to `tests/test_loop.py`: `_Interrupted(BaseException)`, `_RecordingAgent(interrupt_on: int | None = None)` (its `.targets: list[str]` holds the attempt-directory basenames it was called on), and `_interruptible_loop(work: str, agent: _RecordingAgent) -> DreamRSI`. Tasks 4 and 5 also append to this file. They may reuse these names but must not redefine them.

- [ ] **Step 1: Write the failing tests**

Append to the end of `reconstruction/tests/test_loop.py`, with two blank lines after the current last line:
```python
class _Interrupted(BaseException):
    """Stands in for KeyboardInterrupt, which pytest intercepts itself."""


class _RecordingAgent:
    """The scripted discovery agent; records every call and dies on call ``interrupt_on``."""

    def __init__(self, interrupt_on: int | None = None):
        self.inner, self.interrupt_on = ScriptedDiscoveryAgent(seed=7), interrupt_on
        self.targets: list[str] = []

    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        self.targets.append(os.path.basename(target))
        if len(self.targets) == self.interrupt_on:
            raise _Interrupted(f"killed while {self.targets[-1]} was running")
        return self.inner(prompt, cwd=cwd, target=target)


def _interruptible_loop(work: str, agent: _RecordingAgent) -> DreamRSI:
    # one worker, so the agent's calls are serial: b0a0 first, then b0a1
    cfg = LoopConfig(workdir=work, max_parallelism=1, fallback_grid=(2, 1), hard_max_grid=(2, 1))
    return DreamRSI(cfg, make_task(work), agent, ScriptedPolicyAgent())


def test_interrupted_iteration_is_refused_not_merged(tmp_path, stub_prompts):
    """A partial runs/iterNNNN is never reused: online() refuses before any agent call."""
    agent = _RecordingAgent()
    loop = _interruptible_loop(str(tmp_path), agent)
    run_dir = tmp_path / "runs" / "iter0001"
    seeded = ("attempt_b000_a000", "attempt_b005_a000")  # inside and outside the 2 x 1 grid
    for node in seeded:
        (run_dir / "tree" / node).mkdir(parents=True)
        (run_dir / "tree" / node / "proposal.md").write_text("left by the killed run\n")
    state = (tmp_path / "state.json").read_text()
    assert json.loads(state)["iteration"] == 0
    with pytest.raises(RuntimeError, match=r"runs/iter0001 exists: .*delete it, do not merge"):
        loop.online(1)
    assert agent.targets == []
    assert (tmp_path / "state.json").read_text() == state
    assert not (tmp_path / "trace_pool" / "iter0001").exists()
    assert os.listdir(run_dir) == ["tree"]
    assert sorted(os.listdir(run_dir / "tree")) == list(seeded)
    for node in seeded:
        assert os.listdir(run_dir / "tree" / node) == ["proposal.md"]
        assert (run_dir / "tree" / node / "proposal.md").read_text() == "left by the killed run\n"


def test_interrupt_leaves_a_partial_run_that_a_restart_refuses(tmp_path, stub_prompts):
    """An interrupt escapes online() mid-tree; restarting that iteration refuses, never merges."""
    loop = _interruptible_loop(str(tmp_path), _RecordingAgent(interrupt_on=2))
    state = (tmp_path / "state.json").read_text()
    with pytest.raises(_Interrupted):
        loop.online(1)
    tree = tmp_path / "runs" / "iter0001" / "tree"
    assert sorted(os.listdir(tree)) == ["attempt_b000_a000", "attempt_b000_a001"]
    assert not (tmp_path / "trace_pool" / "iter0001").exists()
    assert (tmp_path / "state.json").read_text() == state
    restart_agent = _RecordingAgent()
    restart = _interruptible_loop(str(tmp_path), restart_agent)
    with pytest.raises(RuntimeError, match=r"runs/iter0001 exists: .*delete it, do not merge"):
        restart.online(1)
    assert restart_agent.targets == []
    assert (tmp_path / "state.json").read_text() == state
```

Notes for the implementer:
- `online()` never writes `state.json`. It is written only in `__init__`, in `run()` after both stages, and at the end of `offline()`. The byte-equality checks therefore pin "the counter never advances", and the fresh file has `"iteration": 0`. Do not assert the whole initial dict: Task 5 adds a digest key to `state.json`.
- `_Interrupted` is raised inside the `ThreadPoolExecutor` worker. It passes through `_run_attempt`'s `except Exception` and `online()`'s `except Exception`, is carried by the future, and is re-raised in the caller by `pool.map`. This is the same uncaught path a `KeyboardInterrupt` takes.
- With `max_parallelism=1`, parallel-refine probes `b0a0` and then `b0a1`, so call 2 is always `attempt_b000_a001`. Both attempt directories exist because `_run_attempt` creates the node before calling the agent.

- [ ] **Step 2: Run the new tests to verify they fail for the right reason**

```bash
python -m pytest -q tests/test_loop.py::test_interrupted_iteration_is_refused_not_merged tests/test_loop.py::test_interrupt_leaves_a_partial_run_that_a_restart_refuses
```
Expected: `2 failed`, each with `Failed: DID NOT RAISE RuntimeError`.
- The first test fails at its `pytest.raises`: the current guard looks only at `trace_pool/iter0001`, so `online(1)` runs the whole tree over the seeded directories.
- The second test's first half passes on unfixed code. The interrupt propagates, `runs/iter0001/tree` holds `attempt_b000_a000` and `attempt_b000_a001`, there is no `trace_pool/iter0001`, and `state.json` is unchanged. That half is the defect, and it is expected to pass. The test fails at the restart's `pytest.raises`, because the restart silently merges into the partial run.
- A `FileNotFoundError` naming `generated/` instead means Task 1's `stub_prompts` fixture is missing. Stop and fix that; do not work around it here.

- [ ] **Step 3: Make `online()` refuse a partial `runs/iterNNNN`**

In `reconstruction/see/loop.py`, `DreamRSI.online`, replace lines 131–133:
```python
        out = os.path.join(self.pool, f"iter{t:04d}")
        if os.path.exists(out):
            raise RuntimeError(f"{out} exists: an interrupted iteration must be removed, not mixed")
```
with:
```python
        run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")
        out = os.path.join(self.pool, f"iter{t:04d}")
        for partial in (run_dir, out):  # runs/ is created first, trace_pool/ last
            if os.path.exists(partial):
                raise RuntimeError(
                    f"{partial} exists: iteration {t} was interrupted or already ran; "
                    "delete it, do not merge into it"
                )
```
Then delete the now-duplicate line 138, the one directly above `tree, history = ...`:
```python
        run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")
```
After the edit, the method starts:
```python
    def online(self, t: int):
        """Stage 1: the deployed policy drives discovery; the tree is frozen into the pool."""
        run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")
        out = os.path.join(self.pool, f"iter{t:04d}")
        for partial in (run_dir, out):  # runs/ is created first, trace_pool/ last
            if os.path.exists(partial):
                raise RuntimeError(
                    f"{partial} exists: iteration {t} was interrupted or already ran; "
                    "delete it, do not merge into it"
                )
        policy = load_policy(self.state["deployed"])(None)  # baked-in default beta
        ctx = self._context(self.manifests())
        plan = validate_plan(policy.plan_grid(ctx), ctx)
        grid = (plan.branch_count, plan.refine_count) if plan else tuple(self.c.fallback_grid)
        tree, history = os.path.join(run_dir, "tree"), os.path.join(run_dir, "history")
        os.makedirs(tree, exist_ok=True)
        os.makedirs(history, exist_ok=True)
```
Everything from `for prev in range(1, t):` to the end of the method is unchanged. That includes the `try/except Exception` around `policy.solve`, the `os.makedirs(out)` freeze and the `history/` symlinks. There are no completion markers and no resume.

- [ ] **Step 4: Run the new tests**

```bash
python -m pytest -q tests/test_loop.py::test_interrupted_iteration_is_refused_not_merged tests/test_loop.py::test_interrupt_leaves_a_partial_run_that_a_restart_refuses
```
Expected: `2 passed`.

- [ ] **Step 5: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```
Expected:
- `ruff format` reformats nothing (every file left unchanged).
- `All checks passed!`
- `0 errors, 0 warnings, 0 informations`
- `49 passed`: the 47 from Task 1 plus these 2, with no skips.

The existing `finished_loop` tests still pass, because a normal `run()` never finds `runs/iterNNNN` before `online(t)`. `.github/workflows/ci.yml` still pins `--expect 47` until Task 8, so do not run `check_junit.py --expect 47` here.

- [ ] **Step 6: Commit**

```bash
git add see/loop.py tests/test_loop.py
git commit -m "Refuse to restart an iteration whose runs directory exists

online() guarded only trace_pool/iterNNNN, which it writes last, so an
iteration killed mid-tree (a KeyboardInterrupt passes through the
except Exception around solve) left runs/iterNNNN behind and a restart
silently reused it: attempts inside the new grid were overwritten and
those outside it survived, unreferenced by trace.json but visible to
later iterations through the history link. The guard now checks
runs/iterNNNN first, before any agent call, and its message names the
directory and says to delete it. Two tests pin the refusal and the
interrupt path; exception handling is unchanged."
```

---

### Task 3: A rejected plan replays on the fallback grid

**Files:**
- Modify: `reconstruction/see/objective.py:128-138`. This is `run_episode` only: the block from `plan, error, q = None, None, None` through the closing `)` of the `ReplayQuestion(...)` call. `validate_plan`, `default_context`, `Episode` and `beta_sweep` are not touched.
- Modify: `reconstruction/tests/test_objective.py:1-6` (the import block). Append one test after the last line (line 98, the end of `test_adaptive_policy_beats_the_parallel_refine_floor_on_synthetic_traces`). Do not reorder or renumber the existing tests.

**Interfaces:**
- Consumes:
  - Task 1's `tests/conftest.py::stub_prompts`, indirectly: the gate's pytest run executes the five loop tests without skipping.
  - Task 2's two new tests in `tests/test_loop.py`. The count before this task is 49.
  - Existing, unchanged signatures:
    - `see.objective.run_episode(policy, trace: Trace, context: GridPlanningContext, *, beta=None, use_plan: bool = True, max_rounds: int | None = None, beta1: float = 0.0, beta2: float = 0.0, record: bool = False) -> Episode`
    - `see.objective.validate_plan(plan: GridPlan | None, context: GridPlanningContext) -> GridPlan | None`
    - `see.policy.api.GridPlanningContext(history: tuple, fallback_branch_count: int, fallback_refine_count: int, hard_max_branch_count: int, hard_max_refine_count: int, max_parallelism: int, trace_branch_count: int | None = None, trace_refine_count: int | None = None)`
    - `see.policy.api.GridPlan(branch_count: int, refine_count: int, reason: str = "")`
    - `see.synthetic.synthetic_trace(seed: int, branches: int = 10, refine: int = 10, baseline: float = 1.0, max_parallelism: int = 10, ...) -> Trace`
    - `see.world.ReplayQuestion(trace, max_parallelism=None, branch_count=None, refine_count=None, max_rounds=None, record_episode=False)`
    - `see.policies.parallel_refine.ParallelRefine`
- Produces:
  - Behaviour of `run_episode` with `use_plan=True` (the default, and the only mode any caller uses):
    - A rejected plan, or a `plan_grid` that returns `None`, replays on `(context.fallback_branch_count, context.fallback_refine_count)`.
    - That grid goes through `ReplayQuestion`, so it is clipped to `trace.grid`, and `Episode.out_of_support` is set exactly as it would be for an explicit plan.
    - `Episode.plan` stays `None` for such an episode, mirroring the live manifest's `planned_grid: None`.
  - `use_plan=False` still replays on the trace's full support.
  - No signature changes anywhere.
  - For Task 8's `GAPS.md` §3 row: "rejected or absent plans replay on the fallback grid, clipped to trace support, mirroring the live path (`online()` uses `LoopConfig.fallback_grid`, which `context_factory` passes to replay as `fallback_*`)."
  - For Task 6, which appends to the same file: after this task, the import block of `tests/test_objective.py` is exactly:
    ```python
    import dataclasses

    import pytest

    from see.objective import attainment, beta_sweep, eq1_value, pareto_auc, run_episode, score_of
    from see.policies.parallel_refine import ParallelRefine
    from see.policies.portfolio import OptimalPolicy
    from see.policy.api import (
        GridPlan,
        GridPlanningContext,
        LLMDesignedMethod,
        SimResult,
        finalize_result,
    )
    from see.synthetic import synthetic_trace
    from see.world import Cell, Trace
    ```
    Task 6 adds `validate_plan` to the `see.objective` line (the line then exceeds 100 characters, so ruff format wraps it) and reuses `GridPlanningContext`.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_objective.py`, replace lines 1–6, from `import pytest` through `from see.policy.api import GridPlan, LLMDesignedMethod, SimResult, finalize_result`, with:
```python
import dataclasses

import pytest

from see.objective import attainment, beta_sweep, eq1_value, pareto_auc, run_episode, score_of
from see.policies.parallel_refine import ParallelRefine
from see.policies.portfolio import OptimalPolicy
from see.policy.api import (
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    finalize_result,
)
```
Lines 7–8 (`from see.synthetic import synthetic_trace`, `from see.world import Cell, Trace`) stay as they are.

Append at the end of the file:
```python


def test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace():
    # online() runs a rejected plan on the configured fallback grid; replay must do the same,
    # or a candidate whose plan is rejected is scored on a wider tree than it would see live.
    class Rejected(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(context.hard_max_branch_count + 1, 0, reason="over the hard cap")

    class Explicit(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(
                context.fallback_branch_count, context.fallback_refine_count, reason="fallback"
            )

    trace = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)
    context = GridPlanningContext(
        history=(),
        fallback_branch_count=2,
        fallback_refine_count=2,
        hard_max_branch_count=3,
        hard_max_refine_count=3,
        max_parallelism=5,
        trace_branch_count=5,
        trace_refine_count=6,
    )
    rejected = run_episode(Rejected(None), trace, context, record=True)
    explicit = run_episode(Explicit(None), trace, context, record=True)
    assert rejected.plan is None
    assert explicit.plan is not None
    assert (explicit.plan["branch_count"], explicit.plan["refine_count"]) == (2, 2)
    probed = [trace.cells[cid] for step in rejected.log for cid in step["batch"]]
    assert probed, "the fallback grid holds recorded cells, so the episode must probe some"
    assert [c.id for c in probed if c.branch >= 2 or c.attempt > 2] == []
    assert rejected == dataclasses.replace(explicit, plan=None)
```
Why the context is built inline: `default_context` sets the fallback equal to `trace.grid`, which would make the defect invisible. `synthetic_trace(0, branches=5, refine=6, max_parallelism=5)` has 28 cells, and 6 of them lie inside the 2×2 fallback grid (branches 0–1, attempts 0–2). `Rejected` asks for 4 branches against a hard cap of 3, so `validate_plan` rejects it. `Explicit` runs the same `solve` and hands over the fallback as a valid plan.

- [ ] **Step 2: Run it and confirm it fails for the right reason**

```bash
python -m pytest -q tests/test_objective.py::test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace
```
Expected: `1 failed`. The failing line is `assert [c.id for c in probed if c.branch >= 2 or c.attempt > 2] == []`, with `AssertionError: assert ['b2a0', 'b3a..., 'b4a1', ...] == []` and `Left contains 22 more items, first extra item: 'b2a0'`. The rejected plan replayed all 28 cells of the 5×6 trace instead of the 6 inside the fallback grid. It must not fail earlier, on the `plan` premises or on an import error.

- [ ] **Step 3: Replay rejected or absent plans on the fallback grid**

In `reconstruction/see/objective.py`, `run_episode`, replace lines 128–138 (from `    plan, error, q = None, None, None` through the `)` that closes `q = ReplayQuestion(`) with:
```python
    plan, error, q = None, None, None
    try:
        branch_count = refine_count = None  # use_plan=False replays on the trace's full support
        if use_plan:
            plan = validate_plan(policy.plan_grid(context), context)
            # a rejected or absent plan replays on the fallback grid, as online() runs it live
            branch_count = plan.branch_count if plan else context.fallback_branch_count
            refine_count = plan.refine_count if plan else context.fallback_refine_count
        q = ReplayQuestion(
            trace,
            branch_count=branch_count,
            refine_count=refine_count,
            max_rounds=max_rounds,
            record_episode=record,
        )
```
The rest of the function stays as it is, from `        policy.solve(q, budget=None)` onwards. That includes the `except` path's `ReplayQuestion(trace)`, which probes nothing and scores −∞.

The fallback goes through the same `ReplayQuestion` arguments as an explicit plan, so it gets the same `min(..., trace.grid)` clipping and the same `out_of_support` flag. The live path (`see/loop.py:136-137`) uses `LoopConfig.fallback_grid`, and `DreamRSI._sweep` passes that value to `see sweep --fallback`, which becomes `context_factory`'s `fallback_*`. The two paths therefore agree by construction.

- [ ] **Step 4: Run the test**

```bash
python -m pytest -q tests/test_objective.py::test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace
```
Expected: `1 passed`.

No existing test encodes the defect, so none is changed or deleted:
- `test_serial_policy_pays_the_full_parallel_penalty` (`Serial`, lines 48–63) and `test_crashing_or_illegal_policy_is_scored_minus_infinity` (`Illegal`, lines 66–77) already take the absent-plan path. Neither overrides `plan_grid`, so `LLMDesignedMethod.plan_grid` returns `None`. They replay through `beta_sweep`'s `default_context`, whose fallback equals `trace.grid`, so their grid is unchanged.
- The `ReplayQuestion(...)` calls in `tests/test_world.py` that omit `branch_count` test the constructor's contract, which this task does not touch.
- The policies in `tests/test_loop.py` always return a valid plan.

- [ ] **Step 5: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```
Expected:
- `ruff format` reformats nothing.
- `All checks passed!`
- `0 errors, 0 warnings, 0 informations`
- `50 passed`: 47 before this track, plus Task 2's two tests, plus this one. Nothing is skipped.

- [ ] **Step 6: Commit**

```bash
git add see/objective.py tests/test_objective.py
git commit -m "Replay a rejected plan on the fallback grid, as the live loop does

validate_plan returns None for a rejected or absent plan and documents
that None means the fallback grid; online() runs such a plan on
LoopConfig.fallback_grid. run_episode instead passed None to
ReplayQuestion, which grants the trace's full support, so a candidate
whose plan was rejected was scored on a wider tree than it could ever
see live. run_episode now replays on the context's fallback grid,
clipped to the trace and flagged out of support exactly as an explicit
plan is; use_plan=False keeps full support. A test pins that a rejected
plan probes nothing outside the fallback grid and scores the same as the
fallback handed over as an explicit plan."
```

---

### Task 4: One malformed evaluator result fails one cell, not the batch

**Files:**
- Modify: `reconstruction/see/live.py` in two places:
  - the import block: `import json` / `import os` at lines 17–18, and `from collections.abc import Callable` at line 23;
  - `LiveQuestion._evaluate`, lines 239–249.
  No other symbol changes.
- Modify: `reconstruction/tests/test_loop.py`. Append one test at the end of the file (line 125 at d788c8f), after Task 2's two tests. The imports stay as they are.
- Do not touch these:
  - `see/policy/observation_signal.py`: no new fail class.
  - `see/world.py` and `see/loop.py`.
  - `GAPS.md`: Task 8 writes the §3 row.

**Interfaces:**
- Consumes:
  - The `stub_prompts` pytest fixture from Task 1 (`reconstruction/tests/conftest.py`), requested by name as a test parameter. This works whatever scope Task 1 gives the fixture. It is needed because `LiveQuestion._run_attempt` calls `exploration_prompt(...)` unconditionally (`see/live.py:182`), and that function reads `generated/`.
  - The end of `tests/test_loop.py` as Task 2 leaves it. This is only for the position of the new test.
- Produces:
  - `LiveQuestion._evaluate(self, program: str) -> dict`, with its signature unchanged.
    - It returns `dict(result)` only when the evaluator returned a `collections.abc.Mapping` whose `combined_score` is a `numbers.Real` and whose `error` is a `str` or is absent or falsy.
    - Any other return goes through the existing crash branch. That branch returns `{"combined_score": 0.0, "error": "ValueError: malformed evaluator result: <repr(result)[:200]>", "evaluator_crashed": True}`.
    - So the cell gets `evaluated=False`, and `eval/score.json` records `"evaluator_crashed": true`.
    - The cell's `fail_class` is still whatever `classify_failure(error)` returns: `"code"` for the test's inputs, but it depends on the repr's content.
  - Nothing changes for the toy task, whose evaluator always returns a well-formed result. Task 7's property test is therefore unaffected.
  - Suggested wording for Task 8's GAPS.md §3 row:
    - The paper is silent on evaluator misbehaviour.
    - A result that is not a mapping with a real-number `combined_score` and a string-or-absent `error` fails its own cell through the evaluator-crashed path, and its siblings keep their results.
    - `evaluator_crashed` is a result flag that sets `evaluated=False`. It is not a fail class.

- [ ] **Step 1: Write the failing test**

Append the test below to the end of `reconstruction/tests/test_loop.py`, two blank lines after the last test. `json`, `os`, `LiveQuestion` and `make_task` are already imported (`tests/test_loop.py:1-11`), so the imports do not change.

The test runs two batches of three cells, and exactly one cell in each is malformed. These are the two return values verified to lose the whole batch on current code:
- round 1: `combined_score` is not a number;
- round 2: `error` is not a string.

Two other inputs are handled differently today:
- A missing `combined_score` does not raise. It is scored as a silent `ok` 0.0.
- A non-mapping is already caught by the `dict(...)` call inside the existing guard.

```python
def test_one_malformed_evaluator_result_fails_one_cell_not_the_batch(tmp_path, stub_prompts):
    """Garbage an evaluator returns (not raises) costs its own cell, not its siblings'."""
    task = make_task(str(tmp_path))
    real_evaluate = task.evaluate
    malformed = {  # each of these used to abort the whole batch it was evaluated in
        "attempt_b001_a000": {"combined_score": "n/a", "validity": 1.0},
        "attempt_b001_a001": {"combined_score": 3.0, "error": 42},
    }

    def malformed_on_branch_1(path: str) -> dict:
        node = os.path.basename(os.path.dirname(path))
        return malformed[node] if node in malformed else real_evaluate(path)

    task.evaluate = malformed_on_branch_1
    xs = {
        "attempt_b000_a000": 2.0,
        "attempt_b001_a000": 3.0,
        "attempt_b002_a000": 4.0,
        "attempt_b000_a001": 5.0,
        "attempt_b001_a001": 6.0,
        "attempt_b002_a001": 7.0,
    }

    def writes_a_distinct_program(prompt, *, cwd, target):
        with open(os.path.join(target, task.eval_program), "w") as f:
            json.dump({"x": xs[os.path.basename(target)]}, f)
        return {"returncode": 0}

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        task,
        writes_a_distinct_program,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=3,
        branch_count=3,
        refine_count=1,
    )
    obs = q.probe_batch(q.legal_roots())  # round 1: branch 1's score is not a number
    assert [o.cell_id for o in obs] == ["b0a0", "b1a0", "b2a0"]
    assert [(o.score, o.evaluated) for o in obs] == [(2.0, True), (0.0, False), (4.0, True)]
    assert obs[0].fail_class == obs[2].fail_class == "ok"  # the siblings keep their results
    assert obs[1].error is not None and "malformed evaluator result" in obs[1].error
    assert "'n/a'" in obs[1].error  # the error names what was malformed
    with open(tree / "attempt_b001_a000" / "eval" / "score.json") as f:
        assert json.load(f)["evaluator_crashed"] is True
    obs = q.probe_batch(q.legal_actions())  # round 2: branch 1's error is not a string
    assert [o.cell_id for o in obs] == ["b0a1", "b1a1", "b2a1"]
    assert [(o.score, o.evaluated) for o in obs] == [(5.0, True), (0.0, False), (7.0, True)]
    assert obs[0].fail_class == obs[2].fail_class == "ok"
    assert obs[1].error is not None and "'error': 42" in obs[1].error
    assert len(q.frozen("iter0001", {})) == 6  # every cell of both batches reaches the tree
```

- [ ] **Step 2: Run it and watch it fail for the right reason**

```bash
python -m pytest -q tests/test_loop.py::test_one_malformed_evaluator_result_fails_one_cell_not_the_batch
```
Expected result: `1 failed`, in round 1. The traceback ends at `see/live.py:84: ValueError` with `E       ValueError: could not convert string to float: 'n/a'`. The error is raised in `oriented_score` by this construct:

```python
    score = float(result.get("combined_score", 0.0) or 0.0)
```

The call path:
1. `_evaluate`'s `try` wraps only `return dict(self.task.evaluate(program))` (`see/live.py:243`), so the malformed dict passes through untouched.
2. `_run_attempt` then calls `score = oriented_score(t, result)` at `see/live.py:208`, outside any guard. `classify_failure(error)` at `:207` is also unguarded, and it is what round 2's `error: 42` would hit (`re.search` at `observation_signal.py:68` raises `TypeError`).
3. The exception escapes `cells = list(pool.map(lambda job: self._run_attempt(*job), jobs))` at `see/live.py:159`. That happens before `self.cells[cell.id] = cell` at `:162` runs for any cell, and after `Question.probe_batch` has already advanced `_depth` for all three branches (`see/world.py:269-270`).

The whole batch is lost, including both siblings whose evaluations succeeded.

- [ ] **Step 3: Validate the evaluator's return inside the existing per-cell guard**

In `reconstruction/see/live.py`, edit the import block:
- Add `import numbers` between `import json` and `import os` (lines 17–18).
- Change `from collections.abc import Callable` (line 23) to:
```python
from collections.abc import Callable, Mapping
```

Replace `LiveQuestion._evaluate` with the block below. The old body runs from `def _evaluate(self, program: str) -> dict:` through the closing `}` of the crash dict (lines 239–249).
```python
    def _evaluate(self, program: str) -> dict:
        lock = self._eval_lock or _NullLock()
        with lock:  # timing tasks (Lasso, kernels) are distorted by concurrent evaluation
            try:
                result = self.task.evaluate(program)
                # checked inside the guard: a bad result fails its own cell, not the whole batch
                if not (
                    isinstance(result, Mapping)
                    and isinstance(result.get("combined_score"), numbers.Real)  # numpy scalars pass
                    and isinstance(result.get("error") or "", str)  # absent or None is fine
                ):
                    raise ValueError(f"malformed evaluator result: {repr(result)[:200]}")
                return dict(result)
            except Exception as e:
                return {
                    "combined_score": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                    "evaluator_crashed": True,
                }
```

Why the check is written this way:
- It uses `numbers.Real`, not `(int, float)`. `isinstance(numpy.float32(1), (int, float))` is `False`, while the old `float(...)` path accepted numpy scalars.
- The raise lands in the existing `except`. A malformed result therefore becomes exactly the cell that a raising evaluator produces, with no new fail class and no new result key.
- The `[:200]` bound matches the agent-stderr truncation at `see/live.py:202`.
- Each of the three checks is load-bearing:
  - Without the `error` check, round 2 fails with `TypeError: expected string or bytes-like object, got 'int'` from `observation_signal.py:68`.
  - Without the `combined_score` check, round 1 fails as in Step 2.

- [ ] **Step 4: Run the test**

```bash
python -m pytest -q tests/test_loop.py::test_one_malformed_evaluator_result_fails_one_cell_not_the_batch
```
Expected: `1 passed`.

- [ ] **Step 5: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```
Expected:
- `ruff format` reports every file unchanged.
- `ruff check` prints `All checks passed!`.
- `pyright` prints `0 errors, 0 warnings, 0 informations`.
- pytest prints `51 passed`, with no skips. The count is 47, plus Task 2's 2, Task 3's 1 and this task's 1.

- [ ] **Step 6: Commit**

```bash
git add see/live.py tests/test_loop.py
git commit -m "Contain a malformed evaluator result to its own cell

LiveQuestion._evaluate now checks, inside its existing per-cell guard,
that the evaluator returned a mapping with a real-number combined_score
and a string or absent error. Anything else becomes an evaluator-crashed
cell whose error reads 'ValueError: malformed evaluator result: <repr>'.
Before, a returned {'combined_score': 'n/a'} raised in oriented_score and
a returned {'error': 42} raised in classify_failure, both outside the
guard; either escaped pool.map and discarded every sibling in the batch.
A missing combined_score was scored as a successful 0.0. numbers.Real
keeps numpy scalar scores valid. One test pins both malformations."
```
The message ends with its last line about the change. It has no `Co-Authored-By` line and no other agent-attribution trailer.

---

### Task 5: The deployed policy is the bytes that were scored

**Files:**
- Modify: `reconstruction/see/loop.py` — imports (`:24-25`, insert between them), `DreamRSI.offline` (`:190-214`; the changed lines are `:205-211`), `DreamRSI._archive` (`:216-233`), new method `DreamRSI._deploy` inserted between the two. Task 6 edits only `LoopConfig` (`:43-60`) in this file.
- Modify: `reconstruction/see/__main__.py` — imports (`:3-6`), `cmd_sweep` (`:11-49`; the changed lines are `:15` and `:30`). Task 6 edits only the `--hard-max` default in `main()` (`:88`) in this file.
- Modify: `reconstruction/tests/test_loop.py` — imports (`:1-2`), and two tests plus one helper appended at the end of the file, after Task 4's test.

**Interfaces:**
- Consumes:
  - `stub_prompts` from Task 1 (`tests/conftest.py`). It points `see.prompts.GENERATED` at stub prompts and must be module-scoped or wider, because the module-scoped `finished_loop` requests it. The new tamper test requests it by name.
  - `finished_loop` (`tests/test_loop.py`, module-scoped). It is a `DreamRSI` after `run()` with `iterations=3, versions=3`; after Task 1 it requests `stub_prompts`.
  - Tasks 2 and 4 append their tests to `tests/test_loop.py`. This task appends after them and neither renumbers nor reorders anything.
  - Existing code:
    - `DreamRSI(config, task, discovery_agent, policy_agent)`, `DreamRSI.online(t)`, `DreamRSI.offline(t) -> str`, `DreamRSI.dev_history`, `DreamRSI.w`, `DreamRSI.state`
    - `LoopConfig(workdir, iterations, versions, max_parallelism, fallback_grid, hard_max_grid, betas)`
    - `see.toy.make_task(workdir) -> TaskSpec`, `ScriptedDiscoveryAgent(seed=0)`, `ScriptedPolicyAgent(betas=(0.4, 0.8, 0.6), broken_every=0)`
- Produces (Task 8 documents these: the CLAUDE.md Architecture clause, a GAPS.md §3 row and a §7 open item):
  - `DreamRSI._deploy(self, t: int, record: dict) -> str`. `record` needs the keys `"method"` (path of the archived `method.py`), `"sha256"` and `"round"`. It returns the path of `deployed/iter{t+1:04d}.py`. When the file's current sha256 differs from `record["sha256"]` it raises `RuntimeError("<method> changed after it was scored (sha256 <now>, scored <then>): refusing to deploy it")` before any write.
  - Key `"sha256"` (64-character lowercase hex) is added to:
    - the dict `_archive` returns
    - every `state["log"][i]["offline"][j]`
    - `beta_sweep.json`, for sweeps that ran
  - Key `state["deployed_sha256"]` at the top level of `state.json`, next to `"deployed"`.

**Record shapes, before → after** (read from `loop.py:89`, `:208-212`, `:227-233`, `:278-283`, `__main__.py:29-33` and `objective.py:216-241`):
- `_archive` return: `{"round": "r0004_t02_m0", "m": 0, "method": "<workdir>/policy_dev/history/r0004_t02_m0/method.py", "valid": bool, "score": float}` → the same dict plus `"sha256": "<hex>"`. The digest is taken from the archived bytes before `_sweep` runs them.
- `state.json`, one `log[i]["offline"]` entry: `{"round", "m", "score", "valid"}` → `{"round", "m", "score", "valid", "sha256"}`. This persisted entry is where the archive record's digest can be seen.
- `state.json`, top level: `{"iteration", "round", "deployed", "log", "deployed_round"}` → the same plus `"deployed_sha256"`, which `_deploy` writes.
- `beta_sweep.json` written by `cmd_sweep`: `{"policy", "default_beta", "n_traces", "valid", "errors", "pareto", "per_beta", "eq1", "default_episode", "out_of_support", "plan_grid_override", "method"}` → the same plus `"sha256"`, the digest of `--method` taken before `load_policy` runs its module code.
- The fallback report that `_sweep` writes when the subprocess fails (`{"valid", "errors", "pareto", "eq1"}`) is unchanged. It has no digest because nothing was scored. `cmd_demo` (`__main__.py:71`) reads only `c["score"]` from the offline entries and is unaffected.

- [ ] **Step 1: Write the failing tests**

In `reconstruction/tests/test_loop.py`, add `import hashlib` to the standard-library import group, keeping it in alphabetical order. At d788c8f that places it directly above `import json` on line 1. ruff I001 enforces the order, including any imports Tasks 2 and 4 added.

Append at the very end of the file, after Task 4's test:
```python


def _sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_deployed_policy_digest_matches_the_scored_candidate(finished_loop):
    """Sec. 3's never-regress guarantee is about bytes: deploy exactly what was scored."""
    w = finished_loop.w
    with open(os.path.join(w, "state.json")) as f:
        state = json.load(f)
    for entry in state["log"]:
        chosen = next(c for c in entry["offline"] if c["round"] == entry["selected"])
        summary = os.path.join(
            w, "policy_dev", "history", chosen["round"], "proposal_results", "beta_sweep.json"
        )
        with open(summary) as f:
            swept = json.load(f)["sha256"]
        deployed = os.path.join(w, "deployed", f"iter{entry['iteration'] + 1:04d}.py")
        assert swept == chosen["sha256"] == _sha256(deployed)
    assert state["deployed_sha256"] == _sha256(state["deployed"])


def test_tampered_candidate_is_not_deployed(tmp_path, stub_prompts):
    """A candidate whose file changed after it was scored is refused, not deployed."""
    work = str(tmp_path)
    cfg = LoopConfig(
        workdir=work,
        iterations=1,
        versions=2,
        max_parallelism=2,
        fallback_grid=(2, 2),
        hard_max_grid=(4, 4),
        betas=(0.0, 1.0),
    )
    loop = DreamRSI(cfg, make_task(work), ScriptedDiscoveryAgent(seed=3), ScriptedPolicyAgent())
    loop.online(1)
    loop.offline(1)
    entry = loop.state["log"][-1]
    scored = next(c for c in entry["offline"] if c["round"] == entry["selected"])
    record = {**scored, "method": os.path.join(loop.dev_history, scored["round"], "method.py")}
    deployed_dir = os.path.join(work, "deployed")
    before = {n: _sha256(os.path.join(deployed_dir, n)) for n in os.listdir(deployed_dir)}
    state_before = json.dumps(loop.state)
    with open(record["method"], "a") as f:
        f.write("\n# edited after it was scored\n")
    with pytest.raises(RuntimeError, match=r"changed after it was scored"):
        loop._deploy(1, record)
    assert {n: _sha256(os.path.join(deployed_dir, n)) for n in os.listdir(deployed_dir)} == before
    assert json.dumps(loop.state) == state_before
```

- [ ] **Step 2: Run them and confirm they fail for the right reasons**

```bash
python -m pytest -q tests/test_loop.py -k "digest_matches_the_scored or tampered_candidate"
```
Expected: `2 failed, 8 deselected`.
- `test_deployed_policy_digest_matches_the_scored_candidate` fails with `KeyError: 'sha256'`, raised while reading the winner's `beta_sweep.json`. Nothing records a digest today, and that is the defect.
- `test_tampered_candidate_is_not_deployed` fails with `AttributeError: 'DreamRSI' object has no attribute '_deploy'`. Its online and offline run succeeds; only the deploy seam is missing.

If either test fails with `FileNotFoundError` from `see/prompts.py`, Task 1's `stub_prompts` is not in effect. Fix that first.

- [ ] **Step 3: Record the digest at archive time and verify it at deploy time, in `see/loop.py`**

In the import block, insert `import hashlib` between `import glob` (line 24) and `import json` (line 25).

Then replace lines 190–233 as of d788c8f. Task 2's guard in `online()` sits above these lines and will have shifted them, so find them by symbol. The span runs from `    def offline(self, t: int) -> str:` through the closing `        }` of `_archive`, just before `    def _sweep(`. Replace it with:
```python
    def offline(self, t: int) -> str:
        """Stages 2-3: evaluate M versions by replay over H_t and deploy the argmax."""
        floor = os.path.join(self.dev_history, "baseline")
        os.makedirs(floor, exist_ok=True)
        shutil.copy(BASELINE_POLICY, os.path.join(floor, "method.py"))
        self._sweep(os.path.join(floor, "method.py"), floor)  # re-scored on the current pool
        method_file = os.path.join(self.dev, "method.py")
        candidates = [self._archive(self.state["deployed"], t, 0)]
        for m in range(1, self.c.versions):
            shutil.copy(candidates[-1]["method"], method_file)  # revise the latest version
            prompt = policy_improvement_prompt(
                method_file=method_file, history_dir=self.dev_history, trace_pool=self.pool
            )
            run = self.policy_agent(prompt, cwd=self.dev, target=method_file)
            candidates.append(self._archive(method_file, t, m, agent_run=run))
        best = max(candidates, key=lambda c: (c["score"], -c["m"]))  # ties keep the earlier one
        deployed = self._deploy(t, best)
        self.state["log"][-1]["offline"] = [
            {k: c[k] for k in ("round", "m", "score", "valid", "sha256")} for c in candidates
        ]
        self.state["log"][-1]["selected"] = best["round"]
        self._save_state()
        return deployed

    def _deploy(self, t: int, record: dict) -> str:
        """Copy the scored candidate to deployed/ as pi_{t+1}, refusing any changed bytes."""
        with open(record["method"], "rb") as f:
            code = f.read()
        digest = hashlib.sha256(code).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError(
                f"{record['method']} changed after it was scored (sha256 {digest}, "
                f"scored {record['sha256']}): refusing to deploy it"
            )
        deployed = os.path.join(self.w, "deployed", f"iter{t + 1:04d}.py")
        with open(deployed, "wb") as f:
            f.write(code)  # the verified bytes, not a second read of the file
        self.state["deployed"], self.state["deployed_round"] = deployed, record["round"]
        self.state["deployed_sha256"] = digest
        self._save_state()
        return deployed

    def _archive(self, method_path: str, t: int, m: int, agent_run=None) -> dict:
        self.state["round"] += 1
        name = f"r{self.state['round']:04d}_t{t:02d}_m{m}"
        rdir = os.path.join(self.dev_history, name)
        os.makedirs(rdir, exist_ok=True)
        archived = os.path.join(rdir, "method.py")
        shutil.copy(method_path, archived)
        with open(archived, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()  # before the sweep runs the code
        if agent_run is not None:
            with open(os.path.join(rdir, "dev_agent.json"), "w") as f:
                json.dump(agent_run, f, indent=1, default=str)
        report = self._sweep(archived, rdir)
        return {
            "round": name,
            "m": m,
            "method": archived,
            "sha256": digest,
            "valid": report.get("valid", False),
            "score": score_of(report, self.c.objective) if report.get("valid") else float("-inf"),
        }
```
`offline()` changes in two places: the three copy lines become `deployed = self._deploy(t, best)`, and the persisted key tuple gains `"sha256"`. `_deploy` hashes and writes the same in-memory bytes, so no second read of the file can slip in between the check and the copy. `_sweep` is not touched.

- [ ] **Step 4: Carry the digest in the sweep summary, in `see/__main__.py`**

In the import block, insert `import hashlib` between `import argparse` (line 3) and `import json` (line 4).

In `cmd_sweep`, replace line 15 (`    cls = load_policy(a.method)`) with:
```python
    with open(a.method, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()  # before loading runs the code
    cls = load_policy(a.method)
```
and replace line 30 (`    report["method"] = os.path.abspath(a.method)`) with:
```python
    report["method"] = os.path.abspath(a.method)
    report["sha256"] = digest
```

- [ ] **Step 5: Run the two tests**

```bash
python -m pytest -q tests/test_loop.py -k "digest_matches_the_scored or tampered_candidate"
```
Expected: `2 passed, 8 deselected`. The tamper test takes about 0.3 s: one online round and three sweep subprocesses. The digest test reuses `finished_loop` and adds no loop run.

- [ ] **Step 6: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```
Expected:
- ruff format leaves every file unchanged
- `All checks passed!`
- `0 errors, 0 warnings, 0 informations`
- `53 passed` in about 4 s: 47, plus 2 from Task 2, 1 from Task 3, 1 from Task 4 and 2 from this task, with zero skips

- [ ] **Step 7: Commit**

```bash
git add see/loop.py see/__main__.py tests/test_loop.py
git commit -m "Deploy only the bytes that were scored

_archive records the sha256 of each archived method.py before the sweep
runs it, and the sweep summary (beta_sweep.json) carries the digest of
the file it loaded. The copy at the end of offline() moves into
DreamRSI._deploy, which reads the winner once, refuses it with a
RuntimeError when its digest no longer matches the scored one, writes
exactly the verified bytes to deployed/iterNNNN.py and records
deployed_sha256 in state.json. Two tests pin the digest chain and the
refusal."
```

---

### Task 6: Default hard caps admit no more calls than the paper

**Files:**
- Modify: `reconstruction/tests/test_objective.py`. Two changes to the import block as Task 3 leaves it: insert one line, and wrap Task 3's `from see.objective import ...` line (line 5 after Task 3, line 3 at d788c8f). Then append one test at the end of the file, after Task 3's `test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace`.
- Modify: `reconstruction/see/loop.py:50`, the `hard_max_grid: tuple = (32, 30)` field of `LoopConfig`. It moves to `:51` after Task 5's `import hashlib`.
- Modify: `reconstruction/see/__main__.py:88`, the `--hard-max` default in the `sweep` subparser. It moves to `:92` after Task 5.
- Modify: `reconstruction/scripts/run_dream_rsi.py:37`, the `--hard-max` default. No earlier task touches this file.

Line numbers drift, so locate every edit by its quoted text. Each quoted line below is unique in its file. `see/loop.py` also contains a `"--hard-max",` string literal in `_sweep` (`:260`), which is not an edit target.

**Interfaces:**
- Consumes:
  - From Task 3, the import block of `tests/test_objective.py`, which after Task 3 is exactly:
    ```python
    import dataclasses

    import pytest

    from see.objective import attainment, beta_sweep, eq1_value, pareto_auc, run_episode, score_of
    from see.policies.parallel_refine import ParallelRefine
    from see.policies.portfolio import OptimalPolicy
    from see.policy.api import (
        GridPlan,
        GridPlanningContext,
        LLMDesignedMethod,
        SimResult,
        finalize_result,
    )
    from see.synthetic import synthetic_trace
    from see.world import Cell, Trace
    ```
    This task keeps `import dataclasses`, `run_episode` and `GridPlanningContext`. Task 3's test uses the first two, and this task reuses `GridPlanningContext`.
  - From Task 3, the file's last function, `test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace`. Its last line is `    assert rejected == dataclasses.replace(explicit, plan=None)`.
  - Task 1's `stub_prompts` fixture, indirectly: the gate's pytest run executes the loop tests without skipping.
  - The suite count before this task: 53, after Tasks 1–5.
  - Existing, unchanged:
    - `see.loop.LoopConfig` (`see/loop.py:43-60`), a plain `@dataclasses.dataclass` with no `__post_init__`. Its fields include `workdir: str`, `max_parallelism: int = 10`, `fallback_grid: tuple = (10, 10)` and `hard_max_grid: tuple = (32, 30)`.
    - `see.loop.DreamRSI._context(self, history) -> GridPlanningContext` (`see/loop.py:104-106`). Its whole body is `(fb, fr), (hb, hr) = self.c.fallback_grid, self.c.hard_max_grid` followed by `return GridPlanningContext(tuple(history), fb, fr, hb, hr, self.c.max_parallelism)`.
    - `see.objective.validate_plan(plan: GridPlan | None, context: GridPlanningContext) -> GridPlan | None` (`see/objective.py:72-81`).
    - `see.policy.api.GridPlan(branch_count: int, refine_count: int, reason: str = "")` (`see/policy/api.py:45-54`).
    - `see.policy.api.GridPlanningContext(history: tuple, fallback_branch_count: int, fallback_refine_count: int, hard_max_branch_count: int, hard_max_refine_count: int, max_parallelism: int, trace_branch_count: int | None = None, trace_refine_count: int | None = None)` (`see/policy/api.py:57-72`).
- Produces:
  - `LoopConfig().hard_max_grid == (32, 19)`. Any plan `validate_plan` accepts under the default context has `branch_count * (refine_count + 1) <= 640`.
  - The `--hard-max` default is `(32, 19)` in both `python -m see sweep` and `scripts/run_dream_rsi.py`.
  - Test `tests/test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper`.
  - Suite count 54.
  - No new public name. No signature changes. `validate_plan` is untouched, so there is still no product cap.
  - For Task 8's `GAPS.md` §3 row: hard caps default to `(32, 19)`, the paper's largest reported budget (32 × 20 = 640). The per-dimension form is Listing 2's; the values are ours.

- [ ] **Step 1: Write the failing test**

Run all commands from `reconstruction/` inside the venv.

In `tests/test_objective.py`, replace Task 3's single line
```python
from see.objective import attainment, beta_sweep, eq1_value, pareto_auc, run_episode, score_of
```
with the block below. The new first line sorts before `see.objective`. With `validate_plan` added, the `see.objective` import is 109 characters, so it is written the way `ruff format` wraps it, one name per line:
```python
from see.loop import LoopConfig
from see.objective import (
    attainment,
    beta_sweep,
    eq1_value,
    pareto_auc,
    run_episode,
    score_of,
    validate_plan,
)
```
Leave every other import line exactly as Task 3 left it. The whole block must then read:
```python
import dataclasses

import pytest

from see.loop import LoopConfig
from see.objective import (
    attainment,
    beta_sweep,
    eq1_value,
    pareto_auc,
    run_episode,
    score_of,
    validate_plan,
)
from see.policies.parallel_refine import ParallelRefine
from see.policies.portfolio import OptimalPolicy
from see.policy.api import (
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    finalize_result,
)
from see.synthetic import synthetic_trace
from see.world import Cell, Trace
```

Then append the test at the end of the file. It goes after Task 3's last line, `    assert rejected == dataclasses.replace(explicit, plan=None)`, separated by two blank lines:
```python


def test_default_hard_caps_admit_no_more_calls_than_the_paper():
    # the context online() plans against: DreamRSI._context over LoopConfig's defaults
    cfg = LoopConfig(workdir="unused")
    (fallback_b, fallback_r), (hard_b, hard_r) = cfg.fallback_grid, cfg.hard_max_grid
    context = GridPlanningContext(
        history=(),
        fallback_branch_count=fallback_b,
        fallback_refine_count=fallback_r,
        hard_max_branch_count=hard_b,
        hard_max_refine_count=hard_r,
        max_parallelism=cfg.max_parallelism,
    )
    assert validate_plan(GridPlan(32, 19, reason="test"), context) is not None
    assert validate_plan(GridPlan(33, 19, reason="test"), context) is None
    assert validate_plan(GridPlan(32, 20, reason="test"), context) is None
    for branch_count in range(1, 40):
        for refine_count in range(0, 25):
            plan = validate_plan(GridPlan(branch_count, refine_count, reason="test"), context)
            if plan is not None:
                assert plan.branch_count * (plan.refine_count + 1) <= 640  # 32 x 20, Flash
```

Why the context is built this way:
- `see.objective.default_context` (`see/objective.py:84-95`) takes its hard caps from the trace's own grid, never from `LoopConfig`, so it can never see the shipped default.
- The live loop's context comes from `DreamRSI._context` (`see/loop.py:104-106`). That method is one line over `self.c.fallback_grid`, `self.c.hard_max_grid` and `self.c.max_parallelism`, and the test builds the same context from the same `LoopConfig` fields.
- Calling `_context` itself would need a `DreamRSI`, whose `__init__` creates directories, copies the initial policy and needs a task and two agents. None of that bears on a default value.
- `LoopConfig(workdir="unused")` does no I/O, so the test needs no `tmp_path`, like the file's other pure-function tests.
- The sweep over `branch_count` 1–39 and `refine_count` 0–24 runs past both caps in each dimension, so it would catch any accepted plan above 640.

- [ ] **Step 2: Run it and confirm the exact failure**

```bash
python -m pytest -q tests/test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper
```
Expected: `1 failed`, with:
```
E       AssertionError: assert GridPlan(branch_count=32, refine_count=20, reason='test') is None
tests/test_objective.py:167: AssertionError
```
The first two assertions pass under the current `(32, 30)` default, because `(32, 19)` is within bounds and `(33, 19)` exceeds the branch cap under either default. The test fails on `(32, 20)`: with a refine cap of 30, `validate_plan` returns the plan instead of `None`. It must not fail earlier, at collection or on an import. If it does, Step 1's import block is wrong. Line 167 assumes Task 3's file as published.

- [ ] **Step 3: Lower the default where it is repeated: the dataclass and both CLIs**

`see/loop.py` (`:50` at d788c8f, `:51` after Task 5), in `LoopConfig`, replace
```python
    hard_max_grid: tuple = (32, 30)  # caps plan_grid may request (unstated)
```
with
```python
    hard_max_grid: tuple = (32, 19)  # caps plan_grid may request (unstated)
```

`see/__main__.py` (`:88` at d788c8f, `:92` after Task 5), in `main`'s `sweep` subparser, replace
```python
    s.add_argument("--hard-max", type=int, nargs=2, default=(32, 30))
```
with
```python
    s.add_argument("--hard-max", type=int, nargs=2, default=(32, 19))
```

`scripts/run_dream_rsi.py` (`:37`), in `main`, replace
```python
    ap.add_argument("--hard-max", type=int, nargs=2, default=(32, 30))
```
with
```python
    ap.add_argument("--hard-max", type=int, nargs=2, default=(32, 19))
```

`validate_plan` does not change. Listing 2 states two independent per-dimension caps, not a product bound. A combined `W*(R+1)` check would diverge from the paper's own validation rule, and the spec says "no product cap". The loop's own replay scoring is unaffected by the CLI default, because `DreamRSI._sweep` passes `self.c.hard_max_grid` explicitly as `--hard-max`. The `see sweep` default matters only for manual sweeps.

- [ ] **Step 4: Run the test**

```bash
python -m pytest -q tests/test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper
```
Expected: `1 passed`. With `(32, 19)`, the largest `W*(R+1)` accepted over the sweep is exactly 640.

- [ ] **Step 5: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```
Expected:
- `ruff format` reformats nothing. It prints `31 files left unchanged`: 30 at d788c8f plus Task 1's `tests/conftest.py`.
- `All checks passed!`
- `0 errors, 0 warnings, 0 informations`
- `54 passed in …s`, which is 53 after Tasks 1–5 plus this test. Nothing is skipped.

No existing test depends on the old default. Every `LoopConfig(...)` in `tests/` (Tasks 2, 5 and 7 included) and in `python -m see demo` sets `hard_max_grid` explicitly.

- [ ] **Step 6: Commit**

```bash
git add see/loop.py see/__main__.py scripts/run_dream_rsi.py tests/test_objective.py
git commit -m "Lower the default hard grid cap to the paper's largest budget

LoopConfig.hard_max_grid defaulted to (32, 30), so a deployed policy
could plan 32 x 31 = 992 discovery calls in one online round: 1.5x the
paper's largest reported setting (32 x 20 = 640, the 3.7 Flash grid)
and 9x its default (10 x 11 = 110, which fallback_grid encodes and a
policy reaches only when its plan is absent or rejected). The default
is now (32, 19), so no plan validate_plan accepts exceeds 640 calls.
validate_plan itself is unchanged: Listing 2 states two independent
per-dimension caps, not a product bound, so the fix lowers the values
instead of adding a combined check. The --hard-max defaults of
python -m see sweep and scripts/run_dream_rsi.py follow, so the library
and both entry points agree. A test pins the default against the
640-call budget."
```
The pre-commit hooks (ruff-check, ruff-format, pyright) run on the commit and must pass unchanged.

---

---

### Task 7: Replay reproduces live episodes across seeds and both policies

**Files:**
- Create: `reconstruction/tests/test_replay_equivalence.py`

**Interfaces:**
- Consumes: Task 1's `stub_prompts` fixture in `reconstruction/tests/conftest.py`. It points `see.prompts.GENERATED` at stub prompts. The test requests it with `@pytest.mark.usefixtures("stub_prompts")`, so the fixture's scope does not matter. The test also uses these existing names, all unchanged by Tasks 2–6:
  - `see.loop.DreamRSI(config, task, discovery_agent, policy_agent, directions=None)`, with `DreamRSI.online(t: int) -> dict` (returns the live-cycle manifest), `DreamRSI.pool` and `DreamRSI.state["deployed"]`
  - `see.loop.LoopConfig` fields `workdir, max_parallelism, fallback_grid, hard_max_grid, betas, lam, beta1, beta2, max_replay_rounds, initial_policy`
  - `see.loop.BASELINE_POLICY` and `see.toy.PORTFOLIO` (the paths of the two shipped policies)
  - `see.toy.make_task(workdir) -> TaskSpec`, `see.toy.ScriptedDiscoveryAgent(seed: int = 0)` and `see.toy.ScriptedPolicyAgent()`
  - `see.pool.load_pool(pool_dir) -> list` of `(Trace, manifest)` pairs, and `see.pool.context_factory(pool, fallback, hard_max)`
  - `see.loader.load_policy(path)`
  - `see.objective.run_episode(policy, trace, context, *, beta=None, use_plan=True, max_rounds=None, beta1=0.0, beta2=0.0, record=False) -> Episode`
  - `see.objective.beta_sweep(policy_cls, traces, *, context_for, betas, lam, beta1, beta2, use_plan=True, max_rounds=None, record=True) -> (report, executions)`
  - `see.objective.eq1_value(best, root_score, probes, rounds, beta1, beta2)` and `see.objective.attainment(best, trace)`
- Produces: `tests/test_replay_equivalence.py::test_replay_reproduces_live_episodes_across_seeds_and_policies`, one collected test. It takes the suite from 54 to 55; Task 8 pins that count. No production code changes, and no later task relies on a name from this one.

- [ ] **Step 1: Write the property test**

**The freezing path is reused, not re-implemented.** The test drives `DreamRSI.online(1)` itself (`see/loop.py:129-188`), which freezes the live tree in three writes:
- `LiveQuestion.frozen(f"iter{t:04d}", {"iteration": t})` (`see/live.py:251-259`), then `Trace.save` to `trace_pool/iter0001/trace.json` (`see/loop.py:176-177`)
- `q.episode` to `live_episode.jsonl` (`:178-180`)
- `q.manifest_stats()` into the manifest (`:164-175`)

`LiveQuestion.frozen` has exactly one caller, `DreamRSI.online`. The tree is read back by `see.pool.load_pool` → `Trace.load`, the same reader `cmd_sweep` uses (`see/__main__.py:11-28`) when `DreamRSI._sweep` scores a candidate.

**Where the live numbers come from.** `online()` does not return its `LiveQuestion`, so the live side is read from what it froze:
- `budget_spent` is the sum of the batch sizes in `live_episode.jsonl`. `probe_batch` adds 1 per observation, one per batched cell (`see/world.py:272-275`). The test cross-checks this sum against `manifest["probes"]`, which is `len(q.cells)`.
- `decision_rounds` is `manifest["decision_rounds"]`.
- `best_so_far` is `manifest["best_score"]`; `manifest_stats` uses the same success predicate.
- `effective_sequential_rounds` is read from the manifest.

`reconstruction/tests/test_replay_equivalence.py`:
```python
"""Replay can stand in for live execution only if, on-policy, it repeats the live episode."""

import itertools
import json
import os

import pytest

from see.loader import load_policy
from see.loop import BASELINE_POLICY, DreamRSI, LoopConfig
from see.objective import attainment, beta_sweep, eq1_value, run_episode
from see.pool import context_factory, load_pool
from see.toy import PORTFOLIO, ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

# With this grid, seeds 0 and 1 leave the portfolio's tree partial (cells the live episode could
# have probed but never did); seed 2 and every parallel-refine episode fill the grid.
SEEDS = (0, 1, 2)
POLICIES = {"parallel_refine": BASELINE_POLICY, "portfolio": PORTFOLIO}


def _transcript(log: list) -> list:
    """Per round: the prefix the policy decided on, its batch, and what each probe revealed.

    The prefix's ``legal`` count is left out: live it counts every unprobed cell of the planned
    grid, in replay only those in the recorded tree (GAPS.md §6), so the two differ whenever
    the live tree is partial. The batches are what must agree.
    """
    return [
        (
            step["round"],
            {k: step["prefix"][k] for k in ("opened", "revealed", "best")},
            step["batch"],
            step["revealed"],
        )
        for step in log
    ]


def _first_differing_round(live: list, replay: list) -> int:
    pairs = itertools.zip_longest(live, replay)  # a missing round differs too
    return next(number for number, (a, b) in enumerate(pairs, start=1) if a != b)


@pytest.mark.usefixtures("stub_prompts")
def test_replay_reproduces_live_episodes_across_seeds_and_policies(tmp_path):
    for seed in SEEDS:
        for name, policy_file in POLICIES.items():
            where = f"seed {seed}, {name}"
            work = str(tmp_path / f"{name}-s{seed}")
            cfg = LoopConfig(
                workdir=work,
                max_parallelism=4,
                fallback_grid=(6, 5),  # more branches than workers: roots open across rounds
                hard_max_grid=(8, 8),
                betas=(0.0, 0.5, 1.0),
                beta1=0.01,  # non-zero, so Eq. (1) also depends on probes and rounds
                beta2=0.05,
                initial_policy=policy_file,
            )
            loop = DreamRSI(
                cfg, make_task(work), ScriptedDiscoveryAgent(seed=seed), ScriptedPolicyAgent()
            )
            manifest = loop.online(1)  # the live episode, frozen into trace_pool/iter0001/
            assert manifest["error"] is None, f"{where}: {manifest['error']}"
            with open(os.path.join(loop.pool, "iter0001", "live_episode.jsonl")) as f:
                live_log = [json.loads(line) for line in f]
            live = _transcript(live_log)
            assert len(live) > 1, f"{where}: the live episode should span several rounds"
            spent = sum(len(step["batch"]) for step in live_log)
            live_eq1 = eq1_value(
                manifest["best_score"],
                manifest["baseline_score"],
                spent,
                manifest["decision_rounds"],
                cfg.beta1,
                cfg.beta2,
            )

            pool = load_pool(loop.pool)  # read back exactly as `python -m see sweep` does
            assert len(pool) == 1, where
            trace = pool[0][0]
            context_for = context_factory(pool, cfg.fallback_grid, cfg.hard_max_grid)
            policy_cls = load_policy(loop.state["deployed"])
            episode = run_episode(
                policy_cls(None),
                trace,
                context_for(trace),
                max_rounds=cfg.max_replay_rounds,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
                record=True,
            )
            replay = _transcript(episode.log)
            assert replay == live, (
                f"{where}: replay departs from live at round {_first_differing_round(live, replay)}"
            )
            # exact equality: the claim is determinism, and JSON round-trips floats exactly
            assert episode.probes == spent == manifest["probes"], where
            assert episode.rounds == manifest["decision_rounds"], where
            assert episode.effective_rounds == manifest["effective_sequential_rounds"], where
            assert episode.best == manifest["best_score"], where
            assert episode.eq1 == live_eq1, where

            # the offline scorer's default-beta leg is the live policy: it must score the live run
            report, executions = beta_sweep(
                policy_cls,
                [trace],
                context_for=context_for,
                betas=cfg.betas,
                lam=cfg.lam,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
                max_rounds=cfg.max_replay_rounds,
            )
            assert report["valid"], f"{where}: {report['errors']}"
            default = _transcript(executions[-1]["log"])
            assert default == live, (
                f"{where}: beta_sweep's default episode departs from live at round "
                f"{_first_differing_round(live, default)}"
            )
            assert report["eq1"]["V"] == live_eq1, where
            assert report["default_episode"] == {
                "attainment": attainment(manifest["best_score"], trace),
                "work": spent / len(trace),
                "penalty": manifest["effective_sequential_rounds"] / spent,
            }, where
```

- [ ] **Step 2: Run it (it should pass on the first run)**

```bash
python -m pytest -q tests/test_replay_equivalence.py
```
Expected: `1 passed`.

This test pins a property the code already has; spec §5 expects no fix. While this task was drafted, a scratch copy of this exact file ran on d788c8f with a stub-prompt fixture. It passed with the full assertion set. A variant read `SEEDS`, `max_parallelism` and `fallback_grid` from environment variables and also passed with every assertion: 10 seeds × 2 policies on each of 17 (max_parallelism, fallback_grid) configurations, 340 live/replay pairs, 0 mismatches. The configurations were:
(4,(4,5)), (4,(6,5)), (3,(5,4)), (3,(8,5)), (2,(6,4)), (4,(8,3)), (5,(8,8)), (10,(8,8)), (3,(6,6)), (4,(6,6)), (2,(5,6)), (3,(4,4)), (3,(5,5)), (4,(5,5)), (2,(4,4)), (3,(4,5)), (4,(5,4)).

If the test fails, stop. Copy the seed, the policy and the round from the message into the task report and escalate. Do not narrow `SEEDS`, drop a policy, or relax an assertion.

- [ ] **Step 3: Prove the test can fail (a mutation check that you revert)**

The property already holds, so this mutation stands in for the red half of the cycle.

In `see/world.py`, replace the last line of `ReplayQuestion._execute` (line 364, `        return out`) with the line below. The earlier `return out` at line 242, in `legal_actions`, stays as it is.
```python
        return out[::-1] if self.decision_rounds == 3 else out
```
Run:
```bash
python -m pytest -q tests/test_replay_equivalence.py 2>&1 | grep -E "^E +AssertionError|failed"
```
Expected:
```
E               AssertionError: seed 0, parallel_refine: replay departs from live at round 3
1 failed in …
```
Restore the file and check the tree:
```bash
git checkout -- see/world.py
git status --short
```
Expected: exactly one line, `?? tests/test_replay_equivalence.py`.

- [ ] **Step 4: Measure the runtime**

```bash
python -m pytest -q --durations=1 tests/test_replay_equivalence.py
```
Expected: `1 passed`, with a `call` time of about 0.3–0.4 s. Put the measured value in the task report.

Measured while drafting (d788c8f, `.venv/bin/python`):

| What | Time |
|---|---|
| One live toy episode on this grid (`DreamRSI.online`) | about 55 ms |
| Its replay | 2–6 ms |
| The three-beta `beta_sweep` | under 15 ms |
| The whole test (3 seeds × 2 policies = 6 live episodes), `call` time over 6 runs | 0.31–0.42 s |
| The suite at d788c8f | `47 passed in 2.62s` |
| The suite with only this test added | `48 passed in 3.12s` |

- [ ] **Step 5: Run the full gate**

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
```
Expected:
- `… files left unchanged`
- `All checks passed!`
- `0 errors, 0 warnings, 0 informations`
- `55 passed in …s`: the 47 tests at d788c8f, plus 2, 1, 1, 2 and 1 from Tasks 2–6, plus 1 from this task

`check_junit.py --expect` stays at 47 in `ci.yml` until Task 8 moves it, so this gate leaves it out.

- [ ] **Step 6: Commit**

```bash
git add tests/test_replay_equivalence.py
git commit -m "Test that replay reproduces live episodes across seeds and policies

Offline selection is only valid if replaying a policy on its own frozen
tree repeats what it did live. One property test runs a live toy episode
through DreamRSI.online for three seeds and both shipped policies, reads
the frozen tree back the way see sweep does, and requires the replay and
beta_sweep's default-beta leg to match it round by round (batch, scores,
fail classes, parent deltas), then on probes, decision rounds, effective
rounds, best score and Eq. (1). Seeds 0 and 1 leave the portfolio tree
partial. The prefix's legal-cell count is not compared: live counts the
whole planned grid, replay only the recorded tree."
```

---

### Task 8: Ledger, docs and the pinned count

**Files:**
- Modify: `reconstruction/GAPS.md:67` (insert five §3 rows after), `reconstruction/GAPS.md:170` (append two §7 items after) — as of d788c8f
- Modify: `.claude/CLAUDE.md:27`, `.claude/CLAUDE.md:33-34`, `.claude/CLAUDE.md:50-54` — as of d788c8f (`.agents/AGENTS.md` and `.gemini/GEMINI.md` are symlinks to this file — memory `dream-rsi-agent-config-layout.md` — so no separate edit)
- Modify: `reconstruction/README.md:17`, `reconstruction/README.md:36` — as of d788c8f
- Modify: `.github/workflows/ci.yml:38` — as of d788c8f

Find every block below by its quoted text, not by the line number alone — Tasks 1–7 do not touch
these four files (their own "Ledger" notes describe *what* belongs in the ledger; §8's sequencing
puts the actual doc edit in this, the last task), so the numbers should still match, but text is
the ground truth if they don't.

**Interfaces:**
- Consumes: no code symbols — this task edits prose only. It *describes* T2's interrupted-iteration
  refusal, T3's `run_episode` fallback-grid clipping, T4's per-cell `evaluator_crashed` containment,
  T5's `_archive`/`_deploy` sha256 verification into `state.json`, and T6's lowered
  `LoopConfig.hard_max_grid`, but does not name their internal signatures or the exact JSON key T5
  adds to `state.json` — those are each task's own choice. `see/loop.py:DreamRSI.__init__` (`:63-89`)
  and `DreamRSI.run` (`:119-126`), read for this task (not modified), ground the manual check's claim
  that a refused restart writes nothing new: `__init__` only bootstraps `deployed/iter0001.py` and
  `state.json` when `state.json` is absent, and a restart against the same workdir finds it present
  and skips that branch entirely; `run` computes `t = state["iteration"] + 1`, so a restart replays
  `online(1)` and hits T2's guard before any subprocess or agent call.
- Produces: nothing — this is the plan's last task (§8: "except the first and last" are exempt from
  the red/green test pairing other tasks use). Its output is the merged ledger/doc state and the
  verification evidence (collect-only count, gate-at-55 transcript, clean demo run, manual
  interrupted-run check) that the branch's final whole-branch review and pull-request description are
  built from.

- [ ] **Step 1: Confirm the count is already 55 and the gate is red at the old pin**

  T1–T7 add eight tests net (T1 unskips five existing ones, adding none; T2 +2, T3 +1, T4 +1, T5 +2,
  T6 +1, T7 +1 new property-test function — its existing single-seed sibling
  `test_on_policy_replay_reproduces_the_live_episode` stays, uncounted as new): 47 + 8 = 55.

  ```bash
  cd /Users/controlroom/Dream-RSI/reconstruction
  . .venv/bin/activate
  python -m pytest -q --collect-only | tail -1
  ```
  Expected: the last line reads `55 tests collected in <N>s`. If it reads a different number, stop —
  a neighbouring task added, removed or parametrized a test this plan didn't account for; reconcile
  the count before continuing (per §6: "the plan fixes the exact number").

  ```bash
  python -m pytest -q --junitxml=/tmp/report.xml | tail -1
  python tools/check_junit.py /tmp/report.xml --expect 47
  ```
  Expected: `55 passed`; then `junit: tests=55 (expected 47)` printed and the command exits 1 — this
  is the task's red: the gate is still pinned to Track A's count. Steps 2–6 turn it green.

- [ ] **Step 2: Add the five `GAPS.md` §3 rows**

  Insert five rows immediately after `GAPS.md:67` (the table's last existing row, the block starting
  `| Sandboxing LLM-written policy code |`), before the blank line at `:68`:
  ```markdown
  | Interrupted iteration handling | not discussed | `online(t)` refuses when either `runs/iterNNNN` or `trace_pool/iterNNNN` already exists; the message names the directory and says to delete it, never merge into it |
  | Rejected/absent grid plan in replay | Listing 2's fallback/hard caps (for the live runner); nothing about replay under a rejected plan | `run_episode` clips a rejected or absent plan to `context.fallback_branch_count`/`fallback_refine_count`, mirroring `online()`'s live substitution of `LoopConfig.fallback_grid` |
  | Malformed (non-raising) evaluator result | not discussed | one cell's malformed result (not a mapping, a non-numeric `combined_score`, or a non-string `error`) is recorded as an unevaluated `evaluator_crashed` cell for that cell only; sibling cells in the same batch keep their real results |
  | Deploy-time code integrity | not discussed | the argmax candidate's sha256, computed when it was archived and scored, is re-verified against the file about to be copied to `deployed/iterNNNN.py`; a mismatch raises instead of deploying tampered bytes, and `state.json` records the deployed digest beside the score |
  | `hard_max_grid` default | Listing 2 names `hard_max_branch_count`/`hard_max_refine_count` (the two per-dimension caps) but gives no values | lowered to `(32, 19)`, so `branch_count × (refine_count + 1) ≤ 640`, the paper's largest reported per-round budget; the two caps stay independent per-dimension bounds, matching Listing 2's own validation rule rather than adding an unspecified product cap |
  ```

- [ ] **Step 3: Add the two `GAPS.md` §7 open items**

  Append after `GAPS.md:170` (the file's last line, `- KernelBench problem ids and a GPU for the
  kernel results.`), matching the section's existing "still needed" phrasing:
  ```markdown
  - Whether the trace-pool tamper window during `offline()` should close the same way
    `policy_dev/history/` now does: `_archive`'s sha256 (§3 above) covers only the archived
    `method.py`, not `trace_pool/iterNNNN/trace.json`, and `_sweep` (`see/loop.py:234-285`)
    replay-scores every version directly against the whole pool with no digest on any frozen tree.
  - Whether the discovery agent's filesystem view is meant to extend past the shared-proposal
    reading §4 item 4 already documents: `LiveQuestion._run_attempt` (`see/live.py:192`) runs the
    agent with `cwd=self.tree_dir`, so it can read every sibling `attempt_*` directory's actual
    program and eval output — not just the proposals Listing 1 names — including stale or
    out-of-scope ones a killed prior run left behind.
  ```

- [ ] **Step 4: Update `.claude/CLAUDE.md`**

  Replace `:27` (Commands block; keep the timing, drop the literal count):
  ```
  python -m pytest -q                   # 47 tests, ~3s
  ```
  with:
  ```
  python -m pytest -q                   # whole suite, ~4s; exact count pinned in .github/workflows/ci.yml
  ```

  Replace `:33-34` (keep only the part that stays true once T1's `stub_prompts` fixture lands — the
  suite no longer needs `generated/`, only `see demo` does):
  ```
  - Without `generated/`, `see demo` and `see/prompts.py` raise `FileNotFoundError` and 5 tests in
    `tests/test_loop.py` skip (42 passed, 5 skipped instead of 47 passed).
  ```
  with:
  ```
  - Without `generated/`, `see demo` and `see/prompts.py` raise `FileNotFoundError`; the test suite
    no longer depends on it (`tests/conftest.py`'s `stub_prompts` fixture stands in for the real
    prompts).
  ```
  (If T1 already edited this bullet when it landed, reconcile rather than double-apply — see
  open_questions.)

  Replace `:50-54` (Architecture, `offline(t)` bullet — only its last clause changes):
  ```
  2. `offline(t)`: build M policy versions (`LoopConfig.versions`; version 0 is the current policy,
     so the deployed policy never regresses on the pool). The policy-development agent edits
     `policy_dev/method.py`; each version is replay-scored over every tree in the pool in a
     subprocess with a timeout (crash or illegal batch scores −∞); the argmax is copied to
     `deployed/iterNNNN.py` and `state.json` is updated.
  ```
  with:
  ```
  2. `offline(t)`: build M policy versions (`LoopConfig.versions`; version 0 is the current policy,
     so the deployed policy never regresses on the pool). The policy-development agent edits
     `policy_dev/method.py`; each version is replay-scored over every tree in the pool in a
     subprocess with a timeout (crash or illegal batch scores −∞); the argmax's sha256 is
     re-verified against the bytes that were scored, then copied to `deployed/iterNNNN.py`;
     `state.json` records the deployed digest beside the score.
  ```
  (The interruption bullet at `:94-95` is unchanged — it already reads correctly and T2 makes it
  true.)

- [ ] **Step 5: Update `reconstruction/README.md`**

  Replace `:17` (status table, outer-loop row):
  ```
  | Outer loop: online, pool, M versions, argmax deploy | Sec. 3, Fig. 1 | implemented; tested end to end with scripted agents |
  ```
  with:
  ```
  | Outer loop: online, pool, M versions, argmax deploy | Sec. 3, Fig. 1 | implemented; tested end to end with scripted agents; replay ≡ live pinned across seeds and both policies; deploy integrity verified by digest |
  ```

  Replace `:36` (count line in the "Use" block):
  ```
  python -m pytest -q                       # 47 tests; 5 of them skip while generated/ is missing
  ```
  with:
  ```
  python -m pytest -q                       # whole suite; exact count pinned in .github/workflows/ci.yml
  ```

- [ ] **Step 6: Move the CI pin from 47 to 55**

  Replace `.github/workflows/ci.yml:38`:
  ```yaml
        - run: python tools/check_junit.py report.xml --expect 47
  ```
  with:
  ```yaml
        - run: python tools/check_junit.py report.xml --expect 55
  ```

- [ ] **Step 7: Confirm the gate is green at the new pin**

  ```bash
  python tools/check_junit.py /tmp/report.xml --expect 55
  ```
  Expected: `junit: 55 tests, no skips`, exit 0 (the same `/tmp/report.xml` from Step 1 — its
  contents didn't change, only the pin did).

- [ ] **Step 8: Run the full gate end to end (spec §7 item 1)**

  ```bash
  cd /Users/controlroom/Dream-RSI/reconstruction
  ruff check . && ruff format --check .
  pyright
  python tools/extract_listings.py --check
  python tools/extract_listings.py
  python -m pytest -q --junitxml=/tmp/report.xml | tail -1
  python tools/check_junit.py /tmp/report.xml --expect 55
  ```
  Expected: `All checks passed!`; `0 errors, 0 warnings, 0 informations`; `"drift": []` printed with
  exit 0; the extraction JSON summary; `55 passed`; `junit: 55 tests, no skips` with exit 0.

- [ ] **Step 9: Run the demo cleanly (spec §7 item 2)**

  ```bash
  rm -rf /tmp/drsi
  python -m see demo --workdir /tmp/drsi
  echo "exit=$?"
  ```
  Expected: three `iter N: grid ... probes ... rounds ... best ... -> selected ... from [...]` lines
  (`see/__main__.py:cmd_demo`'s print loop, one per default `--iterations 3`), then `exit=0`.

- [ ] **Step 10: Manual interrupted-run check (spec §7 item 3)**

  This is a manual check, not part of the automated gate — run it once by hand, not from a read-only
  or exploration agent (`_sweep`'s subprocess runs with `cwd=PKG_ROOT` and writes `__pycache__` into
  the repo tree). `kill -INT` does not work here: a background job in a non-interactive shell has
  `SIGINT` set to `SIG_IGN`, so Python never sees it; `SIGTERM` (uncaught by default) kills the
  process the same way a real interrupt would.

  ```bash
  cd /Users/controlroom/Dream-RSI/reconstruction
  rm -rf /tmp/drsi-interrupt
  python -m see demo --workdir /tmp/drsi-interrupt --iterations 3 &
  PID=$!
  until ls /tmp/drsi-interrupt/runs/iter0001/tree/attempt_* >/dev/null 2>&1; do sleep 0.02; done
  kill -TERM "$PID"
  wait "$PID" 2>/dev/null
  test -d /tmp/drsi-interrupt/runs/iter0001/tree && test ! -e /tmp/drsi-interrupt/trace_pool/iter0001 \
    && echo "precondition OK: partial runs/iter0001, no trace_pool/iter0001"
  ```
  Expected: `precondition OK: ...` prints. If it doesn't, the kill landed too early or too late (the
  wait loop should prevent this, but rerun if `trace_pool/iter0001` exists — the kill was too late).

  ```bash
  find /tmp/drsi-interrupt -type f | sort > /tmp/before.txt
  python -m see demo --workdir /tmp/drsi-interrupt --iterations 3
  echo "exit=$?"
  find /tmp/drsi-interrupt -type f | sort > /tmp/after.txt
  diff /tmp/before.txt /tmp/after.txt
  ```
  Expected: a traceback ending in `RuntimeError: ...` whose message names `runs/iter0001` (T2 owns
  the exact wording; the claim pinned here is only that it names `runs/iter0001`, not
  `trace_pool/iter0001`, and says to delete it rather than merge — per T2's §4.1 fix); `exit=1`;
  `diff` prints nothing (`DreamRSI.__init__` finds `state.json` already present from the interrupted
  run and skips its bootstrap branch entirely, and the guard raises before `online(1)` creates or
  writes anything else).

  ```bash
  rm -rf /tmp/drsi-interrupt/runs/iter0001
  python -m see demo --workdir /tmp/drsi-interrupt --iterations 3
  echo "exit=$?"
  ```
  Expected: the run proceeds normally, printing all three `iter N: ...` lines, `exit=0`.

- [ ] **Step 11: Commit**

  ```bash
  cd /Users/controlroom/Dream-RSI
  git add reconstruction/GAPS.md .claude/CLAUDE.md reconstruction/README.md .github/workflows/ci.yml
  git commit -m "$(cat <<'EOF'
  Update the ledger, docs and CI count for track B's fixes

  reconstruction/GAPS.md gets five new §3 rows, one per finding fixed in this
  track (interrupted-iteration refusal, replay's fallback-grid clipping, one
  malformed evaluator result failing one cell instead of the batch, deploy-time
  digest verification, and the lowered hard_max_grid default), plus two new §7
  open items carried over from the design's non-goals: the trace-pool tamper
  window during offline() and the discovery agent's filesystem view of sibling
  attempt directories.

  .claude/CLAUDE.md and reconstruction/README.md stop pinning the literal test
  count in prose and point at .github/workflows/ci.yml instead; the
  Architecture section's offline() bullet notes that the deployed candidate's
  sha256 is re-verified before the copy and recorded in state.json.
  .github/workflows/ci.yml's check_junit --expect moves from 47 to 55, the
  eight tests this track adds.
  EOF
  )"
  ```
  (No `Co-Authored-By` or other agent-attribution trailer — the owner's rule, carried from Track A.)

---
