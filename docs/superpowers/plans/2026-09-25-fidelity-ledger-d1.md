# Fidelity and Ledger D1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every reconstruction choice that affects a published number has a GAPS §3 row with a "How to change" cell and a hand-computed test, and the four places where the code contradicts its own ledger are fixed.

**Architecture:** Four small fixes in `see/live.py`, `see/loop.py`, `see/objective.py` and the runner, each with its test; one new pin module `tests/test_ledger.py` plus extensions of three existing test modules; then the GAPS §3 rewrite (a fourth column, five new rows, the §1 split) and the README/CLAUDE.md/CI follow-through. No new runtime module, no frozen surface touched, no number the ledger already publishes changes.

**Tech Stack:** Python 3.10+ standard library; pytest; ruff; pyright (basic). Everything runs from `reconstruction/` inside its venv (`. .venv/bin/activate`).

**Spec:** `docs/superpowers/specs/2026-09-25-fidelity-ledger-d1-design.md`

## Global Constraints

- Gate stays strict: `ruff format --check .`, `ruff check .`, `pyright` basic with 0 errors, `python tools/extract_listings.py --check`, `python -m pytest -q` with zero skips, `tools/check_junit.py --expect N`.
- No `xfail`, `skip`, `# noqa`, `# type: ignore`.
- Nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes; no new `fail_class` string (the timed-out fix reuses `timeout`).
- No workdir directory is renamed; the agent callable's signature `agent(prompt, *, cwd, target) -> dict` is unchanged.
- Every paper-silent choice gets a GAPS §3 row; pins are hand-computed cases on the toy world, never expectations copied from a run; existing pins are extended, not duplicated.
- One commit per task, plain imperative messages, no attribution trailers; fixes land with their tests, shown failing on the previous tree (stash the source file, run, pop).
- Test count chain: 82 today → 83 after Task 1 → 85 after Task 2 → 87 after Task 3 → 88 after Task 4 → 96 after Task 5 → 96 after Tasks 6 and 7 → Task 8 pins 96 in `.github/workflows/ci.yml`.
- Run everything from `reconstruction/`: `cd reconstruction && . .venv/bin/activate`. `python -m pytest -q` takes about 12 s; the whole suite runs before every commit.

## File structure

- `see/live.py` — Task 1 (`_run_attempt`: the timed-out error), Task 4 (`LiveQuestion.fault`, `_cancel`, `_execute`).
- `see/objective.py` — Task 2 (`OBJECTIVES`, `score_of`).
- `see/loop.py` — Task 2 (`LoopConfig.__post_init__`), Task 3 (`run`, `offline`, `_deploy`), Task 4 (`online`).
- `scripts/run_dream_rsi.py` — Task 2 (`--objective` choices).
- `tests/test_command_agent.py` — Task 1. `tests/test_objective.py` — Tasks 2 and 5. `tests/test_loop.py` — Tasks 3 and 4. `tests/test_ledger.py` (new) — Task 5. `tests/test_replay_equivalence.py` — Task 6.
- `reconstruction/GAPS.md` — Task 7. `reconstruction/README.md`, `.claude/CLAUDE.md`, `.github/workflows/ci.yml` — Task 8.

---

### Task 1: A timed-out attempt is recorded as a `timeout` failure

**Files:**
- Modify: `reconstruction/see/live.py:368-369` (in `_run_attempt`, the two lines `error = result.get("error")` and `fail_class = ...`)
- Test: `reconstruction/tests/test_command_agent.py` (`test_agent_timeout_kills_the_whole_process_group`, lines 41-77, plus one new test after it)

**Interfaces:**
- Consumes: `CommandAgent.__call__` returns `{"returncode": None, "stdout": "", "stderr": "agent timed out after {timeout}s", "timed_out": True}` on a timeout (`see/live.py`); `classify_failure` maps the text "timed out" to `"timeout"` (`see/policy/observation_signal.py:34`).
- Produces: a `Cell` for a timed-out attempt has `error == "agent timed out after Ns"`, `fail_class == "timeout"`, `evaluated == True` when the evaluator ran, and its score; `eval/score.json` carries `fail_class: "timeout"` beside `agent_timed_out: true`; `error.txt` holds the text. Task 7's GAPS row and Task 8's README row describe this.

- [ ] **Step 1: Change the existing test's expectation and add the broken-program test**

In `reconstruction/tests/test_command_agent.py`, change the import line `from see.toy import make_task` to:

```python
from see.toy import PROGRAM, make_task
```

Replace the whole of `test_agent_timeout_kills_the_whole_process_group` (lines 41-77, from its `def` to its last `assert`) so it reads:

```python
def test_agent_timeout_kills_the_whole_process_group(tmp_path, stub_prompts, process_gone):
    """A timed-out agent and everything it forked are dead when the call returns; through
    LiveQuestion the attempt is scored as the program the agent left but recorded as the
    `timeout` failure it is, never as a valid non-improving attempt."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=0.3, kill_grace=0.2)
    started = time.time()
    result = agent("ignored prompt", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 3.0  # the child slept for 30 s; the group was killed
    assert result == {
        "returncode": None,
        "stdout": "",
        "stderr": "agent timed out after 0.3s",
        "timed_out": True,
    }
    [pid] = grandchild_pids(pid_dir)
    assert process_gone(pid)
    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        agent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=1,
        branch_count=1,
        refine_count=0,
    )
    [obs] = q.probe_batch(q.legal_roots())
    # the stand-in never touched the program, so the parent's copy (x=1.0) is what gets scored,
    # but the attempt is a timeout failure: not a success, and it never raises the ceiling
    assert (obs.cell_id, obs.score, obs.evaluated, obs.fail_class) == ("b0a0", 1.0, True, "timeout")
    assert obs.error == "agent timed out after 0.3s"
    assert obs.delta_vs_parent is None and obs.delta_vs_baseline is None  # not a success
    node = tree / "attempt_b000_a000"
    with open(node / "eval" / "score.json") as f:
        score = json.load(f)
    assert score["agent_timed_out"] is True and score["agent_returncode"] is None
    assert score["fail_class"] == "timeout" and score["combined_score"] == 1.0
    assert (node / "error.txt").read_text() == "agent timed out after 0.3s"
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)
```

Then add, directly after that test:

```python
def test_a_timed_out_agent_that_left_a_broken_program_keeps_the_evaluators_verdict(tmp_path):
    """When the evaluator itself rejects what a timed-out agent left, its error and class win:
    the program is broken either way, and score.json's agent_timed_out keeps the cause."""
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tree / "attempt_b000_a000" / PROGRAM
    writes_then_hangs = ["sh", "-c", 'printf "{broken" > "$0"; sleep 30', str(target), "{prompt}"]
    agent = CommandAgent(writes_then_hangs, timeout=0.3, kill_grace=0.2)
    q = LiveQuestion(
        make_task(str(tmp_path)),
        agent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=1,
        branch_count=1,
        refine_count=0,
    )
    [obs] = q.probe_batch(q.legal_roots())
    assert obs.fail_class == "code" and obs.evaluated and obs.score == 0.0
    assert obs.error is not None and obs.error.startswith("ValueError: unreadable solution")
    with open(tree / "attempt_b000_a000" / "eval" / "score.json") as f:
        assert json.load(f)["agent_timed_out"] is True
```

- [ ] **Step 2: Run the two tests to verify they fail**

Run: `python -m pytest -q tests/test_command_agent.py -k "timeout_kills or broken_program"`
Expected: `1 failed, 1 passed` — the first test fails on `("b0a0", 1.0, True, "ok") != (..., "timeout")`; the broken-program test already passes (the evaluator's verdict wins today too) and stays as the pin for that half of the rule.

- [ ] **Step 3: Set the timed-out error in `_run_attempt`**

In `reconstruction/see/live.py`, replace:

```python
        error = result.get("error")
        fail_class = "no_program" if result.get("no_program") else classify_failure(error)
```

with:

```python
        error = result.get("error")
        if run and run.get("timed_out") and not error:
            # the agent was killed at its timeout: what it left is scored, but the attempt is
            # the failure the taxonomy already names (classify_failure maps this text to timeout)
            error = run["stderr"]
        fail_class = "no_program" if result.get("no_program") else classify_failure(error)
```

- [ ] **Step 4: Run the tests to verify they pass, then the suite**

Run: `python -m pytest -q tests/test_command_agent.py -k "timeout_kills or broken_program"`
Expected: `2 passed`
Run: `python -m pytest -q`
Expected: `83 passed`

- [ ] **Step 5: Commit**

```bash
git add see/live.py tests/test_command_agent.py
git commit -m "Record a timed-out attempt as a timeout failure"
```

---

### Task 2: Objective names are validated before any budget is spent

**Files:**
- Modify: `reconstruction/see/objective.py:32-33` (after `DEFAULT_LAMBDA`) and `:250-252` (`score_of`)
- Modify: `reconstruction/see/loop.py:37` (the `see.objective` import) and `:65-83` (`LoopConfig`)
- Modify: `reconstruction/scripts/run_dream_rsi.py:17` (imports) and `:39` (`--objective`)
- Test: `reconstruction/tests/test_objective.py` (two new tests at the end)

**Interfaces:**
- Consumes: `LoopConfig.objective: str` (`see/loop.py:79`); `score_of(report, objective)` (`see/objective.py:250`), whose only loop caller is `_archive` (`see/loop.py:318`).
- Produces: `see.objective.OBJECTIVES == ("pareto", "eq1")`; `LoopConfig(...)` raises `ValueError` for any other name; `score_of` raises `ValueError` for any other name. Task 7's row 3 names `OBJECTIVES`.

- [ ] **Step 1: Write the failing tests**

Append to `reconstruction/tests/test_objective.py`:

```python
def test_a_mistyped_objective_fails_before_any_budget_is_spent():
    """A name that is not one of OBJECTIVES used to fall through to Eq. (1) silently at scoring
    time, after the offline phase had spent its budget; it is refused when the config is built."""
    with pytest.raises(ValueError, match=r"'eq2' is not one of \('pareto', 'eq1'\)"):
        LoopConfig(workdir="unused", objective="eq2")
    assert LoopConfig(workdir="unused", objective="eq1").objective == "eq1"
    assert LoopConfig(workdir="unused").objective == "pareto"


def test_score_of_selects_the_named_statistic_and_refuses_others():
    report = {"pareto": {"reward": 0.25}, "eq1": {"V": 7.0}}
    assert score_of(report, "pareto") == 0.25
    assert score_of(report, "eq1") == 7.0
    with pytest.raises(ValueError, match=r"unknown objective 'auc'"):
        score_of(report, "auc")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest -q tests/test_objective.py -k "mistyped or refuses_others"`
Expected: `2 failed` — `DID NOT RAISE` in both (today `LoopConfig` accepts any string and `score_of("auc")` returns 7.0).

- [ ] **Step 3: Add `OBJECTIVES`, validate in `LoopConfig`, refuse in `score_of`, wire the runner**

In `reconstruction/see/objective.py`, after the line `DEFAULT_LAMBDA = 0.1`, add:

```python
OBJECTIVES = ("pareto", "eq1")  # Listing 2's pareto.reward, Sec. 3's Eq. (1); GAPS §3
```

and replace `score_of`:

```python
def score_of(report: dict, objective: str) -> float:
    """Selection statistic: ``"pareto"`` (Listing 2) or ``"eq1"`` (Sec. 3)."""
    if objective == "pareto":
        return report["pareto"]["reward"]
    if objective == "eq1":
        return report["eq1"]["V"]
    raise ValueError(f"unknown objective {objective!r}: choose one of {OBJECTIVES}")
```

In `reconstruction/see/loop.py`, change the import line to:

```python
from see.objective import DEFAULT_BETAS, DEFAULT_LAMBDA, OBJECTIVES, score_of, validate_plan
```

and add to `LoopConfig`, after its last field (`serialize_eval: bool = True`):

```python

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective {self.objective!r} is not one of {OBJECTIVES}")
```

In `reconstruction/scripts/run_dream_rsi.py`, add after the `from see.loop import ...` line:

```python
from see.objective import OBJECTIVES
```

and change the `--objective` argument to:

```python
    ap.add_argument("--objective", choices=OBJECTIVES, default="pareto")
```

- [ ] **Step 4: Run the tests, the linters and the suite**

Run: `python -m pytest -q tests/test_objective.py -k "mistyped or refuses_others"`
Expected: `2 passed`
Run: `ruff check . && ruff format --check . && pyright`
Expected: all clean (the import order above is what ruff's isort wants: constants before functions)
Run: `python -m pytest -q`
Expected: `85 passed`

- [ ] **Step 5: Commit**

```bash
git add see/objective.py see/loop.py scripts/run_dream_rsi.py tests/test_objective.py
git commit -m "Refuse an unknown objective name when the config is built"
```

---

### Task 3: One `state.json` write per iteration, after the deploy

**Files:**
- Modify: `reconstruction/see/loop.py:144-151` (`run`), `:252-274` (`offline`), `:276-292` (`_deploy`)
- Test: `reconstruction/tests/test_loop.py` (two new tests, inserted before `def _sha256`)

**Interfaces:**
- Consumes: `DreamRSI._save_state()` (`see/loop.py:116-118`); `_deploy` sets `state["deployed"]`, `state["deployed_round"]`, `state["deployed_sha256"]`; `offline` sets `state["log"][-1]["offline"]` and `["selected"]`; `run` sets `state["iteration"]`.
- Produces: `offline(t)` sets `state["iteration"] = t` and is the only writer of `state.json` after `__init__`; `_deploy` and `run` no longer save. Task 7's deploy-integrity row says so.

- [ ] **Step 1: Write the failing tests**

In `reconstruction/tests/test_loop.py`, insert before `def _sha256(path: str) -> str:`:

```python
def test_state_is_written_once_per_iteration_after_the_deploy(tmp_path, stub_prompts, monkeypatch):
    """The deployed policy, its digest, the offline log and the iteration counter reach disk in
    one write at the end of offline(); online() writes nothing."""
    loop = _interruptible_loop(str(tmp_path), _RecordingAgent())
    writes: list[str] = []
    real_save = loop._save_state

    def counting_save():
        writes.append(json.dumps(loop.state))
        real_save()

    monkeypatch.setattr(loop, "_save_state", counting_save)
    loop.online(1)
    assert writes == []
    loop.offline(1)
    assert len(writes) == 1
    with open(tmp_path / "state.json") as f:
        state = json.load(f)
    assert state["iteration"] == 1
    assert state["deployed"].endswith("iter0002.py")
    assert state["deployed_sha256"] == _sha256(state["deployed"])
    assert set(state["log"][-1]) >= {"iteration", "live", "offline", "selected"}


def test_a_crash_after_the_deploy_leaves_the_previous_state_on_disk(
    tmp_path, stub_prompts, monkeypatch
):
    """offline() persists nothing until its end, so a crash between the deploy and that write
    leaves state.json as the previous iteration left it (deployed/iter0001.py, iteration 0),
    the state the restart guard assumes; deployed/iter0002.py exists but nothing points at it."""
    loop = _interruptible_loop(str(tmp_path), _RecordingAgent())
    loop.online(1)
    before = (tmp_path / "state.json").read_text()
    real_deploy = loop._deploy

    def deploy_then_crash(t, record):
        real_deploy(t, record)
        raise RuntimeError("host fault right after the deploy")

    monkeypatch.setattr(loop, "_deploy", deploy_then_crash)
    with pytest.raises(RuntimeError, match=r"right after the deploy"):
        loop.offline(1)
    assert (tmp_path / "deployed" / "iter0002.py").exists()
    assert (tmp_path / "state.json").read_text() == before
```

- [ ] **Step 2: Run them to verify they fail**

Run: `python -m pytest -q tests/test_loop.py -k "written_once or crash_after_the_deploy"`
Expected: `2 failed` — the first on `assert len(writes) == 1` (today two writes: `_deploy` and the end of `offline`) and on `state["iteration"] == 1` (today 0 until `run()` saves); the second on `read_text() == before` (today `_deploy` has already saved).

- [ ] **Step 3: Move the write to the end of `offline()`**

In `reconstruction/see/loop.py`, replace `run`:

```python
    def run(self, iterations: int | None = None):
        for _ in range(iterations or self.c.iterations):
            t = self.state["iteration"] + 1
            self.online(t)
            self.offline(t)  # persists the counter together with what it deployed
        return self.state
```

Replace the tail of `offline` (from `self.state["log"][-1]["selected"] = best["round"]` to `return deployed`):

```python
        self.state["log"][-1]["selected"] = best["round"]
        self.state["iteration"] = t
        self._save_state()  # the only write per iteration: deploy, digest, log and counter together
        return deployed
```

In `_deploy`, delete the line `self._save_state()` (the one after `self.state["deployed_sha256"] = digest`) and change its docstring to:

```python
        """Copy the scored candidate to deployed/ as pi_{t+1}, refusing any changed bytes.

        The state is persisted by offline() at its end, together with the iteration counter.
        """
```

- [ ] **Step 4: Run the tests and the suite**

Run: `python -m pytest -q tests/test_loop.py -k "written_once or crash_after_the_deploy"`
Expected: `2 passed`
Run: `python -m pytest -q`
Expected: `87 passed` (the `finished_loop` tests still see `state["iteration"] == 3` after `run()`, now set by `offline`)

- [ ] **Step 5: Commit**

```bash
git add see/loop.py tests/test_loop.py
git commit -m "Write state.json once per iteration, after the deploy"
```

---

### Task 4: A fault the policy swallows still reaches the manifest

**Files:**
- Modify: `reconstruction/see/live.py:229` (`LiveQuestion.__init__`, next to `self._cancelled`), `:293-300` (`_execute`'s two `_cancel` calls), `:311-321` (`_cancel`)
- Modify: `reconstruction/see/loop.py:204-219` (`online`, after the `try`/`except`)
- Test: `reconstruction/tests/test_loop.py` (one new test after `test_a_fault_while_recording_a_batch_keeps_none_of_it`)

**Interfaces:**
- Consumes: `LiveQuestion._cancel(pool, close)` (`see/live.py:311`), called from `_execute` with the caught exception in scope as `e`; `DreamRSI.online` builds the manifest with `error` (`see/loop.py:219`).
- Produces: `LiveQuestion.fault: str | None`, set to `"<Type>: <message>"` by `_cancel`; `_cancel(pool, close, fault)` takes the exception; `online()` records `error = f"batch abandoned: {q.fault}"` when `solve` returned normally with a fault recorded. Task 7's lifetime row says so.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_loop.py`, insert after `test_a_fault_while_recording_a_batch_keeps_none_of_it`:

```python
SWALLOWS = '''"""A policy that hides the batch's fault: for the manifest test only."""

from see.policy.api import GridPlan, LLMDesignedMethod, SimResult, finalize_result

NAME = "Swallows"


class Swallows(LLMDesignedMethod):
    NAME = NAME

    def solve(self, question, budget=None):
        question.reset()
        try:
            question.probe_batch(question.legal_roots())
        except Exception:
            pass
        return finalize_result(question, SimResult())

    def plan_grid(self, context):
        return GridPlan(context.fallback_branch_count, context.fallback_refine_count, reason="t")
'''


def test_a_fault_the_policy_swallows_still_reaches_the_manifest(tmp_path, stub_prompts, monkeypatch):
    """A policy that catches the batch's exception and returns cannot produce a clean-looking
    truncated cycle: the manifest names the fault, and the cycle is frozen with that error just
    as when the fault propagates. The fault here is a missing prompt file, raised in the worker
    before any agent call."""
    policy = tmp_path / "swallows.py"
    policy.write_text(SWALLOWS)
    cfg = LoopConfig(
        workdir=str(tmp_path),
        max_parallelism=2,
        fallback_grid=(2, 1),
        hard_max_grid=(2, 1),
        initial_policy=str(policy),
    )
    agent = _RecordingAgent()
    loop = DreamRSI(cfg, make_task(str(tmp_path)), agent, ScriptedPolicyAgent())
    monkeypatch.setattr(see.prompts, "GENERATED", str(tmp_path / "no-such-dir"))
    manifest = loop.online(1)
    assert manifest["error"] is not None
    assert manifest["error"].startswith("batch abandoned: FileNotFoundError: ")
    assert manifest["probes"] == 0 and agent.targets == []
    with open(tmp_path / "trace_pool" / "iter0001" / "live_cycle_manifest.json") as f:
        assert json.load(f)["error"] == manifest["error"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest -q tests/test_loop.py -k "policy_swallows"`
Expected: `1 failed` on `assert manifest["error"] is not None` (today the manifest says `None`).

- [ ] **Step 3: Record the fault and surface it**

In `reconstruction/see/live.py`, in `LiveQuestion.__init__`, after the `self._cancelled = threading.Event()` line add:

```python
        self.fault: str | None = None  # why the batch in flight was abandoned, once one was
```

Replace `_cancel`:

```python
    def _cancel(
        self, pool: concurrent.futures.ThreadPoolExecutor, close: bool, fault: BaseException
    ) -> None:
        """Abandon the batch: queued attempts never start and running agents are killed.

        ``close`` (an interrupt) also refuses every later call. A worker fault leaves the agent
        usable, because the loop goes on to offline() and to the next iteration with it. The
        fault is kept so online() can record it even when the policy swallows the exception.
        """
        self.fault = f"{type(fault).__name__}: {fault}"
        self._cancelled.set()
        pool.shutdown(wait=False, cancel_futures=True)
        stop = getattr(self.agent, "terminate" if close else "kill_running", None)
        if callable(stop):
            stop()
```

In `_execute`, change the two calls: `self._cancel(pool, close=not isinstance(e, Exception))` becomes `self._cancel(pool, close=not isinstance(e, Exception), fault=e)`, and in the `finally` block change `except BaseException:` to `except BaseException as e:` with the call `self._cancel(pool, close=True, fault=e)`.

In `reconstruction/see/loop.py`, in `online`, after the whole `try`/`except` statement (before `manifest = self._manifest(t, policy, plan, grid, q, error, started)`), add:

```python
        if error is None and q.fault is not None:  # the policy swallowed the batch's fault
            error = f"batch abandoned: {q.fault}"
```

- [ ] **Step 4: Run the test, the linters and the suite**

Run: `python -m pytest -q tests/test_loop.py -k "policy_swallows"`
Expected: `1 passed`
Run: `ruff check . && ruff format --check . && pyright`
Expected: clean
Run: `python -m pytest -q`
Expected: `88 passed`

- [ ] **Step 5: Commit**

```bash
git add see/live.py see/loop.py tests/test_loop.py
git commit -m "Record a swallowed batch fault in the manifest"
```

---

### Task 5: The ledger pins (`tests/test_ledger.py`) and the flat-trace case

**Files:**
- Create: `reconstruction/tests/test_ledger.py`
- Modify: `reconstruction/tests/test_objective.py:36-42` (`test_attainment_normalises_between_root_and_ceiling`)

**Interfaces:**
- Consumes: `LoopConfig` defaults (`fallback_grid=(10, 10)`, `max_parallelism=10`); `LiveQuestion(task, agent, tree_dir, history_dir, baseline_score, max_parallelism, branch_count, refine_count)`; `ParallelRefine`, `OptimalPolicy(config)` with `config["beta"]` (default 0.6); `beta_sweep(policy_cls, traces, *, betas, beta1, beta2, context_for)` returning `(report, executions)` with `executions` as dicts in order: every beta over every trace, then one default-beta episode per trace; `run_episode(policy, trace, context, record=True) -> Episode`; `synthetic_trace(seed, branches, refine, max_parallelism)` with `trace_id == f"synthetic-{seed}"`; `Trace.ceiling()`; `DreamRSI(cfg, task, discovery_agent, policy_agent)` copying `cfg.initial_policy` to `deployed/iter0001.py` on a fresh workdir.
- Produces: eight tests, one per ruling; Task 7's rows cite them by name.

- [ ] **Step 1: Write the pins**

Create `reconstruction/tests/test_ledger.py`:

```python
"""Pins for the reconstruction choices GAPS.md §3 records.

Each test is a hand-computed case for one ruling, named for the claim it protects, so the
ledger cannot drift from the code without a test going red.
"""

import os

import pytest

from see.live import LiveQuestion
from see.loop import BASELINE_POLICY, DreamRSI, LoopConfig
from see.objective import attainment, beta_sweep, eq1_value, run_episode
from see.policies.parallel_refine import ParallelRefine
from see.policies.portfolio import OptimalPolicy
from see.policy.api import GridPlan, GridPlanningContext
from see.synthetic import synthetic_trace
from see.toy import PORTFOLIO, ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task
from see.world import Cell, Trace


def test_the_default_fallback_grid_is_the_papers_110_call_round(tmp_path, stub_prompts):
    """Sec. 4's budget, 10 workers x 11 attempts = 110 calls per round, is the fallback grid's
    size under LoopConfig's defaults; nothing else enforces a per-round budget (GAPS §3)."""
    cfg = LoopConfig(workdir="unused")
    branches, refines = cfg.fallback_grid
    assert (branches, refines, cfg.max_parallelism) == (10, 10, 10)
    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        ScriptedDiscoveryAgent(seed=0),
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        cfg.max_parallelism,
        branches,
        refines,
    )
    ParallelRefine(None).solve(q, budget=None)
    assert (q.budget_spent, q.decision_rounds) == (110, 11)


def test_eq1_scores_the_root_when_nothing_revealed_beats_it():
    """Eq. (1)'s max runs over the revealed subtree including the root, so a policy that
    reveals only worse cells scores the root, not its best worse cell."""
    assert eq1_value(0.5, 1.0, probes=3, rounds=1, beta1=0.0, beta2=0.0) == 1.0
    assert eq1_value(1.5, 1.0, probes=3, rounds=1, beta1=0.0, beta2=0.0) == 1.5


class _DefaultBeta045(OptimalPolicy):
    """The portfolio policy with a baked-in default beta that no grid beta equals."""

    def __init__(self, config=None):
        super().__init__({"beta": 0.45} if config is None else config)


class _FailsAtItsDefault(_DefaultBeta045):
    def solve(self, question, budget=None):
        if self.beta == 0.45:
            raise RuntimeError("only the default-beta episode fails")
        return super().solve(question, budget)


def test_pareto_comes_from_the_beta_grid_and_eq1_from_the_default_beta_episode():
    """Listing 2's reward is the sweep over the beta grid; Sec. 3's Eq. (1) is scored on the
    extra episode at the policy's own default beta, the one a live rollout uses (GAPS §3)."""
    traces = [synthetic_trace(s) for s in range(2)]
    report, execs = beta_sweep(_DefaultBeta045, traces, betas=(0.0, 1.0), beta1=0.01, beta2=0.05)
    assert report["valid"] and report["default_beta"] == 0.45
    assert [e["beta"] for e in execs] == [0.0, 0.0, 1.0, 1.0, None, None]
    assert [p["beta"] for p in report["per_beta"]] == [0.0, 1.0]
    default = execs[-2:]
    assert report["eq1"]["V"] == pytest.approx(sum(e["eq1"] for e in default) / 2)
    assert report["eq1"]["per_trace"] == {e["trace_id"]: e["eq1"] for e in default}
    grid = execs[:-2]
    for p in report["per_beta"]:  # the frontier is built from the grid, never the default
        mine = [e for e in grid if e["beta"] == p["beta"]]
        assert p["work"] == pytest.approx(sum(e["work"] for e in mine) / 2)
        assert p["attainment"] == pytest.approx(sum(e["attainment"] for e in mine) / 2)


def test_an_error_in_either_episode_set_invalidates_both_objectives():
    report, execs = beta_sweep(_FailsAtItsDefault, [synthetic_trace(0)], betas=(0.0, 1.0))
    assert [bool(e["error"]) for e in execs] == [False, False, True]
    assert not report["valid"]
    assert report["pareto"]["reward"] == report["eq1"]["V"] == float("-inf")


def test_an_out_of_support_plan_is_flagged_and_scored_as_its_clipped_grid():
    """Replay clips a plan wider than the recorded tree to the tree and scores it like the
    clipped plan; the flag is informational (zero-versus-clip is a D2 decision, GAPS §3)."""
    trace = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)
    context = GridPlanningContext(
        history=(),
        fallback_branch_count=5,
        fallback_refine_count=6,
        hard_max_branch_count=8,
        hard_max_refine_count=8,
        max_parallelism=5,
        trace_branch_count=5,
        trace_refine_count=6,
    )

    class Wide(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(8, 8, reason="wider and deeper than the tree")

    class Clipped(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(5, 6, reason="the tree's own grid")

    wide = run_episode(Wide(None), trace, context, record=True)
    clipped = run_episode(Clipped(None), trace, context, record=True)
    assert wide.out_of_support and not clipped.out_of_support
    assert (wide.probes, wide.best, wide.attainment, wide.penalty, wide.log) == (
        clipped.probes,
        clipped.best,
        clipped.attainment,
        clipped.penalty,
        clipped.log,
    )
    assert wide.plan == {"branch_count": 8, "refine_count": 8, "reason": "wider and deeper than the tree"}
    report, _ = beta_sweep(Wide, [trace], context_for=lambda t: context, betas=(0.5,))
    assert report["out_of_support"] is True and report["valid"]


def test_the_floor_is_reswept_for_reference_and_never_deployed(tmp_path, stub_prompts):
    """policy_dev/history/baseline/ is scored on the current pool every offline phase, but only
    the deployed policy and the agent's revisions are candidates (GAPS §3)."""
    work = str(tmp_path)
    cfg = LoopConfig(
        workdir=work, max_parallelism=2, fallback_grid=(2, 1), hard_max_grid=(2, 1), versions=2
    )
    loop = DreamRSI(cfg, make_task(work), ScriptedDiscoveryAgent(seed=7), ScriptedPolicyAgent())
    loop.online(1)
    loop.offline(1)
    floor = tmp_path / "policy_dev" / "history" / "baseline" / "proposal_results"
    assert (floor / "beta_sweep.json").exists()
    rounds = [c["round"] for c in loop.state["log"][-1]["offline"]]
    assert rounds == ["r0001_t01_m0", "r0002_t01_m1"]
    assert loop.state["deployed_round"] in rounds


def test_the_initial_policy_seeds_only_a_fresh_workdir(tmp_path):
    """LoopConfig.initial_policy is pi_1: it is copied to deployed/iter0001.py when a workdir is
    created and never again, so a resumed run keeps the policy it started with."""
    work = str(tmp_path)
    deployed = os.path.join(work, "deployed", "iter0001.py")
    with open(PORTFOLIO, "rb") as f:
        portfolio = f.read()
    first = DreamRSI(
        LoopConfig(workdir=work, initial_policy=PORTFOLIO),
        make_task(work),
        ScriptedDiscoveryAgent(),
        ScriptedPolicyAgent(),
    )
    with open(deployed, "rb") as f:
        assert f.read() == portfolio
    again = DreamRSI(
        LoopConfig(workdir=work, initial_policy=BASELINE_POLICY),
        make_task(work),
        ScriptedDiscoveryAgent(),
        ScriptedPolicyAgent(),
    )
    with open(deployed, "rb") as f:
        assert f.read() == portfolio  # an existing workdir keeps its policy
    assert first.state["deployed"] == again.state["deployed"] == deployed


def test_the_trace_ceiling_counts_only_successful_cells():
    """A timed-out or failed attempt keeps its score in the trace but never raises the ceiling
    attainment is measured against (GAPS §3)."""
    t = Trace(
        [
            Cell(0, 0, 0, 1.5),
            Cell(0, 1, 1, 9.0, fail_class="timeout", error="agent timed out after 60s"),
            Cell(1, 0, 2, 2.0, evaluated=False),
        ],
        baseline_score=1.0,
        max_parallelism=2,
    )
    assert t.ceiling() == 1.5
    assert attainment(9.0, t) == 1.0  # clipped: nothing recorded beats 1.5
```

In `reconstruction/tests/test_objective.py`, extend `test_attainment_normalises_between_root_and_ceiling` so its last two lines read:

```python
    flat = Trace([Cell(0, 0, 0, 0.5)], baseline_score=1.0, max_parallelism=1)
    # a flat trace: nothing recorded beats the root, so every episode attains all there is
    assert attainment(None, flat) == 1.0
    assert attainment(0.1, flat) == 1.0 and attainment(5.0, flat) == 1.0
```

- [ ] **Step 2: Run the new module and the extended test**

Run: `python -m pytest -q tests/test_ledger.py tests/test_objective.py::test_attainment_normalises_between_root_and_ceiling`
Expected: `9 passed`. These are pins, not fixes: they pass on the current tree by design. If any fails, stop and report it: it means the code and the ledger already disagree, and the controller rules on which is right before this task continues.

- [ ] **Step 3: Lint and run the suite**

Run: `ruff check . && ruff format --check . && pyright`
Expected: clean
Run: `python -m pytest -q`
Expected: `96 passed`

- [ ] **Step 4: Commit**

```bash
git add tests/test_ledger.py tests/test_objective.py
git commit -m "Pin the ledger's scoring conventions with hand-computed cases"
```

---

### Task 6: Replay equivalence asserts what its comment claims

**Files:**
- Modify: `reconstruction/tests/test_replay_equivalence.py:79-83` (after `trace = pool[0][0]`) and `:84-93` (after `episode = run_episode(...)`)

**Interfaces:**
- Consumes: `Trace.grid == (branch_count, refine_count)` and `len(trace)` (recorded cells); `manifest["planned_grid"]` (`dataclasses.asdict(plan)` or `None`) and `manifest["effective_grid"]`; `Episode.plan` (the same `asdict` or `None`).
- Produces: nothing new; the module comment's claim becomes an assertion.

- [ ] **Step 1: Add the assertions**

After the line `trace = pool[0][0]` add:

```python
            branches, refines = trace.grid
            assert (branches, refines) == (
                manifest["effective_grid"]["branch_count"],
                manifest["effective_grid"]["refine_count"],
            ), where
            partial = len(trace) < branches * (refines + 1)
            assert partial == (name == "portfolio" and seed in (0, 1)), (
                f"{where}: {len(trace)} cells recorded of a {branches} x {refines + 1} grid"
            )
```

After the `episode = run_episode(...)` call (before `replay = _transcript(episode.log)`) add:

```python
            assert episode.plan == manifest["planned_grid"], where
```

- [ ] **Step 2: Run the test**

Run: `python -m pytest -q tests/test_replay_equivalence.py`
Expected: `1 passed`. If the `partial` assertion fails for some seed, do not change the seeds or the expected set: report which (seed, policy) pairs are partial; the controller rules whether the comment or the expectation is corrected.

- [ ] **Step 3: Run the suite and commit**

Run: `python -m pytest -q`
Expected: `96 passed`

```bash
git add tests/test_replay_equivalence.py
git commit -m "Assert the partial-tree and planned-grid claims in the replay test"
```

---

### Task 7: GAPS §3 gains its "How to change" column, five rows and the §1 split

**Files:**
- Modify: `reconstruction/GAPS.md` §1 (lines 15-16), §3 (lines 48-75: the header, every row, the rows named below)

**Interfaces:**
- Consumes: the rulings and the fixes of Tasks 1-6 (test names are cited in the new rows).
- Produces: the table Task 8's README and CLAUDE.md wording refers to.

Use the Read tool for `GAPS.md` (it is tables) and the Edit tool per row; do not regenerate the file. Each existing §3 row is one line; append the new cell before the row's final `|`. Verify each claim against the code while editing: the reviewer will.

- [ ] **Step 1: Split the two §1 rows**

Replace the row beginning `| Tree structure, legal set A(T) = {r} ∪ leaves, batch size ≤ W |` with:

```
| Tree structure, batch size ≤ W | Sec. 3 | `see/world.py` |
```

Replace the row beginning `| Replay reveal rule, termination (empty batch, K2, tree exhausted) |` with:

```
| Replay reveal rule, termination on K2 or an exhausted tree | Sec. 3 | `see/world.py` |
```

- [ ] **Step 2: Widen the §3 header**

Replace the two header lines of the §3 table:

```
| Item | What the paper gives | Chosen here | How to change |
|---|---|---|---|
```

- [ ] **Step 3: Append the "How to change" cell to every existing row**

For each row, insert ` <cell> |` before the row's final `|`. The cells, by the row's first column:

- `M (versions per offline phase), K1, K2` → `` `LoopConfig.versions` (`run_dream_rsi.py --versions`, `see demo --versions`); `LoopConfig.max_live_rounds` (no flag); `LoopConfig.max_replay_rounds` (`see sweep --max-rounds`) ``
- `β1, β2 in Eq. (1)` → `` `LoopConfig.beta1`/`beta2` (no runner flag); `see sweep --beta1/--beta2` ``
- `Which objective selects π_{t+1}` → `` `LoopConfig.objective`, one of `see.objective.OBJECTIVES`; `run_dream_rsi.py --objective` ``
- `"Per-trace attainment"` → `` fixed in `see/objective.py` (`attainment`) ``
- `` `pareto.auc` `` → `` fixed in `see/objective.py` (`pareto_auc`) ``
- `Beta grid, λ` → `` `LoopConfig.betas`/`lam` (no runner flag); `see sweep --betas/--lam` ``
- `Failure taxonomy` → `` fixed in `see/policy/observation_signal.py` (frozen surface) ``
- `Helper semantics (`branch_promising`, …)` → `` fixed in `see/policy/observation_signal.py` (frozen surface) ``
- `` `GridPlanningContext` field names `` → `` fixed in `see/policy/api.py` (frozen surface) ``
- `` `live_cycle_manifest.json` schema, `_current` `` → `` fixed in `see/loop.py` (`_manifest`, `online`) ``
- `Workspace snapshot` → `` fixed in `see/live.py` (`_resume_from`) ``
- `History given to the discovery agent` → `` fixed in `see/loop.py` (`online`) ``
- `Direction provider / `$direction_guidance`` → `` `DreamRSI(directions=...)`; no flag ``
- `Concurrent evaluation` → `` `LoopConfig.serialize_eval` (no flag) ``
- `Policy-development agent` → `` `DreamRSI(policy_agent=...)`; `run_dream_rsi.py --policy-agent` (a preset name or a JSON argv) ``
- `Sandboxing LLM-written policy code` → `` `LoopConfig.sweep_timeout`/`kill_grace` (no flag); the mechanism is fixed in `see/loop.py` (`_sweep`) ``
- `Interrupted iteration handling` → `` fixed in `see/loop.py` (`online`) ``
- `Rejected/absent grid plan in replay` → `` the rule is fixed in `see/objective.py` (`run_episode`); the grid is `LoopConfig.fallback_grid` (`run_dream_rsi.py --grid`, `see sweep --fallback`) ``
- `Malformed (non-raising) evaluator result` → `` fixed in `see/live.py` (`_evaluate`) ``
- `Deploy-time code integrity` → `` fixed in `see/loop.py` (`_archive`, `_deploy`, `online`) ``
- `` `hard_max_grid` default `` → `` `LoopConfig.hard_max_grid`; `run_dream_rsi.py --hard-max`; `see sweep --hard-max` ``
- `Interrupt semantics (SIGINT, SIGTERM, SIGHUP mid-iteration)` → `` fixed in `see/loop.py` (`install_signal_handlers`, `online`) ``
- `Agent and sweep child-process lifetime` → `` `CommandAgent(timeout=, kill_grace=)`; `run_dream_rsi.py --agent-timeout`; the mechanism is fixed in `see/live.py` ``
- `Archive-name guard on restart` → `` fixed in `see/loop.py` (`archive_name`, `online`) ``

- [ ] **Step 4: Amend four existing rows' "Chosen here" cells**

- `β1, β2 in Eq. (1)`: append to the cell: `; Eq. (1)'s max runs over the revealed subtree including the root, so a policy that reveals only worse cells scores the root (pinned by ` `` `test_ledger.py::test_eq1_scores_the_root_when_nothing_revealed_beats_it` `` `)`
- `Which objective selects π_{t+1}`: replace `Listing 2 by default, `objective="eq1"` switch` with `Listing 2 by default; `objective="eq1"` selects Eq. (1); any other name is refused when `LoopConfig` is built and by `score_of`, so a typo cannot silently select Eq. (1)`
- `"Per-trace attainment"`: append `; on a trace whose ceiling does not beat the root, 1.0 for every episode, best or none, because nothing recorded beats the root (pinned in `test_objective.py::test_attainment_normalises_between_root_and_ceiling`)`
- `Deploy-time code integrity`: replace `on success `state.json` records the deployed digest beside the score, absent until the first `offline()` completes;` with `on success `state.json` records the deployed digest beside the score in the single write `offline()` makes at its end, together with the offline log and the iteration counter (`online()` writes nothing, so a crash anywhere in `offline()` leaves the file as the previous iteration left it; pinned in `test_loop.py::test_state_is_written_once_per_iteration_after_the_deploy`), absent until the first `offline()` completes;`

- [ ] **Step 5: Amend the lifetime row for Tasks 1 and 4**

In the row `Agent and sweep child-process lifetime`:

- replace `a timed-out agent's attempt is scored as whatever program it left (its parent's copy if it wrote nothing) — whether that should count as `no_program` is open (track D: a fidelity question, not a lifetime one);` with `a timed-out agent's attempt is scored as whatever program it left but recorded as a `timeout` failure with the agent's own message as its error (the evaluator's error wins when there is one), so it is never a success and never raises the trace ceiling; its parent's copy is what gets scored if it wrote nothing, and whether an untouched program should count as `no_program` is D2 (pinned in `test_command_agent.py::test_agent_timeout_kills_the_whole_process_group`);`
- replace `and if its `solve` then returns normally the manifest records no error for the truncated tree;` with `and if its `solve` then returns normally the manifest still records `batch abandoned: <fault>` and the cycle is frozen with that error (pinned in `test_loop.py::test_a_fault_the_policy_swallows_still_reaches_the_manifest`);`

- [ ] **Step 6: Add the five new rows at the end of the §3 table**

```
| Per-round call budget | Sec. 4: 10 workers × 11 attempts = 110 calls (Pro), 32 × 20 = 640 (Flash) | the fallback grid's size and the hard caps' product; no product cap and no budget is enforced, so a plan may be (32, 0) or (1, 19); the demo hardcodes (4, 5) and (8, 8) (pinned in `test_ledger.py::test_the_default_fallback_grid_is_the_papers_110_call_round` and `test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper`) | `LoopConfig.fallback_grid`/`hard_max_grid`; `run_dream_rsi.py --grid/--hard-max`; `see sweep --fallback/--hard-max`; a budget knob is D2 |
| Legal set A(T) | Sec. 3: {r} ∪ leaves, one branch opened per round | Listing 2's: one root cell per unopened branch, so several branches may open in one batch (§4 item 1; pinned in `test_world.py::test_initial_legality_is_roots_only_in_creation_order`) | fixed in `see/world.py` (`legal_roots`) |
| `probe_batch([])` | Sec. 3 lists the empty batch among the terminations | illegal (`IllegalBatch`: "stop by not probing"); in replay it scores the version −∞ like any other exception, live it propagates to the policy (pinned in `test_world.py::test_illegal_batches_are_refused`, `test_objective.py::test_crashing_or_illegal_policy_is_scored_minus_infinity`) | fixed in `see/world.py` (`probe_batch`); ending the episode instead is D2 |
| Two objectives, two episode sets | Eq. (1) per episode; Listing 2 sweeps beta | `pareto.reward` is computed from the beta-grid episodes (one per trace per beta) and `eq1.V` from one extra episode per trace at the policy's baked-in default beta, the one a live rollout uses; an error in either set invalidates both (pinned in `test_ledger.py::test_pareto_comes_from_the_beta_grid_and_eq1_from_the_default_beta_episode` and `::test_an_error_in_either_episode_set_invalidates_both_objectives`) | betas and λ as above; the split is fixed in `see/objective.py` (`beta_sweep`) |
| Trace ceiling | not named | the best successful cell's score, else the baseline; failed and timed-out cells keep their scores in the trace but never raise it; an out-of-support plan is clipped to the trace and scored as the clipped plan, with `out_of_support` reported for information (pinned in `test_ledger.py::test_the_trace_ceiling_counts_only_successful_cells` and `::test_an_out_of_support_plan_is_flagged_and_scored_as_its_clipped_grid`) | fixed in `see/world.py` (`Trace.ceiling`, `ReplayQuestion`); zero-versus-clip is D2 |
```

- [ ] **Step 7: Check and commit**

Run the cell count over §3 (the header row counts as one row):

```bash
awk '/^## 3/{s=1} s&&/^## 4/{exit} s&&/^\| /{n=split($0,a,"|"); c[n-2]++} END{for(k in c) print "cells=" k " rows=" c[k]}' GAPS.md
```

Expected: exactly one line, `cells=4 rows=30` (the header plus 29 data rows; today it prints `cells=3 rows=25`).
Run: `python -m pytest -q`
Expected: `96 passed`

```bash
git add GAPS.md
git commit -m "Give every GAPS ledger row a how-to-change cell and add the D1 rulings"
```

---

### Task 8: README, CLAUDE.md and the CI pin

**Files:**
- Modify: `reconstruction/README.md:14,16-17` (status rows)
- Modify: `.claude/CLAUDE.md` (the "Two objectives that disagree" paragraph, the conventions bullet "Every choice the paper does not specify…", the suite time on the `python -m pytest -q` line)
- Modify: `.github/workflows/ci.yml` (`--expect 82` → the measured count)

**Interfaces:**
- Consumes: the measured test count and suite time.
- Produces: docs in step with the code; the CI pin.

- [ ] **Step 1: README status rows**

Row 14, replace `implemented; AUC/attainment details inferred` with `implemented; AUC/attainment details inferred and pinned (`tests/test_ledger.py`); `pareto` scores the beta grid, `eq1` the default-beta episode`.

Row 16, append before the final `|`: `; a timed-out attempt is scored as the program it left but recorded as a `timeout` failure`.

Row 17, replace `deploy integrity verified by digest when a version is scored, when it is deployed and again when it is loaded;` with `deploy integrity verified by digest when a version is scored, when it is deployed and again when it is loaded; `state.json` is written once per iteration, after the deploy;`.

- [ ] **Step 2: CLAUDE.md**

In the "Two objectives that disagree" paragraph, replace `` `LoopConfig.objective` selects one; default `"pareto"`. `` with `` `LoopConfig.objective` selects one, `"pareto"` by default, and any other name is refused when the config is built. ``

In the Conventions list, replace the bullet

```
- Every choice the paper does not specify gets a row in `GAPS.md` §3; every contradiction found
  in the paper goes in §4. Keep the README status table in step with what is actually implemented.
```

with

```
- Every choice the paper does not specify gets a row in `GAPS.md` §3 with its "How to change" cell
  and a hand-computed pin (`tests/test_ledger.py` or the module that owns the path); every
  contradiction found in the paper goes in §4. Keep the README status table in step with what is
  actually implemented.
```

Measure the suite: `python -m pytest -q --junitxml=report.xml` and read the time; update the `~12s` on the `python -m pytest -q` line if it changed by more than a second.

- [ ] **Step 3: CI pin**

Run: `python tools/check_junit.py report.xml --expect 96`
Expected: `junit: 96 tests, no skips`. Set `--expect 96` in `.github/workflows/ci.yml`. If the count differs, use the measured count and say so in the commit body.

- [ ] **Step 4: Demo and commit**

Run: `python -m see demo --workdir /tmp/drsi-d1 && rm -rf /tmp/drsi-d1`
Expected: three `iter N:` lines and exit 0.

```bash
git add README.md ../.claude/CLAUDE.md ../.github/workflows/ci.yml
git commit -m "Bring the docs and the CI count up to the D1 ledger"
```

---

## Self-review

- Spec coverage: §3.1-3.4 → Tasks 1-4; §4 rulings 1, 4, 5, 6, 7, 8 → Task 5; ruling 3 → Task 5 (the extended test); ruling 2 and 9 → Task 7 rows (already pinned); ruling 10 → Task 6; §5 → Task 7; §6-7 → every task's gate steps and Task 8; §9 stays in the spec.
- Placeholders: none; every code step carries its code, every doc step its text.
- Type consistency: `_cancel(pool, close, fault)` in Task 4 matches both call sites; `OBJECTIVES` is defined in Task 2 before Task 7 cites it; test names cited in Task 7 match Tasks 1, 3, 4, 5 exactly.
- Spiked on the current tree before this plan was committed (then reverted): the eight `test_ledger.py` pins and the Task 6 assertions pass unchanged; the Task 1 fix turns its test red then green with ruff clean; every string Tasks 7 and 8 replace was grepped verbatim.
