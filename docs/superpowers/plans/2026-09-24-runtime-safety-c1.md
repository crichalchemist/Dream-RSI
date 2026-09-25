# Runtime Safety C1 (Interruption and Child-Process Lifetime) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an interrupted or timed-out run of `reconstruction/`'s outer loop stop promptly, kill everything it forked, preserve what it collected as a readable trace under `runs/` (never in the pool), and refuse a restart that would overwrite the interrupted iteration's archives.

**Architecture:** Five responsibilities, each in the file where the process already lives, no new module and no constructor-signature change. `CommandAgent` runs each agent CLI in its own session, kills the group on timeout, and gains a duck-typed `terminate()`; `LiveQuestion._execute` cancels queued attempts and calls `terminate()` when an interrupt reaches the main thread; `DreamRSI.online` freezes the partial tree under `runs/iterNNNN/partial/` on any interrupt; `DreamRSI._sweep` runs the replay subprocess in its own session and kills it on timeout or interrupt; a shared `archive_name` helper lets `online()`'s guard refuse when the first archive directory exists, and `install_signal_handlers()` gives SIGTERM the Ctrl-C path in the two CLI entry points.

**Tech Stack:** Python ≥ 3.10 standard library (`subprocess.Popen(start_new_session=True)`, `os.killpg`, `signal`, `concurrent.futures.as_completed`), pytest, ruff 0.16.8, pyright 1.1.414, pre-commit; `see/` stays standard-library only; POSIX only.

**Spec:** `docs/superpowers/specs/2026-09-24-runtime-safety-c1-design.md` (binding authority; commit 27c857c).

## Global Constraints

- Branch `spec/runtime-safety-c1`, forked from `main` at e56fb5a; the spec is commit 27c857c. Never push without the owner's explicit go-ahead.
- All commands run from `reconstruction/` inside its venv (`export PATH="$PWD/.venv/bin:$PATH"` first; never the system interpreter). The package is installed editable.
- The gate is strict and stays strict: `ruff format --check .`, `ruff check .` (line-length 100, rules `E,F,I,UP,B,RUF`), `pyright` (basic, 0 errors), `python tools/extract_listings.py --check`, `python -m pytest -q` with ZERO skips, `python tools/check_junit.py report.xml --expect N`. No `xfail`, no `skip`, no `# noqa`, no `# type: ignore`. Run `ruff format .` once before the gate if `--check` complains; the resulting diff must touch only the task's files.
- Test count: 55 today → 62 after Task 7 (Task 1 +1, Task 2 +1, Task 3 +1, Task 4 +1, Task 5 +1, Task 6 +1, Task 7 +1). The count is pinned ONLY in `.github/workflows/ci.yml` (Task 8); `.claude/CLAUDE.md` and `reconstruction/README.md` never quote it.
- Suite runtime target: about eight seconds (owner decision; track B's was five). Every child-process test uses a shell stand-in (`sh -c` forking a `sleep`) with sub-second timeouts — never a real agent CLI or an API.
- Frozen surfaces: nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes; no new `fail_class` string; the agent callable's signature `agent(prompt, *, cwd, target) -> dict` is unchanged (toy agents in `see/toy.py` are untouched); no workdir directory (`runs/`, `trace_pool/`, `policy_dev/history/`, `deployed/`, `state.json`) is renamed; `generated/` is never committed.
- Owner decisions binding every task: an interrupt freezes what was collected under `runs/iterNNNN/partial/` and never into `trace_pool/`; cells whose agent was killed are discarded, not recorded; resumability is not planned (refuse loudly on restart); POSIX only (`os.killpg`, `start_new_session`; the CI matrix is ubuntu 3.10/3.13 and macos 3.13).
- Test conventions: names state the claim they protect; worlds are built from `see.toy` and `tmp_path`, never from files in the repository; `pytest.raises(match=r"...")` uses raw strings; the process helpers are `process_gone` (a fixture in `tests/conftest.py`, Task 1) and `STAND_IN`/`stand_in`/`grandchild_pids` (module-level in `tests/test_command_agent.py`, Task 1); tests that need `see.prompts` (anything that runs `LiveQuestion._run_attempt` or `DreamRSI.online`) request the module-scoped `stub_prompts` fixture.
- Every paper-silent choice gets a `reconstruction/GAPS.md` §3 row (written in Task 8).
- Commit messages: plain imperative sentences like the existing history (no `feat:` prefixes), **no `Co-Authored-By` or other agent-attribution trailers** (owner's rule). One commit per task; the pre-commit hooks run on every commit (ruff-check, ruff-format, pyright on `reconstruction/` changes) and may print each hook twice (a local hook-template quirk).
- Line numbers cited in tasks are as of e56fb5a; locate every edit by the quoted text if they have moved.
- Deviations from the spec, decided while writing this plan: (a) spec §8 places test 4 in `tests/test_live_interrupt.py`; it lives in `tests/test_command_agent.py` so it can share the stand-in helpers without importing another test module (`tests/` is not a package). (b) spec §4 says `_execute` cancels on "any `BaseException`"; the plan cancels only when the exception is not an `Exception` (an interrupt): an ordinary worker exception keeps today's path, so an LLM policy that catches it is not left holding a terminated agent — C2 contains those. (c) test 4 raises the interrupt from a SIGALRM handler in the main thread (the shape a real SIGINT has) rather than from a worker, which makes "queued attempts never start" deterministic: both workers are blocked in `communicate()` when the signal lands. (d) spec §5 assumes a timed-out agent yields a `no_program` cell; `_run_attempt` copies the parent's program into the attempt before calling the agent, so an agent that wrote nothing leaves that copy and the cell is scored as it (an evaluated cell equal to its parent). C1 keeps that behaviour, records `agent_timed_out`, and Task 8's GAPS row says so; whether such a cell should count as `no_program` is C2's. (e) test 6 seeds the archive directory by hand and asserts the shared helper's name instead of running a whole toy `offline()` (saves ~1.5 s of an 8 s budget). (f) the spec §10.3 manual check uses a small wrapper script with a slowed toy agent so SIGTERM lands during `online()` deterministically.

---

### Task 1: `CommandAgent` — own session, process-group kill on timeout, `timed_out`

**Files:**
- Modify: `reconstruction/see/live.py` — the import block (`:12-24`: add `import signal`), a new module-level `kill_process_group` placed between the `TaskSpec` dataclass (`:39-49`) and `class CommandAgent`, the whole `CommandAgent` class (`:52-81`), and the `eval/score.json` dict in `LiveQuestion._run_attempt` (`:213-218`, the line `"agent_returncode": run.get("returncode") if run else None,`).
- Modify: `reconstruction/tests/conftest.py` — add `import os`, `import time`, and the `process_gone` fixture after `stub_prompts`.
- Create: `reconstruction/tests/test_command_agent.py`.

**Interfaces:**
- Consumes: `see.live.LiveQuestion(task, agent, tree_dir, history_dir, baseline_score, max_parallelism, branch_count, refine_count, ...)`, `Question.probe_batch(cells) -> list[Observation]`, `Question.legal_roots() -> list[str]`, `see.toy.make_task(workdir) -> TaskSpec` (a toy program is `{"x": <float>}` and the baseline scores 1.0); the module-scoped `stub_prompts` fixture in `tests/conftest.py`.
- Produces (Tasks 2, 3 and 5 rely on these exact names):
  - `see.live.kill_process_group(p: subprocess.Popen, grace: float) -> None`: SIGTERM the group, wait up to `grace` seconds for `p`, SIGKILL the group (whatever ignored SIGTERM, including members that outlived `p`), reap `p`; an already-dead group is not an error.
  - `see.live.CommandAgent(argv, timeout: float = 3600.0, env: dict | None = None, kill_grace: float = 5.0)` with attributes `argv`, `timeout`, `env`, `kill_grace`, and private `_live: set[subprocess.Popen]`, `_lock: threading.Lock`.
  - A call that times out returns exactly `{"returncode": None, "stdout": "", "stderr": f"agent timed out after {timeout}s", "timed_out": True}`; a call that finishes returns `{"returncode", "stdout", "stderr"}` as today.
  - `eval/score.json` gains `"agent_timed_out": <bool>` beside `"agent_returncode"`.
  - Fixture `process_gone(pid: int, seconds: float = 2.0) -> bool` in `tests/conftest.py`.
  - Module-level helpers in `tests/test_command_agent.py`: `STAND_IN` (argv template), `stand_in(pid_dir) -> list` and `grandchild_pids(pid_dir) -> list` (pids of every `sleep` the stand-in forked, one per call, read from `child-<shell pid>.pid` files).
- Behaviour kept on purpose (plan deviation (d)): a timed-out agent that wrote nothing leaves the parent's program copy in the attempt directory, so the cell is evaluated as that program. Only the bookkeeping (`agent_timed_out`) is new.

- [ ] **Step 1: Write the fixture and the failing test**

Add to `reconstruction/tests/conftest.py`. The import block becomes:

```python
"""Shared test fixtures for reconstruction/tests."""

import os
import time

import pytest

from see import prompts
```

and this fixture goes at the end of the file, after `stub_prompts`:

```python
@pytest.fixture
def process_gone():
    """``process_gone(pid)``: True once no process with that pid exists, polling up to 2 s.

    The shell stand-ins in tests/test_command_agent.py and the hanging sweep in
    tests/test_loop.py fork a ``sleep``; the tests assert that grandchild is dead.
    """

    def gone(pid: int, seconds: float = 2.0) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.02)
        return False

    return gone
```

Create `reconstruction/tests/test_command_agent.py`:

```python
"""Child-process lifetime through CommandAgent and LiveQuestion.

Every test forks a real child with a shell stand-in for an agent CLI (never a real agent or an
API) and asserts that a timeout, terminate() or an interrupt kills the whole process group.
"""

import json
import time

from see.live import CommandAgent, LiveQuestion
from see.toy import make_task

# Stands in for an agent CLI that forks work and waits for it. ``$0`` is the pid-file directory
# (``{dir}`` below); ``{prompt}`` is substituted by CommandAgent and ignored by the script. The
# pid file is renamed into place so a poller never reads it half-written.
STAND_IN = [
    "sh",
    "-c",
    'sleep 30 & echo $! > "$0/tmp-$$" && mv "$0/tmp-$$" "$0/child-$$.pid"; wait',
    "{dir}",
    "{prompt}",
]


def stand_in(pid_dir) -> list:
    return [str(pid_dir) if a == "{dir}" else a for a in STAND_IN]


def grandchild_pids(pid_dir) -> list:
    """The pids of every ``sleep`` the stand-in forked, one per call so far."""
    return [int(p.read_text()) for p in sorted(pid_dir.glob("child-*.pid"))]


def test_agent_timeout_kills_the_whole_process_group(tmp_path, stub_prompts, process_gone):
    """A timed-out agent and everything it forked are dead when the call returns; through
    LiveQuestion the attempt is scored as the program the agent left, with agent_timed_out set."""
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
    # the stand-in never touched the program, so the parent's copy (x=1.0) is what gets scored
    assert (obs.cell_id, obs.score, obs.evaluated, obs.fail_class) == ("b0a0", 1.0, True, "ok")
    with open(tree / "attempt_b000_a000" / "eval" / "score.json") as f:
        score = json.load(f)
    assert score["agent_timed_out"] is True and score["agent_returncode"] is None
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)
```

- [ ] **Step 2: Run it and confirm it fails because the interface does not exist yet**

Run: `python -m pytest -q tests/test_command_agent.py`
Expected: `1 failed` with `TypeError: CommandAgent.__init__() got an unexpected keyword argument 'kill_grace'`.

- [ ] **Step 2b: Observe the orphan the old code leaves (the defect this task fixes)**

Run, from `reconstruction/`:

```bash
python - <<'EOF'
import glob, os, tempfile, time
from see.live import CommandAgent
d = tempfile.mkdtemp()
argv = ["sh", "-c", 'sleep 30 & echo $! > "$0/tmp-$$" && mv "$0/tmp-$$" "$0/child-$$.pid"; wait', d, "{prompt}"]
print(CommandAgent(argv, timeout=0.3)("p", cwd=d, target=d))
pid = int(open(glob.glob(d + "/child-*.pid")[0]).read())
time.sleep(0.5)
os.kill(pid, 0)  # raises ProcessLookupError only if the grandchild is gone
print(f"grandchild {pid} is still alive on the old code")
os.kill(pid, 9)  # clean up
EOF
```

Expected: the timeout dict without `timed_out`, then `grandchild <pid> is still alive on the old code`. Record both lines in the report.

- [ ] **Step 3: Implement the session, the group kill and the bookkeeping**

In `reconstruction/see/live.py`, the import block becomes:

```python
import collections
import concurrent.futures
import dataclasses
import json
import numbers
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
```

Replace `class CommandAgent` (from `class CommandAgent:` through the end of its `__call__`, i.e. the `except subprocess.TimeoutExpired:` return block) with this, keeping the two blank lines before `def oriented_score`:

```python
def kill_process_group(p: subprocess.Popen, grace: float) -> None:
    """End ``p``'s whole process group: SIGTERM, then SIGKILL after ``grace`` seconds.

    ``p`` must have been started with ``start_new_session=True``, so its pid is the group id.
    Members that outlive ``p`` (a CLI that exits and leaves a background process holding its
    pipes) are still in the group, so both signals go to the group whether or not ``p`` is
    still running; a group that is already gone is not an error.
    """
    _signal_group(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    _signal_group(p.pid, signal.SIGKILL)  # whatever ignored SIGTERM, including survivors of p
    p.wait()


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:  # nothing left in the group
        pass


class CommandAgent:
    """Run a coding-agent CLI; ``{prompt}`` in the argv template is replaced.

    Every call runs in its own session, so a timeout kills the CLI together with
    everything it forked, not just the CLI.
    """

    def __init__(
        self, argv, timeout: float = 3600.0, env: dict | None = None, kill_grace: float = 5.0
    ):
        self.argv = list(AGENT_PRESETS.get(argv, argv) if isinstance(argv, str) else argv)
        self.timeout = timeout
        self.env = env
        self.kill_grace = kill_grace  # seconds between SIGTERM and SIGKILL
        self._live: set[subprocess.Popen] = set()
        self._lock = threading.Lock()

    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        argv = [prompt if a == "{prompt}" else a for a in self.argv]
        p = subprocess.Popen(
            argv,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **(self.env or {})},
            start_new_session=True,
        )
        with self._lock:
            self._live.add(p)
        try:
            try:
                out, err = p.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                kill_process_group(p, self.kill_grace)
                p.communicate()  # the group is dead; drain what it left in the pipes
                return {
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"agent timed out after {self.timeout}s",
                    "timed_out": True,
                }
            return {"returncode": p.returncode, "stdout": out[-4000:], "stderr": err[-4000:]}
        finally:
            with self._lock:
                self._live.discard(p)
```

In `LiveQuestion._run_attempt`, the `json.dump` dict written to `eval/score.json` gains one line after `"agent_returncode": run.get("returncode") if run else None,`:

```python
                    "agent_timed_out": bool(run.get("timed_out")) if run else False,
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest -q tests/test_command_agent.py`
Expected: `1 passed` in about one second (two 0.3 s timeouts plus grace).

- [ ] **Step 5: Run the full gate**

Run, from `reconstruction/`:

```bash
ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q
```

Expected: `33 files already formatted` (32 today plus the new test file); `All checks passed!`; `0 errors, 0 warnings, 0 informations`; no drift; `56 passed`, no skips.

- [ ] **Step 6: Commit**

```bash
git add see/live.py tests/conftest.py tests/test_command_agent.py
git commit -m "Kill the agent's whole process group on timeout

CommandAgent starts each agent CLI in its own session and, when the
timeout expires, sends SIGTERM and then SIGKILL to the whole group, so the
processes the CLI forked die with it instead of outliving the run. The
result records timed_out and eval/score.json records agent_timed_out;
the cell is still scored as the program the agent left behind."
```

### Task 2: `CommandAgent.terminate()` and the closed state

**Files:**
- Modify: `reconstruction/see/live.py` — `CommandAgent.__init__` (add `self._closed`), `CommandAgent.__call__` (spawn under the lock, refuse when closed, report `agent terminated` when closed mid-call), new `_terminated` and `terminate` methods.
- Modify: `reconstruction/tests/test_command_agent.py` — add `import threading`; append one test after `test_agent_timeout_kills_the_whole_process_group`.

**Interfaces:**
- Consumes: from Task 1, `kill_process_group(p, grace)`, `CommandAgent(argv, timeout, env, kill_grace)` with `_live`, `_lock`, `kill_grace`; the test helpers `stand_in`, `grandchild_pids` and the `process_gone` fixture.
- Produces (Task 3 calls it through `getattr(agent, "terminate", None)`):
  - `CommandAgent.terminate() -> None`: kills every live call's process group with `kill_process_group(p, self.kill_grace)` and closes the agent. Idempotent.
  - A closed agent never spawns: every later call, and any call that was in flight when `terminate()` ran, returns exactly `{"returncode": None, "stdout": "", "stderr": "agent terminated"}` (no `timed_out` key).
  - Toy agents (`see/toy.py`) have no `terminate`; Task 3 only calls it when present.

- [ ] **Step 1: Write the failing test**

Append to `reconstruction/tests/test_command_agent.py` (and add `import threading` to its import block, between `import json` and `import time`):

```python
def test_terminate_kills_live_agents_and_refuses_new_calls(tmp_path, process_gone):
    """terminate() ends a call in flight and kills what it forked; later calls spawn nothing."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)
    results: list = []
    in_flight = threading.Thread(
        target=lambda: results.append(agent("p", cwd=str(tmp_path), target=str(tmp_path))),
        daemon=True,
    )
    in_flight.start()
    deadline = time.time() + 2.0
    while not grandchild_pids(pid_dir) and time.time() < deadline:
        time.sleep(0.02)  # until the stand-in has forked its child
    [pid] = grandchild_pids(pid_dir)
    started = time.time()
    agent.terminate()
    in_flight.join(timeout=2.0)
    assert not in_flight.is_alive() and time.time() - started < 2.0  # not the 3 s timeout
    assert results == [{"returncode": None, "stdout": "", "stderr": "agent terminated"}]
    assert process_gone(pid)
    assert agent("p", cwd=str(tmp_path), target=str(tmp_path)) == {
        "returncode": None,
        "stdout": "",
        "stderr": "agent terminated",
    }
    assert grandchild_pids(pid_dir) == [pid]  # the refused call forked nothing
    agent.terminate()  # idempotent with nothing live
```

- [ ] **Step 2: Run it and confirm it fails because `terminate` does not exist**

Run: `python -m pytest -q tests/test_command_agent.py::test_terminate_kills_live_agents_and_refuses_new_calls`
Expected: `1 failed` with `AttributeError: 'CommandAgent' object has no attribute 'terminate'` (the in-flight stand-in drains at its 3 s timeout, so the run takes about 3 s).

- [ ] **Step 3: Implement `terminate()` and the closed state**

In `reconstruction/see/live.py`, `CommandAgent.__init__` gains one line after `self._lock = threading.Lock()`:

```python
        self._closed = False  # set by terminate(); a closed agent never spawns again
```

Replace `CommandAgent.__call__` with:

```python
    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        argv = [prompt if a == "{prompt}" else a for a in self.argv]
        with self._lock:  # spawning under the lock closes the race with terminate()
            if self._closed:
                return self._terminated()
            p = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, **(self.env or {})},
                start_new_session=True,
            )
            self._live.add(p)
        try:
            try:
                out, err = p.communicate(timeout=self.timeout)
            except subprocess.TimeoutExpired:
                kill_process_group(p, self.kill_grace)
                p.communicate()  # the group is dead; drain what it left in the pipes
                return {
                    "returncode": None,
                    "stdout": "",
                    "stderr": f"agent timed out after {self.timeout}s",
                    "timed_out": True,
                }
            if self._closed:  # terminate() ended this call
                return self._terminated()
            return {"returncode": p.returncode, "stdout": out[-4000:], "stderr": err[-4000:]}
        finally:
            with self._lock:
                self._live.discard(p)

    @staticmethod
    def _terminated() -> dict:
        return {"returncode": None, "stdout": "", "stderr": "agent terminated"}

    def terminate(self) -> None:
        """Kill every call in flight (whole process groups) and refuse every later call."""
        with self._lock:
            self._closed = True
            live = list(self._live)
        for p in live:
            kill_process_group(p, self.kill_grace)
```

- [ ] **Step 4: Run the file's tests**

Run: `python -m pytest -q tests/test_command_agent.py`
Expected: `2 passed`, the new test in well under two seconds.

- [ ] **Step 5: Run the full gate**

Run: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q`
Expected: formatted, `All checks passed!`, `0 errors, 0 warnings, 0 informations`, no drift, `57 passed`, no skips.

- [ ] **Step 6: Commit**

```bash
git add see/live.py tests/test_command_agent.py
git commit -m "Let a CommandAgent terminate its live calls and refuse new ones

terminate() kills every running agent's process group and closes the
agent, so a call in flight and every later call return an agent
terminated result instead of spawning. Spawning happens under the lock,
which closes the race between a call starting and terminate() running.
This is the hook the interrupt path uses to stop a batch."
```

### Task 3: An interrupt cancels queued attempts and kills running agents

**Files:**
- Modify: `reconstruction/see/live.py` — `LiveQuestion.__init__` (`:114-128`: add `self._interrupted`), `LiveQuestion._execute` (`:154-166`, replaced), a new `LiveQuestion._cancel` method after `_execute`, and the first lines of `LiveQuestion._run_attempt` (`:176-177`).
- Modify: `reconstruction/tests/test_command_agent.py` — add `import os`, `import signal`, `import pytest`; append the stand-in exception class and one test at the end.

**Interfaces:**
- Consumes: from Task 1, `CommandAgent(argv, timeout, kill_grace)` and the helpers `stand_in`, `grandchild_pids`, `process_gone`; from Task 2, `CommandAgent.terminate()`; `see.world.Question.probe_batch`, which calls `self._execute(metas)` after advancing its round counters and records `self.cells` only through `_execute`'s return.
- Produces (Task 4 relies on the first two):
  - `LiveQuestion._interrupted: threading.Event`, set once an interrupt has cancelled a batch.
  - `LiveQuestion._execute` raises the interrupt after cancelling: queued attempts never start, `agent.terminate()` is called when the agent has one, and no cell from the interrupted batch enters `self.cells`. An ordinary `Exception` from a worker propagates as today, without cancelling (plan deviation (b)).
  - `LiveQuestion._cancel(pool: concurrent.futures.ThreadPoolExecutor) -> None`.
  - `_run_attempt` raises `RuntimeError` before creating its attempt directory when `_interrupted` is set.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_command_agent.py`, the import block becomes:

```python
import json
import os
import signal
import threading
import time

import pytest

from see.live import CommandAgent, LiveQuestion
from see.toy import make_task
```

Append at the end of the file:

```python
class _Interrupted(BaseException):
    """Stands in for KeyboardInterrupt, which pytest intercepts itself."""


def test_interrupt_cancels_queued_attempts_and_kills_running_agents(
    tmp_path, stub_prompts, process_gone
):
    """With two workers over four roots, an interrupt in the main thread (as SIGINT arrives)
    kills both running agents' children, and the two queued attempts never start."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)
    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        agent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=4,
        refine_count=0,
    )

    def interrupt(signum, frame):
        raise _Interrupted("SIGALRM while two agents were running")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.5)  # both workers are inside communicate() by then
    started = time.time()
    try:
        with pytest.raises(_Interrupted):
            q.probe_batch(q.legal_roots())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 2.0  # not two rounds of the 3 s timeout
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)  # only the two running calls
    assert q.cells == {}  # nothing from the interrupted batch is kept
    assert sorted(os.listdir(tree)) == ["attempt_b000_a000", "attempt_b001_a000"]
```

- [ ] **Step 2: Run it and confirm it fails for the right reason**

Run: `python -m pytest -q tests/test_command_agent.py::test_interrupt_cancels_queued_attempts_and_kills_running_agents`
Expected: `1 failed`, after roughly six seconds, on `assert time.time() - started < 2.0` (the old `_execute` waits for both 3 s timeouts, then runs the two queued attempts for another 3 s). Record the elapsed time from the assertion message; `grandchild_pids` would show four pid files.

- [ ] **Step 3: Cancel the batch when an interrupt reaches the main thread**

In `reconstruction/see/live.py`, `LiveQuestion.__init__` gains one line after `self._next_seq = 0`:

```python
        self._interrupted = threading.Event()  # set once an interrupt has cancelled a batch
```

Replace `_execute` (from `def _execute(self, metas: list) -> list:` through `return out`) with:

```python
    def _execute(self, metas: list) -> list:
        jobs = []
        for m in metas:
            jobs.append((m, self._next_seq, self._direction(m.branch)))
            self._next_seq += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallelism) as pool:
            futures = [pool.submit(self._run_attempt, *job) for job in jobs]
            try:
                for f in concurrent.futures.as_completed(futures):
                    f.result()  # surfaces a worker's exception as soon as it happens
            except BaseException as e:
                if not isinstance(e, Exception):  # an interrupt, not a worker bug
                    self._cancel(pool)
                raise
            cells = [f.result() for f in futures]
        out = []
        for cell in cells:
            self.cells[cell.id] = cell
            parent = self.cells.get(cell.parent_id) if cell.parent_id else None
            out.append(observation_for(cell, parent, self.baseline_score))
        return out

    def _cancel(self, pool: concurrent.futures.ThreadPoolExecutor) -> None:
        """Stop the batch: queued attempts never start and running agents are killed."""
        self._interrupted.set()
        pool.shutdown(wait=False, cancel_futures=True)
        terminate = getattr(self.agent, "terminate", None)
        if callable(terminate):
            terminate()
```

`_run_attempt` gains a check as its first statement, before `t = self.task`:

```python
        if self._interrupted.is_set():  # the batch was cancelled before this attempt started
            raise RuntimeError(f"attempt b{meta.branch}a{meta.attempt} cancelled by an interrupt")
```

Why this shape: `as_completed` lets the main thread see a worker's exception (or its own signal) without waiting on the first future in order; `shutdown(wait=False, cancel_futures=True)` drops the queued futures; the `with` block's own `shutdown(wait=True)` then completes as soon as the killed children return to their workers. A cancelled future is never `result()`-ed, so the `RuntimeError` a late worker raises is stored and ignored.

- [ ] **Step 4: Run the file's tests**

Run: `python -m pytest -q tests/test_command_agent.py`
Expected: `3 passed`; the new test finishes in about one second.

- [ ] **Step 5: Run the full gate**

Run: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q`
Expected: formatted, `All checks passed!`, `0 errors, 0 warnings, 0 informations`, no drift, `58 passed`, no skips. `tests/test_replay_equivalence.py` and the `finished_loop` tests still pass: the toy agents have no `terminate` and no interrupt occurs.

- [ ] **Step 6: Commit**

```bash
git add see/live.py tests/test_command_agent.py
git commit -m "Cancel the batch and kill running agents when an interrupt arrives

LiveQuestion._execute now waits with as_completed, so an interrupt in the
main thread is acted on at once: queued attempts are cancelled, the agent's
terminate() (when it has one) kills every running process group, and no
cell from the interrupted batch is recorded. An ordinary exception from a
worker keeps today's path."
```

### Task 4: An interrupted iteration is frozen under `runs/`, never into the pool

**Files:**
- Modify: `reconstruction/see/loop.py` — `DreamRSI.online` from `started, error = time.time(), None` (`:164`) through `return manifest` (`:193`); two new methods `_manifest` and `_freeze` placed right after `online`.
- Modify: `reconstruction/tests/test_loop.py` — add `from see.world import Trace` to the import block; append one test at the end of the file (after `test_one_malformed_evaluator_result_fails_one_cell_not_the_batch` and the digest tests).

**Interfaces:**
- Consumes: from Task 3, an interrupt propagates out of `probe_batch` with nothing from its batch in `q.cells`; `LiveQuestion.frozen(trace_id, info) -> Trace`, `LiveQuestion.manifest_stats() -> dict`, `LiveQuestion.episode` (list of recorded rounds); `see.world.Trace.save(path)` / `Trace.load(path)`; track B's helpers in `tests/test_loop.py`: `_Interrupted(BaseException)`, `_RecordingAgent(interrupt_on=2)` (dies on its second call, message `killed while attempt_b000_a001 was running`), `_interruptible_loop(work, agent)` (one worker, grid `(2, 1)`).
- Produces:
  - On any non-`Exception` `BaseException` escaping `policy.solve`, `online()` writes `runs/iterNNNN/partial/trace.json` (trace id `iterNNNN-partial`, info `{"iteration": t, "partial": True}`), `runs/iterNNNN/partial/live_episode.jsonl` and `runs/iterNNNN/partial/live_cycle_manifest.json` (the normal manifest plus `"partial": True`, with `error` = `f"{type(e).__name__}: {e}"`), then re-raises. `trace_pool/` and `state.json` are untouched.
  - The completed path is unchanged in effect: `trace_pool/iterNNNN/` with the same three files and info `{"iteration": t}`; the manifest has no `partial` key.
  - `DreamRSI._manifest(t, policy, plan, grid, q, error, started) -> dict` and `DreamRSI._freeze(q, into, trace_id, info, manifest) -> None` (static).

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_loop.py`, add to the import block (it sorts after `from see.toy import ...`):

```python
from see.world import Trace
```

Append at the end of the file:

```python
def test_interrupt_freezes_the_partial_tree_under_runs_not_the_pool(tmp_path, stub_prompts):
    """What an interrupted iteration collected is a readable trace under runs/, and the pool
    never sees it."""
    loop = _interruptible_loop(str(tmp_path), _RecordingAgent(interrupt_on=2))
    state = (tmp_path / "state.json").read_text()
    with pytest.raises(_Interrupted):
        loop.online(1)
    partial = tmp_path / "runs" / "iter0001" / "partial"
    trace = Trace.load(str(partial / "trace.json"))
    assert trace.trace_id == "iter0001-partial"
    assert trace.info == {"iteration": 1, "partial": True}
    assert sorted(trace.cells) == ["b0a0"]  # only the cell completed before the interrupt
    assert trace.grid == (2, 1)
    with open(partial / "live_cycle_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["error"] == "_Interrupted: killed while attempt_b000_a001 was running"
    assert manifest["partial"] is True
    assert manifest["probes"] == 1
    assert manifest["effective_grid"] == {"branch_count": 2, "refine_count": 1}
    with open(partial / "live_episode.jsonl") as f:
        assert len(f.readlines()) == 1  # the one round that completed
    assert os.listdir(tmp_path / "trace_pool") == []
    assert (tmp_path / "state.json").read_text() == state
```

- [ ] **Step 2: Run it and confirm it fails because nothing is frozen**

Run: `python -m pytest -q tests/test_loop.py::test_interrupt_freezes_the_partial_tree_under_runs_not_the_pool`
Expected: `1 failed` with `FileNotFoundError` on `runs/iter0001/partial/trace.json`.

- [ ] **Step 3: Freeze the partial tree on an interrupt, sharing the manifest builder**

In `reconstruction/see/loop.py`, `DreamRSI.online`: replace everything from `started, error = time.time(), None` to `return manifest` with:

```python
        started, error = time.time(), None
        try:
            policy.solve(q, budget=None)
        except Exception as e:  # keep what was collected
            error = f"{type(e).__name__}: {e}"
        except BaseException as e:  # an interrupt: freeze under runs/, never into the pool
            partial = os.path.join(run_dir, "partial")
            os.makedirs(partial, exist_ok=True)
            error = f"{type(e).__name__}: {e}"
            manifest = self._manifest(t, policy, plan, grid, q, error, started)
            manifest["partial"] = True
            self._freeze(
                q, partial, f"iter{t:04d}-partial", {"iteration": t, "partial": True}, manifest
            )
            raise
        manifest = self._manifest(t, policy, plan, grid, q, error, started)
        os.makedirs(out)
        self._freeze(q, out, f"iter{t:04d}", {"iteration": t}, manifest)
        current = os.path.join(self.pool, "_current")
        if os.path.lexists(current):
            os.remove(current)
        os.symlink(f"iter{t:04d}", current)
        self.state["log"].append({"iteration": t, "live": manifest})
        return manifest

    def _manifest(self, t, policy, plan, grid, q: LiveQuestion, error, started) -> dict:
        return {
            "iteration": t,
            "policy_round": self.state.get("deployed_round", "initial"),
            "beta": getattr(policy, "beta", None),
            "planned_grid": dataclasses.asdict(plan) if plan else None,
            "used_fallback": plan is None,
            "effective_grid": {"branch_count": grid[0], "refine_count": grid[1]},
            **q.manifest_stats(),
            "error": error,
            "started": started,
            "finished": time.time(),
        }

    @staticmethod
    def _freeze(q: LiveQuestion, into: str, trace_id: str, info: dict, manifest: dict) -> None:
        q.frozen(trace_id, info).save(os.path.join(into, "trace.json"))
        with open(os.path.join(into, "live_episode.jsonl"), "w") as f:
            for step in q.episode:
                f.write(json.dumps(step) + "\n")
        with open(os.path.join(into, "live_cycle_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
```

The cells completed before the interrupt form contiguous chains (an attempt runs only after its parent was recorded), so `Trace`'s validator accepts them; with no cells at all it accepts an empty trace.

- [ ] **Step 4: Run the loop tests**

Run: `python -m pytest -q tests/test_loop.py`
Expected: `13 passed` (the ten from track B, the two interrupt tests, and the new one); track B's `test_interrupt_leaves_a_partial_run_that_a_restart_refuses` still lists only `attempt_b000_a000` and `attempt_b000_a001` under `runs/iter0001/tree`, because `partial/` is a sibling of `tree/`.

- [ ] **Step 5: Run the full gate**

Run: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q`
Expected: formatted, `All checks passed!`, `0 errors, 0 warnings, 0 informations`, no drift, `59 passed`, no skips. `tests/test_replay_equivalence.py` still passes: completed manifests and traces are byte-for-byte what they were.

- [ ] **Step 6: Commit**

```bash
git add see/loop.py tests/test_loop.py
git commit -m "Freeze what an interrupted iteration collected under runs/

When an interrupt escapes policy.solve, online() writes the cells,
episode log and manifest collected so far to runs/iterNNNN/partial/,
marks the manifest partial with the interrupt as its error, and re-raises.
trace_pool/ is never touched, so replay scoring never sees a truncated
tree, and the restart guard keeps refusing the iteration."
```

### Task 5: The replay sweep runs in its own session and cannot outlive `_sweep`

**Files:**
- Modify: `reconstruction/see/loop.py` — the `from see.live import ...` line (`:34`), `LoopConfig` (`:44-61`: add `kill_grace` after `sweep_timeout`), and `DreamRSI._sweep` from `report_path = os.path.join(out, "beta_sweep.json")` (`:292`) to the end of the method (`:310`).
- Modify: `reconstruction/tests/test_loop.py` — add `import time` to the import block; append one test at the end of the file.

**Interfaces:**
- Consumes: from Task 1, `see.live.kill_process_group(p, grace)` and the `process_gone` fixture; `see sweep` (`see/__main__.py:cmd_sweep`) hashes the method file and then imports it through `load_policy`, so a `method.py` whose top level hangs makes the subprocess hang before it reads the pool.
- Produces:
  - `LoopConfig.kill_grace: float = 5.0` (seconds between SIGTERM and SIGKILL for the sweep subprocess).
  - `_sweep` starts the subprocess with `start_new_session=True`; on timeout it kills the group and returns the invalid report with `errors == [f"RuntimeError: sweep timed out after {sweep_timeout}s"]`; on any exception escaping `communicate()` (an interrupt) its `finally` kills the group before propagating. The report's shape and the `except Exception` branch are unchanged.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_loop.py`, add `import time` to the import block (after `import shutil` once Task 6 adds it; for now after `import os`). Append at the end of the file:

```python
def test_sweep_timeout_kills_the_subprocess_group_and_scores_invalid(tmp_path, process_gone):
    """A replay subprocess that hangs is killed with everything it forked, and the version
    scores invalid exactly as a crashed one does."""
    work = str(tmp_path)
    cfg = LoopConfig(workdir=work, sweep_timeout=1.0, kill_grace=0.2)
    loop = DreamRSI(cfg, make_task(work), ScriptedDiscoveryAgent(), ScriptedPolicyAgent())
    rdir = tmp_path / "hang"
    rdir.mkdir()
    pid_file = rdir / "child.pid"
    method = rdir / "method.py"
    method.write_text(  # importing this "policy" forks a sleep, records its pid, then hangs
        "import subprocess, time\n"
        f"open({str(pid_file)!r}, 'w').write(str(subprocess.Popen(['sleep', '30']).pid))\n"
        "time.sleep(30)\n"
    )
    started = time.time()
    report = loop._sweep(str(method), str(rdir))
    assert time.time() - started < 4.0
    assert report["valid"] is False
    assert report["errors"] == ["RuntimeError: sweep timed out after 1.0s"]
    assert report["pareto"]["reward"] == float("-inf")
    assert process_gone(int(pid_file.read_text()))
    with open(rdir / "proposal_results" / "beta_sweep.json") as f:
        assert json.load(f)["valid"] is False
```

- [ ] **Step 2: Run it and confirm it fails because the interface does not exist yet**

Run: `python -m pytest -q tests/test_loop.py::test_sweep_timeout_kills_the_subprocess_group_and_scores_invalid`
Expected: `1 failed` with `TypeError: LoopConfig.__init__() got an unexpected keyword argument 'kill_grace'`. (On the old code the call would otherwise block: `subprocess.run`'s timeout kills only the direct child, and the forked `sleep` keeps the stdout pipe open for 30 s.)

- [ ] **Step 3: Run the sweep in its own session and kill it on timeout or interrupt**

In `reconstruction/see/loop.py`, the `see.live` import becomes:

```python
from see.live import LiveQuestion, TaskSpec, kill_process_group, oriented_score
```

`LoopConfig` gains one field after `sweep_timeout: float = 1800.0`:

```python
    kill_grace: float = 5.0  # seconds between SIGTERM and SIGKILL for the sweep subprocess
```

In `_sweep`, replace everything from `report_path = os.path.join(out, "beta_sweep.json")` to the end of the method with:

```python
        report_path = os.path.join(out, "beta_sweep.json")
        p = subprocess.Popen(
            cmd,
            cwd=PKG_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # a timeout or an interrupt kills what the policy code forked
        )
        try:
            try:
                _, err = p.communicate(timeout=self.c.sweep_timeout)
            except subprocess.TimeoutExpired:
                kill_process_group(p, self.c.kill_grace)
                raise RuntimeError(f"sweep timed out after {self.c.sweep_timeout}s") from None
            if p.returncode != 0 or not os.path.exists(report_path):
                raise RuntimeError(err[-2000:] or f"exit {p.returncode}")
            with open(report_path) as f:
                return json.load(f)
        except Exception as e:
            report = {
                "valid": False,
                "errors": [f"{type(e).__name__}: {e}"],
                "pareto": {"reward": float("-inf")},
                "eq1": {"V": float("-inf")},
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=1)
            return report
        finally:
            if p.poll() is None:  # an interrupt escaped communicate(): take the child with us
                kill_process_group(p, self.c.kill_grace)
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest -q tests/test_loop.py::test_sweep_timeout_kills_the_subprocess_group_and_scores_invalid`
Expected: `1 passed` in about 1.5 s.

- [ ] **Step 5: Run the full gate**

Run: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q`
Expected: formatted, `All checks passed!`, `0 errors, 0 warnings, 0 informations`, no drift, `60 passed`, no skips. The `finished_loop` fixture's sweeps behave as before (they finish well within the default timeout).

- [ ] **Step 6: Commit**

```bash
git add see/loop.py tests/test_loop.py
git commit -m "Run the replay sweep in its own session and kill it on timeout

_sweep starts the see sweep subprocess with its own process group and,
on timeout or on an interrupt escaping communicate(), kills the whole
group before returning or propagating. A timed-out version still scores
invalid; what changes is that the LLM-written policy code can no longer
leave processes writing under policy_dev/history/ after the run is gone."
```

### Task 6: A restart refuses when the first archive directory exists

**Files:**
- Modify: `reconstruction/see/loop.py` — a new module-level `archive_name` after `BASELINE_POLICY = ...` (`:41`), the guard at the top of `DreamRSI.online` (`:132-139`, from `run_dir = ...` through the `raise RuntimeError(...)` block), and the `name = f"r{...}"` line in `_archive` (`:239`).
- Modify: `reconstruction/tests/test_loop.py` — add `archive_name` to the `from see.loop import ...` line; append one test at the end of the file.

**Interfaces:**
- Consumes: track B's helpers `_RecordingAgent` and `_interruptible_loop` in `tests/test_loop.py`; `DreamRSI.state["round"]` (persisted only by `_deploy` and at the end of `offline()`), `DreamRSI.dev_history` (`policy_dev/history`).
- Produces:
  - `see.loop.archive_name(round_no: int, t: int, m: int) -> str` returning `f"r{round_no:04d}_t{t:02d}_m{m}"`; `_archive` uses it.
  - `online()` refuses when any of `runs/iterNNNN`, `trace_pool/iterNNNN` or `policy_dev/history/<archive_name(round + 1, t, 0)>` exists, in one `RuntimeError` naming every existing path: `"{path} exists: iteration {t} was interrupted or already ran; delete it, do not merge into it"` for one path, `"{a} and {b} exist: ...; delete them, do not merge into them"` for several. Track B's tests match the singular form.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_loop.py`, the `see.loop` import becomes:

```python
from see.loop import DreamRSI, LoopConfig, archive_name
```

Append at the end of the file:

```python
def test_restart_refuses_when_the_first_archive_dir_exists(tmp_path, stub_prompts):
    """A crash in offline() leaves state["round"] unsaved, so a restart would recreate and
    overwrite r{round+1}_tNN_m0; the guard refuses first, naming every leftover at once."""
    agent = _RecordingAgent()
    loop = _interruptible_loop(str(tmp_path), agent)
    archive = tmp_path / "policy_dev" / "history" / archive_name(1, 1, 0)
    assert archive.name == "r0001_t01_m0"  # the name _archive gives iteration 1's version 0
    archive.mkdir(parents=True)
    (archive / "method.py").write_text("# archived by the crashed run\n")
    state = (tmp_path / "state.json").read_text()
    assert json.loads(state)["round"] == 0
    with pytest.raises(
        RuntimeError, match=r"history/r0001_t01_m0 exists: .*delete it, do not merge into it"
    ):
        loop.online(1)
    assert agent.targets == []
    (tmp_path / "runs" / "iter0001").mkdir()
    (tmp_path / "trace_pool" / "iter0001").mkdir()
    with pytest.raises(
        RuntimeError,
        match=r"runs/iter0001 and .*trace_pool/iter0001 and .*r0001_t01_m0 exist: .*delete them",
    ):
        loop.online(1)
    assert (archive / "method.py").read_text() == "# archived by the crashed run\n"
    assert (tmp_path / "state.json").read_text() == state
```

- [ ] **Step 2: Run it and confirm it fails because the helper does not exist**

Run: `python -m pytest -q tests/test_loop.py::test_restart_refuses_when_the_first_archive_dir_exists`
Expected: collection error `ImportError: cannot import name 'archive_name' from 'see.loop'`. Then, to see the defect itself, temporarily replace `archive_name(1, 1, 0)` in the test with the literal `"r0001_t01_m0"` and drop the import: the first `pytest.raises` fails with `DID NOT RAISE` because the old guard checks only `runs/` and `trace_pool/` and runs the iteration on top of the archive. Restore the test afterwards.

- [ ] **Step 3: Share the archive name and check it in the guard**

In `reconstruction/see/loop.py`, after `BASELINE_POLICY = os.path.join(PKG_ROOT, "see", "policies", "parallel_refine.py")` and before `@dataclasses.dataclass`, add (with two blank lines on each side):

```python
def archive_name(round_no: int, t: int, m: int) -> str:
    """The policy_dev/history/ entry for version m of iteration t; rounds are numbered globally."""
    return f"r{round_no:04d}_t{t:02d}_m{m}"
```

In `_archive`, replace `name = f"r{self.state['round']:04d}_t{t:02d}_m{m}"` with:

```python
        name = archive_name(self.state["round"], t, m)
```

At the top of `online`, replace from `run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")` through the closing `)` of `raise RuntimeError(` with:

```python
        run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")
        out = os.path.join(self.pool, f"iter{t:04d}")
        # runs/ is created first and trace_pool/ last; the first archive is what a restart's
        # offline() would recreate, because the round counter is persisted only on success
        first_archive = os.path.join(self.dev_history, archive_name(self.state["round"] + 1, t, 0))
        existing = [p for p in (run_dir, out, first_archive) if os.path.exists(p)]
        if existing:
            one = len(existing) == 1
            raise RuntimeError(
                f"{' and '.join(existing)} {'exists' if one else 'exist'}: iteration {t} was "
                f"interrupted or already ran; delete {'it' if one else 'them'}, do not merge into "
                f"{'it' if one else 'them'}"
            )
```

- [ ] **Step 4: Run the loop tests**

Run: `python -m pytest -q tests/test_loop.py`
Expected: `15 passed`; track B's two interrupt tests still match their singular-form regexes.

- [ ] **Step 5: Run the full gate**

Run: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q`
Expected: formatted, `All checks passed!`, `0 errors, 0 warnings, 0 informations`, no drift, `61 passed`, no skips.

- [ ] **Step 6: Commit**

```bash
git add see/loop.py tests/test_loop.py
git commit -m "Refuse a restart that would overwrite the interrupted iteration's archives

The round counter is persisted only when offline() succeeds, so after a
crash a restart would recreate policy_dev/history/r{round+1}_tNN_m0 and
overwrite it. online()'s guard now checks that directory as well as
runs/iterNNNN and trace_pool/iterNNNN, and names every one that exists in
a single message. archive_name is the one place the name is built."
```

### Task 7: SIGTERM takes the same path as Ctrl-C in the entry points

**Files:**
- Modify: `reconstruction/see/loop.py` — the import block (`:23-32`: add `import signal`), and two new module-level functions `install_signal_handlers` and `_raise_keyboard_interrupt` placed after `archive_name` (Task 6) and before `@dataclasses.dataclass`.
- Modify: `reconstruction/see/__main__.py` — `cmd_demo` (`:56-58`, the local imports and the first statement).
- Modify: `reconstruction/scripts/run_dream_rsi.py` — the `from see.loop import ...` line (`:17`) and the first statement of `main` (`:26`).
- Modify: `reconstruction/tests/test_loop.py` — add `import signal` and `install_signal_handlers` to the imports; append one test at the end of the file.

**Interfaces:**
- Consumes: nothing from earlier tasks beyond the import block Task 6 left (`from see.loop import DreamRSI, LoopConfig, archive_name`).
- Produces: `see.loop.install_signal_handlers() -> None`, which installs a SIGTERM handler that raises `KeyboardInterrupt(f"signal {signum}")` in the main thread. Called by the two CLI entry points only; never on import, never by tests other than the one below.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_loop.py`, the import block gains `import signal` (between `import shutil`, if present, and `import time`; otherwise between `import os` and `import time`) and the `see.loop` import becomes:

```python
from see.loop import DreamRSI, LoopConfig, archive_name, install_signal_handlers
```

Append at the end of the file:

```python
def test_sigterm_takes_the_same_path_as_ctrl_c():
    """`kill <pid>` raises KeyboardInterrupt in the main thread, so a run freezes its partial
    tree and refuses on restart like Ctrl-C does, instead of exiting at once."""
    previous = signal.getsignal(signal.SIGTERM)
    try:
        install_signal_handlers()
        with pytest.raises(KeyboardInterrupt, match=r"signal 15"):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.5)  # the handler runs at the next bytecode boundary
    finally:
        signal.signal(signal.SIGTERM, previous)
```

- [ ] **Step 2: Run it and confirm it fails because the helper does not exist**

Run: `python -m pytest -q tests/test_loop.py::test_sigterm_takes_the_same_path_as_ctrl_c`
Expected: collection error `ImportError: cannot import name 'install_signal_handlers' from 'see.loop'`. (Without the handler a SIGTERM would kill the pytest process outright, which is the defect.)

- [ ] **Step 3: Add the helper and call it from both entry points**

In `reconstruction/see/loop.py`, the import block becomes:

```python
import dataclasses
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
```

After `archive_name` (Task 6) and before `@dataclasses.dataclass`, add:

```python
def install_signal_handlers() -> None:
    """Make SIGTERM take the same path as Ctrl-C: raise KeyboardInterrupt in the main thread.

    Called by the CLI entry points only; a library must not change signal disposition on import.
    """
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")
```

In `reconstruction/see/__main__.py`, `cmd_demo` begins:

```python
def cmd_demo(a):
    from see.loop import DreamRSI, LoopConfig, install_signal_handlers
    from see.toy import ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
    cfg = LoopConfig(
```

In `reconstruction/scripts/run_dream_rsi.py`, the import becomes `from see.loop import DreamRSI, LoopConfig, install_signal_handlers` and `main` begins:

```python
def main(argv=None):
    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest -q tests/test_loop.py::test_sigterm_takes_the_same_path_as_ctrl_c`
Expected: `1 passed`.

- [ ] **Step 5: Run the full gate and the demo**

Run: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q`
Expected: formatted, `All checks passed!`, `0 errors, 0 warnings, 0 informations`, no drift, `62 passed`, no skips, in about eight seconds (record the time).
Run: `python -m see demo --workdir <scratchpad>/c1-demo` (a fresh directory under the session scratchpad, never `/tmp` or the repository).
Expected: exit 0 with three `iter N:` lines.

- [ ] **Step 6: Commit**

```bash
git add see/loop.py see/__main__.py scripts/run_dream_rsi.py tests/test_loop.py
git commit -m "Map SIGTERM to KeyboardInterrupt in the CLI entry points

A plain kill used to end the process at once: nothing frozen, agents and
sweeps orphaned. install_signal_handlers() gives SIGTERM the Ctrl-C path
and is called by see demo and scripts/run_dream_rsi.py; the library never
changes signal disposition on import."
```

### Task 8: Ledger, docs and the pinned count

**Files:**
- Modify: `reconstruction/GAPS.md` — §3 table: rewrite the `Interrupted iteration handling` row (`:68`) and append three rows after the `hard_max_grid` default row (`:72`).
- Modify: `.claude/CLAUDE.md` — the Conventions bullet that begins `- An interrupted iteration cannot be resumed:` (`:96-102`). `.agents/AGENTS.md` and `.gemini/GEMINI.md` are symlinks to it; do not touch them.
- Modify: `reconstruction/README.md` — status table rows `Online rollout: ...` (`:16`) and `Outer loop: ...` (`:17`).
- Modify: `.github/workflows/ci.yml` — `--expect 55` (`:38`) → `--expect 62`.

**Interfaces:**
- Consumes: no code symbols; this task describes what Tasks 1–7 landed (`CommandAgent` sessions and `terminate()`, `agent_timed_out`, `LiveQuestion._execute` cancellation, `runs/iterNNNN/partial/`, `_sweep` sessions, `archive_name` and the three-candidate guard, `install_signal_handlers()`).
- Produces: nothing; the plan's last task.

- [ ] **Step 1: Confirm the count is 62 and the gate is red at the old pin**

Run, from `reconstruction/`: `python -m pytest -q --junitxml=report.xml && python tools/check_junit.py report.xml --expect 55`
Expected: `62 passed`, then `check_junit` exits non-zero saying it expected 55 tests and found 62. Record its exact line. (`report.xml` is gitignored.)

- [ ] **Step 2: Rewrite the interrupted-iteration row and add three §3 rows**

In `reconstruction/GAPS.md`, replace the whole row that begins `| Interrupted iteration handling | not discussed |` with:

```markdown
| Interrupted iteration handling | not discussed | `online(t)` refuses when `runs/iterNNNN`, `trace_pool/iterNNNN` or the iteration's first archive `policy_dev/history/r{round+1}_tNN_m0` already exists, naming every one that does in a single message (delete them, never merge into them); the archive is checked because the round counter is persisted only on success, so a restart after a crash in `offline()` would otherwise recreate and overwrite the aborted iteration's archives (see the archive-name guard row) |
```

Append these three rows after the `hard_max_grid` default row (the last row of the table):

```markdown
| Interrupt semantics (SIGINT, SIGTERM mid-iteration) | not discussed | the batch in flight is cancelled: queued attempts never start, every running agent's process group is killed, and cells whose agent was killed are discarded (no new fail class); what was collected before the interrupt is frozen as `runs/iterNNNN/partial/{trace.json,live_episode.jsonl,live_cycle_manifest.json}` with the manifest's `error` naming the interrupt and `"partial": true`; `trace_pool/` is never written on that path, so replay scoring never sees a truncated tree; SIGTERM is mapped to `KeyboardInterrupt` by the CLI entry points (`install_signal_handlers()`), SIGKILL runs none of this; an in-process evaluator cannot be interrupted, so the return waits for evaluations already running |
| Agent and sweep child-process lifetime | not discussed | `CommandAgent` runs each agent CLI in its own session (`start_new_session=True`) and on timeout SIGTERMs the whole group, then SIGKILLs it after `kill_grace` (5 s); the result records `timed_out` and `eval/score.json` records `agent_timed_out`; a timed-out agent's attempt is scored as whatever program it left (its parent's copy if it wrote nothing) — whether that should count as `no_program` is open (C2); `_sweep` starts `see sweep` in its own session and kills the group on timeout or when an interrupt escapes; POSIX only (`os.killpg`); a child that calls `setsid` itself leaves the group and is not killed |
| Archive-name guard on restart | not discussed | `archive_name(round, t, m)` is the single source of the `rNNNN_tNN_mM` names; `online(t)` refuses when `policy_dev/history/<archive_name(round+1, t, 0)>` exists because `_archive` increments the round in memory and it is persisted only by `_deploy` and at the end of `offline()`; recovery deletes that iteration's `r*_tNN_m*` directories together with `runs/iterNNNN` and `trace_pool/iterNNNN` |
```

- [ ] **Step 3: Rewrite the CLAUDE.md conventions bullet**

In `.claude/CLAUDE.md`, replace the whole bullet from `- An interrupted iteration cannot be resumed:` through `keeps writing under \`policy_dev/history/\`.` with:

```markdown
- An interrupted iteration cannot be resumed: `online()` raises if `runs/iterNNNN/`,
  `trace_pool/iterNNNN/` or the iteration's first archive `policy_dev/history/rNNNN_tNN_m0/`
  already exists, naming every one that does. Ctrl-C and SIGTERM (the entry points call
  `install_signal_handlers()`) kill the running agents' process groups and freeze what was
  collected under `runs/iterNNNN/partial/`, never into the pool. To restart, delete the named
  directories and the aborted iteration's other `policy_dev/history/r*_tNN_m*` entries — the
  round counter is persisted only on success — and do not merge into them. Only after a SIGKILL
  of the parent (which runs no cleanup) check for a surviving `see sweep` process.
```

- [ ] **Step 4: Update the README status rows**

In `reconstruction/README.md`, replace the two status-table rows that begin `| Online rollout:` and `| Outer loop:` with:

```markdown
| Online rollout: workspaces, parallel workers, agent + evaluator | Sec. 3, Listing 1 | implemented; tested with scripted agents and the real Lasso evaluator; an agent timeout or an interrupt kills the agent's whole process group |
| Outer loop: online, pool, M versions, argmax deploy | Sec. 3, Fig. 1 | implemented; tested end to end with scripted agents; replay ≡ live pinned across seeds and both policies; deploy integrity verified by digest; an interrupt freezes the partial tree under `runs/`, never the pool, and a restart refuses |
```

Nothing else in the file changes.

- [ ] **Step 5: Move the CI pin**

In `.github/workflows/ci.yml`, change `- run: python tools/check_junit.py report.xml --expect 55` to `--expect 62`. Nothing else in the file changes.

- [ ] **Step 6: Confirm the gate is green at the new pin and no count is quoted in the docs**

Run, from `reconstruction/`: `ruff format --check . && ruff check . && pyright && python tools/extract_listings.py --check && python -m pytest -q --junitxml=report.xml && python tools/check_junit.py report.xml --expect 62`
Expected: all clean, `62 passed`, `check_junit` reports 62 tests and no skips.
Run, from the repository root: `grep -nE '47|55|62|passed|tests,|skip' .claude/CLAUDE.md reconstruction/README.md`
Expected: no output.

- [ ] **Step 7: Manual interrupt check (spec §10.3)**

Write `<scratchpad>/c1-sigterm-check.py` (under the session scratchpad, never the repository):

```python
"""SIGTERM during online(): the partial tree is frozen, the pool is empty, a restart refuses."""

import os
import shutil
import signal
import sys
import time

from see.loop import DreamRSI, LoopConfig, install_signal_handlers
from see.toy import ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task


class Slow(ScriptedDiscoveryAgent):
    def __call__(self, prompt, *, cwd, target):
        time.sleep(0.4)
        return super().__call__(prompt, cwd=cwd, target=target)


work = sys.argv[1]
shutil.rmtree(work, ignore_errors=True)
cfg = LoopConfig(workdir=work, max_parallelism=1, fallback_grid=(2, 2), hard_max_grid=(2, 2))
install_signal_handlers()
signal.setitimer(signal.ITIMER_REAL, 1.0)  # lands during online(1), after one or two cells
signal.signal(signal.SIGALRM, lambda *_: os.kill(os.getpid(), signal.SIGTERM))
try:
    DreamRSI(cfg, make_task(work), Slow(), ScriptedPolicyAgent()).run()
except KeyboardInterrupt as e:
    print("interrupted:", e)
partial = os.path.join(work, "runs", "iter0001", "partial")
print("partial files:", sorted(os.listdir(partial)))
print("pool entries:", sorted(os.listdir(os.path.join(work, "trace_pool"))))
try:
    DreamRSI(cfg, make_task(work), Slow(), ScriptedPolicyAgent()).online(1)
except RuntimeError as e:
    print("restart refused:", e)
shutil.rmtree(os.path.join(work, "runs", "iter0001"))
print("after deleting runs/iter0001, online(1) probes:", DreamRSI(cfg, make_task(work), Slow(), ScriptedPolicyAgent()).online(1)["probes"])
```

Run, from `reconstruction/` with the stub or real `generated/` present: `python <scratchpad>/c1-sigterm-check.py <scratchpad>/c1-sigterm-work`
Expected, in order: `interrupted: signal 15`; `partial files: ['live_cycle_manifest.json', 'live_episode.jsonl', 'trace.json']`; `pool entries: []`; `restart refused: .../runs/iter0001 exists: iteration 1 was interrupted or already ran; delete it, do not merge into it`; a final `probes:` count of 6. Record the output.

- [ ] **Step 8: Manual orphan check (spec §10.4)**

Run, from `reconstruction/`:

```bash
python -m see demo --workdir <scratchpad>/c1-orphan-work & PID=$!
sleep 1.2   # the toy demo is inside offline(1)'s sweeps by then
kill -TERM $PID; wait $PID; echo "demo exit $?"
sleep 0.5; pgrep -fl "see sweep" || echo "no see sweep survives"
```

Expected: the demo exits with a non-zero status (KeyboardInterrupt) and `no see sweep survives`. If the kill landed during `online()` instead, `runs/iter0001/partial/` exists and the orphan check is vacuous; rerun with `sleep 1.6`.

- [ ] **Step 9: Commit**

```bash
cd ..   # repository root
git add reconstruction/GAPS.md .claude/CLAUDE.md reconstruction/README.md .github/workflows/ci.yml
git commit -m "Record track C1's runtime-safety choices and move the CI count to 62

GAPS.md gains rows for the interrupt semantics, the agent and sweep
child-process lifetime, and the archive-name guard, and the
interrupted-iteration row now names the three directories the guard
checks. CLAUDE.md's recovery bullet describes the partial trace and the
one case (SIGKILL) that can still orphan a sweep. The README status rows
mention the new behaviour. The junit pin moves from 55 to 62, the only
place the count is quoted."
```


---

## Execution rulings (2026-09-25)

Recorded during subagent-driven execution; the spec was the binding authority. Where a ruling
changed the code a task prints above, the committed code is the one described here.

1. Task 5's `import time` goes where isort puts it; the "after `import shutil` once Task 6 adds
   it" note was stale (Task 6 adds no `shutil`).
2. `tests/test_loop.py` had 10 tests at the base, not 12: Task 4's file-level count is 11, Task
   6's is 13. The suite-level chain was authoritative.
3. Work happened in place on `spec/runtime-safety-c1` (a feature branch), as tracks A and B did.
4. Implementers were resumable Agent dispatches with one reviewer each; the final whole-branch
   review ran as a workflow with adversarial verification on the most capable model.
5. Task 1's `p.communicate()` drain after the kill was removed (it blocks on a `setsid` escapee
   holding the pipes and raises `UnicodeDecodeError` on a half-written multibyte character):
   `CommandAgent._close_pipes(p)` closes the pipes instead, and the `finally` kills the group when
   an exception escapes `communicate()` in the calling thread (a Ctrl-C during the main-thread
   policy-agent call), which `subprocess.run` used to do. Two tests pin both behaviours
   (`test_interrupt_during_an_agent_call_kills_its_process_group`,
   `test_timeout_returns_even_when_an_escapee_holds_the_pipes`), so the count chain became
   58, 59, 60, 61, 62, 63, 64 and the CI pin moved from 55 to 64. Task 2's `__call__` carries the
   same two corrections.
6. `.github/workflows/ci.yml` stayed at `--expect 55` until Task 8 moved it; the branch was not
   pushed before then.
7. Task 2's post-`communicate()` `_closed` check ran outside the lock, so a call that finished on
   its own just before `terminate()` could be reported as terminated. `terminate()` now claims a
   still-running call by removing its `Popen` from `_live` under the lock, skipping processes
   whose `poll()` is not `None`, and a call reports terminated only if it was claimed. The window
   is not pinned by a test (it cannot be hit deterministically without hooks).
8. Task 3's "two workers over four roots, two queued" scenario is unreachable: `probe_batch`
   caps a batch at `max_parallelism` and the pool has that many workers, so every attempt in a
   batch is running. The test became `test_interrupt_kills_running_agents_and_records_nothing`
   over a batch of two; `cancel_futures=True` and the entry guard in `_run_attempt` stay because a
   submitted job can still be cancelled between `submit` and a worker dequeuing it. Spec §4's
   "queued attempts never start" is vacuous for batches.
9. After `_cancel` killed the agents, each worker went on to evaluate the killed attempt's
   leftover program and wrote `eval/score.json`, so an interrupt waited for W serialized
   evaluator runs. A second `_interrupted` check after the agent call raises instead; the test
   triggers the interrupt from a helper thread once both stand-ins have forked (no timer) and
   asserts no score file exists.
10. Task 5's `Popen` sits before the `try`, so a failure to spawn the sweep subprocess (a host
    error such as `EMFILE`) propagates instead of scoring the version invalid. Kept on purpose:
    it describes the host, not the version, and a −∞ would misattribute it and skew the argmax.
    Recorded in GAPS.md §3.
11. GAPS.md's "Deploy-time code integrity" row (written in track B) still described a two-path
    guard; it now names all three paths and says recovery deletes every `r*_tNN_m*` of the aborted
    iteration.
12. `install_signal_handlers()` also maps SIGHUP, leaving it alone when it was inherited ignored (a
    nohup or setsid launch): with the agents and the sweep in their own sessions they no longer
    receive the terminal's hangup, so an unmapped SIGHUP that killed the parent orphaned them. The
    docs and the spec say "SIGKILL, or any signal the entry points do not map" instead of "only
    SIGKILL".
13. Spec §7's "an interrupt during the baseline evaluation or planning writes an empty partial
    trace" was never implementable as written (the partial freeze needs the live question, which
    needs the baseline): the baseline is now evaluated before `runs/iterNNNN/` is created, so such
    an interrupt leaves nothing under `runs/`; the spec bullet is amended.
14. The terminate path had no equivalent of Ruling 5's guard: a claimed call waited for EOF on
    pipes a descendant outside the group still held, up to the agent timeout. `__call__` now waits
    in one-second slices and a claimed call closes its pipes and returns terminated at the next
    slice; the unclaimed leader-exits-while-a-member-holds-the-pipes case still waits for EOF or
    the timeout (deferred).
15. `kill_process_group` sends SIGKILL unconditionally after the grace wait, a superset of spec
    §5's "if still alive". Tasks 1, 2, 3 and 8 landed as a task commit plus a review-fix commit
    each, not squashed; the owner decides the merge shape. Rulings 12–14 and 16 added four tests:
    the count is 68, pinned only in `.github/workflows/ci.yml`.
16. Agent and sweep output is decoded with `errors="replace"`: a process killed or exiting
    between the bytes of one character raised `UnicodeDecodeError` out of the call (pre-existing
    under `subprocess.run`, more likely now that kills are routine); the output is diagnostic text
    only. With its test the count is 68.

Deferred minors from the task reviews are listed in the final review's triage (see the pull
request), not here.
