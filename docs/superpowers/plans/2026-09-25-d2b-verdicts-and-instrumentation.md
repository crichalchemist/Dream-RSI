# D2's Verdicts and the Next Run's Instrumentation (Track D2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply the verdicts D2a's counts support on D1's four deferred changes, and give the next real-agent run the instrumentation D2a lacked:
- an attempt whose agent changed nothing fails as `no_program`;
- an evaluation repeat setting;
- each agent's stderr tail;
- a live-plan signal in replay;
- a launch record written only after the restart guard;
- a redacting evidence copy;
- report columns for all of it.

**Architecture:** Behaviour changes live in the loop:
- `see/live.py`: the untouched check, the stderr tail, `repeated()`;
- `see/loop.py`: `LoopConfig.eval_repeats`, `DreamRSI.check_iteration`;
- `see/objective.py`: the live-plan signal.

The runner gains `--eval-repeats` and checks the guard before recording a launch. `scripts/report_run.py` redacts what it copies and reads the new fields, saying "not measured" on older workdirs. No public name in `see/policy/` changes.

**Tech Stack:** Python 3.10+ standard library (the `see` package), pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-09-25-d2b-verdicts-and-instrumentation-design.md`, amended in this plan's commit (new section 13). Executors read both.

## Global Constraints

- The gate runs from `reconstruction/` inside its venv, and every task must pass it:
  - `ruff format --check .`;
  - `ruff check .`;
  - `pyright` (basic mode, 0 errors);
  - `python tools/extract_listings.py --check`;
  - `python -m pytest -q --junitxml=report.xml` with zero skips;
  - `python tools/check_junit.py report.xml --expect N`.

  N is the task's own measured count, named in its gate step: 128, 132, 134, 135, 138, 142 and 142 for Tasks 1–7. `.github/workflows/ci.yml` stays at `--expect 123` until Task 7 pins 142.
- No `xfail`, `skip`, `# noqa` or `# type: ignore`.
- Nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes. No workdir directory is renamed, and the agent callable's signature is unchanged.
- Every paper-silent choice gets a GAPS §3 row with its "How to change" cell and a hand-computed pin.
- Commits:
  - one commit per task;
  - a plain imperative subject;
  - the message is written to a file and committed with `git commit -F <file>`;
  - check `git diff --stat` before every commit;
  - **never add a `Co-Authored-By:` line, or any other line naming an agent, to a commit message.**
- SimpleTES is AGPL and stays unvendored. No attempt program enters the repository, nor anything under `reconstruction/generated/`. `reconstruction/evidence/` is committed run evidence: never edit it.
- Symbol work goes through Serena first (`.claude/CLAUDE.md`, "Code navigation: Serena first"). Load the `mcp__serena__*` tools via ToolSearch; their paths are relative to the repo root.
- Test names state the claim or outcome they protect. Tests that run the loop request the `stub_prompts` fixture.
- Edits are given as replace blocks. Each block's "replace" text occurs exactly once in its file at that point in the task, and blocks for one file are applied in the order given. A block marked **mid-line** begins partway through a long line (a GAPS or README table row): match it as a substring. Every block ends at the end of a line.

## Rulings made while planning

Each ruling is recorded in the spec's new section 13, in this plan's commit.

1. **Seven tasks.** Section 6's report work is split: redaction at copy time (Task 5) and the new report columns (Task 6). A reviewer could accept one and reject the other.
2. **`score.json` records `untouched` on every attempt**, as true or false. The report must tell "checked, and not untouched" from a workdir that predates the flag.
3. **A program the untouched check cannot read is left to the evaluator.** `_sha256` returns None on `OSError`, and None never matches.
   - Why: hashing it unguarded would raise a worker fault that abandons the whole batch, where the evaluator used to fail one cell.
   - Cost if wrong: none; such a program was never untouched.
4. **Where edits land.**
   - Task 1: the lifetime row's untouched clause, with the code that settles it.
   - Task 5: `.claude/CLAUDE.md`'s evidence line.
   - Task 7: every README change and the three remaining verdict clauses.
5. **Two coverage-preservation tests in Task 1 pass before it**, and are exempt from the spec's "fails first" criterion. They guard behaviour the untouched check must not change:
   - a timed-out agent that edited its program is still scored as a `timeout`;
   - an unreadable program still goes to the evaluator.
6. **The CI pin moves once**, in Task 7, as in D1. Earlier tasks check their own count in the gate step.

## File structure

- `reconstruction/see/live.py`:
  - `_sha256(path) -> str | None`;
  - the untouched check and `agent_stderr`/`untouched` in `score.json`, in `LiveQuestion._run_attempt`;
  - `repeated(task, k) -> TaskSpec`.
- `reconstruction/see/loop.py`:
  - `LoopConfig.eval_repeats` and its refusal;
  - `DreamRSI` wrapping its task;
  - `DreamRSI.check_iteration(t)`, split out of `online()`.
- `reconstruction/see/objective.py`: `Episode.live_plan`, `.beyond_support`, `.live_plan_error`; the extra `plan_grid` call in `run_episode`; `beyond_support` in `beta_sweep`'s report.
- `reconstruction/scripts/run_dream_rsi.py`: `--eval-repeats`; `eval_repeats` in the launch record; the guard before `record_launch`.
- `reconstruction/scripts/report_run.py`:
  - `redact`, `redact_json`, `_redacted_copy`;
  - the redacting `copy_evidence`;
  - the report redacted before it is written;
  - the new fields and columns.
- Tests:
  - `tests/test_loop.py` (+7, one changed);
  - `tests/test_command_agent.py` (+1, one changed);
  - `tests/test_run_dream_rsi.py` (+2);
  - `tests/test_ledger.py` (+2, a docstring);
  - `tests/test_objective.py` (one changed);
  - `tests/test_report_run.py` (+7, several changed).
- Docs: `reconstruction/GAPS.md`, `reconstruction/README.md` and `.claude/CLAUDE.md`. Config: `.github/workflows/ci.yml`.

Every block below was run in a throwaway worktree before this plan was written, one commit per task:
- at each task, the gate passed (ruff, format and pyright clean; 128, 132, 134, 135, 138, 142 and 142 tests, no skips);
- `python -m see demo` ran end to end after Task 7;
- the suite also passed under Python 3.10, except one pre-existing `verify_lasso` test that needs numpy.

A script re-parsed this plan and rebuilt every task's files from the one before, byte for byte.

---

### Task 1: An untouched attempt is `no_program`; the agent's stderr is kept

**Files:**
- Modify: `reconstruction/see/live.py` (import `hashlib`; `LiveQuestion._run_attempt`; new `_sha256`)
- Modify: `reconstruction/tests/test_loop.py`
  - import `PROGRAM`;
  - the crash test's children re-save their program;
  - four new tests.
- Modify: `reconstruction/tests/test_command_agent.py` (the process-group test's expectations; one new test)
- Modify: `reconstruction/tests/test_report_run.py` (the fixture's two untouched cells are now `no_program`)
- Modify: `reconstruction/GAPS.md`: §3 "Agent and sweep child-process lifetime" clause; new rows "Untouched attempt" and "Agent exit code"

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `see.live._sha256(path: str) -> str | None`: None when the file cannot be read.
  - Every attempt's `eval/score.json` gains two fields:
    - `"agent_stderr"`: the agent's stderr tail, or None;
    - `"untouched"`: a bool.
  - An untouched attempt's result has `no_program: True` and the error `"agent left its program unchanged (<first 200 characters of stderr>)"`. It is not evaluated.

Why: in the D2a run, agents stopped by a quota error left their programs as copied, and the loop evaluated those copies as fresh `ok` attempts: evaluation noise recorded as improvement. The digest is taken of the copy, not of the source, because agents run with the whole tree as their working directory, so another agent may rewrite the source mid-attempt.

- [ ] **Step 1: Write the failing tests.**

The process-group test's stand-in never touches its program, so it now pins the timed-out untouched case:

In `reconstruction/tests/test_command_agent.py`, replace:

````python
    LiveQuestion the attempt is scored as the program the agent left but recorded as the
    `timeout` failure it is, never as a valid non-improving attempt."""
````

with:

````python
    LiveQuestion the attempt, whose program the stand-in never touched, is `no_program` and is
    never evaluated, rather than scored as its parent's copy."""
````

In `reconstruction/tests/test_command_agent.py`, replace:

````python
    # the stand-in never touched the program, so the parent's copy (x=1.0) is what gets scored,
    # but the attempt is a timeout failure: not a success, and it never raises the ceiling
    assert (obs.cell_id, obs.score, obs.evaluated, obs.fail_class) == ("b0a0", 1.0, True, "timeout")
    assert obs.error == "agent timed out after 0.3s"
````

with:

````python
    # the stand-in never touched the program: there is nothing new to score, so it is not evaluated
    untouched = "agent left its program unchanged (agent timed out after 0.3s)"
    assert (obs.cell_id, obs.score, obs.evaluated) == ("b0a0", 0.0, False)
    assert (obs.fail_class, obs.error) == ("no_program", untouched)
````

In `reconstruction/tests/test_command_agent.py`, replace:

````python
    assert score["fail_class"] == "timeout" and score["combined_score"] == 1.0
    assert (node / "error.txt").read_text() == "agent timed out after 0.3s"
````

with:

````python
    assert score["fail_class"] == "no_program" and score["untouched"] is True
    assert (node / "error.txt").read_text() == untouched
````

In `reconstruction/tests/test_command_agent.py`, replace:

````python
        assert json.load(f)["agent_timed_out"] is True
````

with:

````python
        assert json.load(f)["agent_timed_out"] is True


def test_a_timed_out_agent_that_edited_its_program_is_still_scored_as_a_timeout(
    tmp_path, stub_prompts
):
    """What a timed-out agent changed is scored, but the attempt is the `timeout` failure it is:
    never a success, so it never raises the trace ceiling. Only an agent that changed nothing is
    `no_program`."""
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tree / "attempt_b000_a000" / PROGRAM
    script = """printf '{"x": 2.0}' > "$0"; sleep 30"""
    writes_then_hangs = ["sh", "-c", script, str(target), "{prompt}"]
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
    assert (obs.score, obs.evaluated, obs.fail_class) == (2.0, True, "timeout")
    assert obs.delta_vs_baseline is None  # not a success, although 2.0 beats the root's 1.0
````

In `test_loop.py`, the crash test's children used to leave their resumed program untouched, and are now untouched attempts. They re-save it as a real edit instead, so the test still proves where they resume from:

In `reconstruction/tests/test_loop.py`, replace:

````python
from see.toy import ScriptedDiscoveryAgent, ScriptedPolicyAgent, evaluate, make_task
````

with:

````python
from see.toy import PROGRAM, ScriptedDiscoveryAgent, ScriptedPolicyAgent, evaluate, make_task
````

In `reconstruction/tests/test_loop.py`, replace:

````python
        return {"returncode": 0}  # leaves the resumed program untouched
````

with:

````python
        path = os.path.join(target, PROGRAM)
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:  # re-saved: a real edit that keeps the resumed x
            json.dump({"x": x}, f, indent=1)
        return {"returncode": 0}
````

In `reconstruction/tests/test_loop.py`, replace:

````python
    assert obs[0].delta_vs_parent is None  # the parent failed
````

with:

````python
    assert obs[0].delta_vs_parent is None  # the parent failed


def _counting(task):
    """The toy task with an evaluator that records every call."""
    calls = []

    def evaluate(path):
        calls.append(path)
        return task.evaluate(path)

    return dataclasses.replace(task, evaluate=evaluate), calls


def test_an_agent_that_changes_nothing_is_no_program_and_never_evaluated(tmp_path, stub_prompts):
    """D2a: agents stopped by a quota error left their programs as copied, and the loop scored
    those copies as fresh `ok` attempts, so evaluation noise read as improvement. An attempt whose
    agent changed nothing, whether it returned or timed out, is `no_program` and not evaluated."""
    task, calls = _counting(make_task(str(tmp_path)))

    def changes_nothing(prompt, *, cwd, target):
        if target.endswith("attempt_b001_a000"):
            return {"returncode": None, "timed_out": True, "stderr": "agent timed out after 9s"}
        return {"returncode": 3, "stderr": "quota exhausted"}

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        task,
        changes_nothing,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=0,
    )
    obs = q.probe_batch(q.legal_roots())
    assert [(o.cell_id, o.fail_class, o.evaluated, o.score) for o in obs] == [
        ("b0a0", "no_program", False, 0.0),
        ("b1a0", "no_program", False, 0.0),
    ]
    assert calls == []
    node = tree / "attempt_b000_a000"
    with open(node / "eval" / "score.json") as f:
        score = json.load(f)
    assert (score["untouched"], score["agent_returncode"]) == (True, 3)
    assert score["error"] == "agent left its program unchanged (quota exhausted)"
    assert (node / PROGRAM).exists()  # kept, so a child resumes the same bytes


def test_a_source_edited_mid_attempt_does_not_hide_an_untouched_program(tmp_path, stub_prompts):
    """Agents run with the whole tree as their working directory, so another agent may rewrite
    the parent's program while this attempt runs. The check compares the attempt's program with
    the bytes copied in, not with the source as it stands afterwards."""
    task, calls = _counting(make_task(str(tmp_path)))

    def rewrites_its_parent(prompt, *, cwd, target):
        if target.endswith("_a001"):  # leaves its own copy alone, rewrites the parent's
            with open(os.path.join(cwd, "attempt_b000_a000", PROGRAM), "w") as f:
                json.dump({"x": 5.0}, f)
        else:
            with open(os.path.join(target, PROGRAM), "w") as f:
                json.dump({"x": 2.0}, f)
        return {"returncode": 0}

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        task,
        rewrites_its_parent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=1,
        branch_count=1,
        refine_count=1,
    )
    q.probe_batch(q.legal_roots())
    [obs] = q.probe_batch(q.legal_actions())
    assert (obs.cell_id, obs.fail_class, obs.evaluated) == ("b0a1", "no_program", False)
    assert len(calls) == 1  # the root only


def test_a_program_the_check_cannot_read_is_left_to_the_evaluator(tmp_path, stub_prompts):
    """An agent that replaces its program with something unreadable, here a directory, gets one
    failed cell from the evaluator, as before the untouched check existed, not a fault that
    abandons the whole batch."""

    def replaces_it_with_a_directory(prompt, *, cwd, target):
        path = os.path.join(target, PROGRAM)
        os.remove(path)
        os.mkdir(path)
        return {"returncode": 0}

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        replaces_it_with_a_directory,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=1,
        branch_count=1,
        refine_count=0,
    )
    [obs] = q.probe_batch(q.legal_roots())
    assert (obs.fail_class, obs.evaluated, obs.score) == ("code", True, 0.0)


def test_score_json_keeps_the_agents_stderr_tail(tmp_path, stub_prompts):
    """D2a's workdir kept each agent's exit code but not its stderr, so it could not tell a quota
    error from a lost login. The exit code stays a record, not a failure class: this agent exits
    3 after a real edit, and its attempt is an ordinary success."""

    def edits_then_exits_3(prompt, *, cwd, target):
        with open(os.path.join(target, PROGRAM), "w") as f:
            json.dump({"x": 2.0}, f)
        return {"returncode": 3, "stdout": "the reply", "stderr": "HTTP 429: quota exhausted"}

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        edits_then_exits_3,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=1,
        branch_count=1,
        refine_count=0,
    )
    [obs] = q.probe_batch(q.legal_roots())
    with open(tree / "attempt_b000_a000" / "eval" / "score.json") as f:
        score = json.load(f)
    assert (obs.fail_class, obs.score) == ("ok", 2.0)
    assert (score["agent_returncode"], score["agent_stderr"]) == (3, "HTTP 429: quota exhausted")
    assert score["untouched"] is False  # recorded either way, so a reader can tell it was checked
    assert "the reply" not in json.dumps(score)  # stdout can quote the program: never kept
````

The report fixture's two scripted untouched cells (b0a0 identical to the baseline, b1a1 identical to its parent and timed out) are now `no_program`. Iteration 1 therefore has 2 successes, and the evidence copy gains b0a0's `error.txt` (44 files):

In `reconstruction/tests/test_report_run.py`, replace:

````python
    ("iter0001", "attempt_b000_a000"): "untouched",  # identical to the baseline
    ("iter0001", "attempt_b001_a001"): "timeout",  # identical to its parent, agent timed out
````

with:

````python
    ("iter0001", "attempt_b000_a000"): "untouched",  # identical to the baseline: no_program
    ("iter0001", "attempt_b001_a001"): "timeout",  # identical to its parent, timed out: no_program
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
            {"iteration": 1, "cell": "b0a0", "source": "baseline", "fail_class": "ok"}
            | {"agent_timed_out": False, "agent_returncode": 0}
            | {"evaluated": True, "score": 1.0, "source_score": 1.0},
            {"iteration": 1, "cell": "b1a1", "source": "parent", "fail_class": "timeout"}
            | {"agent_timed_out": True, "agent_returncode": None}
            | {"evaluated": True, "score": 1.1, "source_score": 1.1},
````

with:

````python
            {"iteration": 1, "cell": "b0a0", "source": "baseline", "fail_class": "no_program"}
            | {"agent_timed_out": False, "agent_returncode": 0}
            | {"evaluated": False, "score": 0.0, "source_score": 1.0},
            {"iteration": 1, "cell": "b1a1", "source": "parent", "fail_class": "no_program"}
            | {"agent_timed_out": True, "agent_returncode": None}
            | {"evaluated": False, "score": 0.0, "source_score": 1.1},
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
        {"attempts": 4, "successes": 3, "fail_classes": {"ok": 3, "timeout": 1}}
        | {"agent_timeouts": 1, "no_program": 0, "evaluator_crashed": 0}
````

with:

````python
        {"attempts": 4, "successes": 2, "fail_classes": {"no_program": 2, "ok": 2}}
        | {"agent_timeouts": 1, "no_program": 2, "evaluator_crashed": 0}
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
    """43 files: state.json and launches.jsonl; three per frozen iteration (6); eight score.json;
    two error.txt (the timeout, the deleted program); seven of eight proposals; method.py and two
    sweep files for each of six versions (18). SimpleTES programs are AGPL and stay out."""
    report, out = toy_run
    copied = [f for _, _, files in os.walk(out / "workdir") for f in files]
    assert PROGRAM not in copied
    assert len(copied) == 43
    assert report["evidence"] == {
        "copied": 43,
````

with:

````python
    """44 files: state.json and launches.jsonl; three per frozen iteration (6); eight score.json;
    three error.txt (the two untouched attempts, the deleted program); seven of eight proposals;
    method.py and two sweep files for each of six versions (18). SimpleTES programs are AGPL and
    stay out."""
    report, out = toy_run
    copied = [f for _, _, files in os.walk(out / "workdir") for f in files]
    assert PROGRAM not in copied
    assert len(copied) == 44
    assert report["evidence"] == {
        "copied": 44,
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_loop.py tests/test_command_agent.py tests/test_report_run.py`

Expected: 7 failed.
- `test_command_agent.py`: `test_agent_timeout_kills_the_whole_process_group`.
- `test_loop.py`:
  - `test_an_agent_that_changes_nothing_is_no_program_and_never_evaluated`;
  - `test_a_source_edited_mid_attempt_does_not_hide_an_untouched_program`;
  - `test_score_json_keeps_the_agents_stderr_tail`.
- `test_report_run.py`:
  - `test_an_untouched_attempt_is_reported_with_its_source_and_both_scores`;
  - `test_the_health_table_counts_each_iterations_outcomes`;
  - `test_the_evidence_subset_carries_no_program_and_withholds_a_quoted_one`.

The new `test_a_timed_out_agent_that_edited_its_program_is_still_scored_as_a_timeout` and `test_a_program_the_check_cannot_read_is_left_to_the_evaluator` pass already: they guard behaviour this task must keep (ruling 5).

- [ ] **Step 3: Implement the check and the stderr tail**

In `reconstruction/see/live.py`, replace:

````python
import concurrent.futures
import dataclasses
import json
````

with:

````python
import concurrent.futures
import dataclasses
import hashlib
import json
````

In `reconstruction/see/live.py`, replace:

````python

def oriented_score(task: TaskSpec, result: dict) -> float:
````

with:

````python

def _sha256(path: str) -> str | None:
    """The file's digest, or None when it cannot be read: the evaluator then judges it."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def oriented_score(task: TaskSpec, result: dict) -> float:
````

In `reconstruction/see/live.py`, replace:

````python
            self._resume_from(meta.branch, meta.attempt), os.path.join(node, t.eval_program)
        )
        prompt = exploration_prompt(
````

with:

````python
            self._resume_from(meta.branch, meta.attempt), os.path.join(node, t.eval_program)
        )
        # the copy, not the source: agents run in the whole tree, so another may edit the source
        copied = _sha256(os.path.join(node, t.eval_program))
        prompt = exploration_prompt(
````

In `reconstruction/see/live.py`, replace:

````python
            raise RuntimeError(f"attempt b{meta.branch}a{meta.attempt} killed with its batch")
````

with:

````python
            raise RuntimeError(f"attempt b{meta.branch}a{meta.attempt} killed with its batch")
        digest = _sha256(program) if os.path.exists(program) else None
        untouched = digest is not None and digest == copied
````

In `reconstruction/see/live.py`, replace:

````python
                "error": f"agent left no program ({(run or {}).get('stderr', '')[:200]})",
````

with:

````python
                "error": f"agent left no program ({(run or {}).get('stderr', '')[:200]})",
            }
        elif untouched:  # the agent changed nothing: there is nothing new to score
            stderr = (run or {}).get("stderr", "")
            result = {
                "combined_score": 0.0,
                "no_program": True,
                "error": f"agent left its program unchanged ({stderr[:200]})",
````

In `reconstruction/see/live.py`, replace:

````python
                    "agent_timed_out": bool(run.get("timed_out")) if run else False,
````

with:

````python
                    "agent_timed_out": bool(run.get("timed_out")) if run else False,
                    "agent_stderr": run.get("stderr") if run else None,
                    "untouched": untouched,
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_loop.py tests/test_command_agent.py tests/test_report_run.py`
Expected: all pass.

- [ ] **Step 5: Ledger**

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
the evaluator's `validity` when one is reported, else false; its parent's copy is what gets scored if it wrote nothing, and whether an untouched program should count as `no_program` is D2 (pinned in `test_command_agent.py::test_agent_timeout_kills_the_whole_process_group` and `::test_a_timed_out_agent_that_left_a_broken_program_keeps_the_evaluators_verdict`); `_sweep` starts `see sweep` in its own session and kills the group on timeout or when an interrupt escapes; a failure to spawn the sweep subprocess (a host error such as `EMFILE`) propagates instead of scoring the version invalid, because it describes the host, not the version; POSIX only (`os.killpg`); a child that calls `setsid` itself leaves the group and is not killed | `CommandAgent(timeout=, kill_grace=)`; `run_dream_rsi.py --agent-timeout`; the mechanism is fixed in `see/live.py` |
````

with:

````markdown
the evaluator's `validity` when one is reported, else false; one that changed nothing is an untouched attempt (see that row), `no_program` and never evaluated (pinned in `test_command_agent.py::test_a_timed_out_agent_that_edited_its_program_is_still_scored_as_a_timeout`, `::test_a_timed_out_agent_that_left_a_broken_program_keeps_the_evaluators_verdict` and `::test_agent_timeout_kills_the_whole_process_group`); `_sweep` starts `see sweep` in its own session and kills the group on timeout or when an interrupt escapes; a failure to spawn the sweep subprocess (a host error such as `EMFILE`) propagates instead of scoring the version invalid, because it describes the host, not the version; POSIX only (`os.killpg`); a child that calls `setsid` itself leaves the group and is not killed | `CommandAgent(timeout=, kill_grace=)`; `run_dream_rsi.py --agent-timeout`; the mechanism is fixed in `see/live.py` |
````

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
--eigen-include` (search score only); the compiler: `PATH` |
````

with:

````markdown
--eigen-include` (search score only); the compiler: `PATH` |
| Untouched attempt | not discussed; Listing 1's agent resumes the parent's saved workspace | the sha256 of the program copied into the attempt's directory is taken before the agent runs (the copy, not the source: agents run in the whole tree, so another agent may rewrite the source meanwhile); an attempt whose program still matches it when the agent returns, timed out or not, is `no_program` with the error "agent left its program unchanged (…)" (`score.json` records `untouched`, true or false, for every attempt): it is not evaluated, scores 0.0, is never a success and never raises the trace ceiling; its file stays, so its children resume the same bytes; a program the check cannot read is left to the evaluator, as before. In the D2a run, 1 of 16 attempts in iteration 1 and all 4 in iteration 2's trace were untouched, and the latter scored `ok` at up to +12.7% on evaluation noise alone (§5, "D2a run"; pinned in `test_loop.py::test_an_agent_that_changes_nothing_is_no_program_and_never_evaluated`, `::test_a_source_edited_mid_attempt_does_not_hide_an_untouched_program` and `::test_a_program_the_check_cannot_read_is_left_to_the_evaluator`) | fixed in `see/live.py` (`_run_attempt`) |
| Agent exit code | not discussed | recorded as `agent_returncode` in `score.json`, beside `agent_stderr`, the stderr tail `CommandAgent` returns (at most 4000 characters; stdout, the agent's reply, can quote the program and is not kept); never a failure class: the D2a run's best cell exited 3 on a quota error after editing its program, so failing on the exit code would have discarded it, while the untouched rule already fails the attempts that changed nothing (pinned in `test_loop.py::test_score_json_keeps_the_agents_stderr_tail`) | fixed in `see/live.py` (`_run_attempt`) |
````

- [ ] **Step 6: Run the gate**

Run the Global Constraints gate with `--expect 128`. Expected: `128 passed`, `junit: 128 tests, no skips`, ruff and pyright clean.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 5 files above. Then:

```bash
cat > /tmp/d2b-t1-msg.txt <<'EOF'
Fail an untouched attempt as no_program and keep the agent's stderr

An attempt whose program still matches the bytes copied in when its
agent returns, timed out or not, is no_program and is not evaluated;
in the D2a run such copies scored ok on evaluation noise alone. The
check hashes the copy, not the source, because agents run in the whole
tree. score.json keeps the agent's stderr tail beside its exit code,
which stays a record rather than a failure class.
EOF
git add reconstruction/see/live.py reconstruction/tests/test_loop.py \
  reconstruction/tests/test_command_agent.py reconstruction/tests/test_report_run.py \
  reconstruction/GAPS.md
git commit -F /tmp/d2b-t1-msg.txt
```

---

### Task 2: An evaluation repeat setting that keeps the median run

**Files:**
- Modify: `reconstruction/see/live.py` (new `repeated`, before `_sha256`)
- Modify: `reconstruction/see/loop.py` (import `repeated`; `LoopConfig.eval_repeats` and its refusal; `DreamRSI.__init__`)
- Modify: `reconstruction/scripts/run_dream_rsi.py` (`--eval-repeats`; the launch record's `config`; `LoopConfig(...)` in `main`)
- Modify: `reconstruction/tests/test_loop.py` (import `repeated`; three new tests)
- Modify: `reconstruction/tests/test_run_dream_rsi.py` (one new test)
- Modify: `reconstruction/GAPS.md` (new §3 row "Evaluation repeats")

**Interfaces:**
- Consumes: Task 1's `_sha256` (the anchor `repeated` is inserted before).
- Produces:
  - `see.live.repeated(task: TaskSpec, k: int) -> TaskSpec`. Its `evaluate`:
    - returns the first run whose `error` is not None, with `repeat_scores` of the runs so far;
    - otherwise returns the run with the median `combined_score`, with all k scores in run order in `repeat_scores`.
  - `LoopConfig.eval_repeats: int = 1`. `__post_init__` raises `ValueError("eval_repeats … is not an odd count of at least 1")` unless it is an odd int ≥ 1.
  - `DreamRSI.task` is `repeated(task, k)` when k > 1, so `baseline_score()` and every attempt use it.
  - Runner flag `--eval-repeats` (`a.eval_repeats`, default 1) and the launch record's `config["eval_repeats"]`.

Why: D2a's eight evaluations of one unchanged program spread 14%, about the size of the run's improvement. The default of 1 keeps the paper's single evaluation, so no published number moves. With k odd, the median is one real run, whatever the task's direction.

- [ ] **Step 1: Write the failing tests**

In `reconstruction/tests/test_loop.py`, replace:

````python
from see.live import LiveQuestion
````

with:

````python
from see.live import LiveQuestion, repeated
````

In `reconstruction/tests/test_loop.py`, replace:

````python
    assert "the reply" not in json.dumps(score)  # stdout can quote the program: never kept
````

with:

````python
    assert "the reply" not in json.dumps(score)  # stdout can quote the program: never kept


def test_repeats_score_the_median_run_and_stop_at_the_first_error(tmp_path):
    """D2a's eight evaluations of one unchanged program spread 14%, about the size of the
    improvement. With k repeats a program scores its median run; a run that fails ends the repeats
    and is returned as it is, so a failure is never averaged away."""
    runs = iter(
        [
            {"combined_score": 0.3, "run": 1},
            {"combined_score": 0.1, "run": 2},
            {"combined_score": 0.2, "run": 3},
            {"combined_score": 0.4, "run": 4},
            {"combined_score": 0.0, "error": "ValueError: boom", "run": 5},
            {"combined_score": 0.9, "run": 6},
        ]
    )
    calls = []

    def evaluate(path):
        calls.append(path)
        return next(runs)

    three = repeated(dataclasses.replace(make_task(str(tmp_path)), evaluate=evaluate), 3)
    assert three.evaluate("p") == {
        "combined_score": 0.2,
        "run": 3,
        "repeat_scores": [0.3, 0.1, 0.2],
    }
    assert three.evaluate("p") == {
        "combined_score": 0.0,
        "error": "ValueError: boom",
        "run": 5,
        "repeat_scores": [0.4, 0.0],
    }
    assert len(calls) == 5  # the sixth run never happened


def test_an_even_zero_or_negative_repeat_count_is_refused_when_the_config_is_built(tmp_path):
    for k in (0, 2, -1):
        with pytest.raises(ValueError, match="not an odd count of at least 1"):
            LoopConfig(workdir=str(tmp_path), eval_repeats=k)


def test_the_baseline_is_scored_with_the_same_repeats_as_the_attempts(tmp_path, stub_prompts):
    """Every improvement is measured against the baseline, so it is scored the way attempts are."""
    task, calls = _counting(make_task(str(tmp_path)))
    work = tmp_path / "w"
    cfg = LoopConfig(
        workdir=str(work),
        max_parallelism=1,
        fallback_grid=(1, 0),
        hard_max_grid=(1, 0),
        eval_repeats=3,
    )
    DreamRSI(cfg, task, ScriptedDiscoveryAgent(), ScriptedPolicyAgent()).online(1)
    with open(work / "baseline_eval.json") as f:
        assert len(json.load(f)["repeat_scores"]) == 3
    node = work / "runs" / "iter0001" / "tree" / "attempt_b000_a000"
    with open(node / "eval" / "score.json") as f:
        assert len(json.load(f)["repeat_scores"]) == 3
    assert len(calls) == 6  # three for the baseline, three for the one attempt
````

In `reconstruction/tests/test_run_dream_rsi.py`, replace:

````python

def test_an_env_wrapped_agent_reports_the_version_of_the_cli_it_wraps():
````

with:

````python

def test_the_launch_record_carries_eval_repeats(tmp_path):
    """The report reads the repeats in force from the last launch; the default is the paper's
    single evaluation."""
    assert _record(tmp_path)["config"]["eval_repeats"] == 1
    a = runner.build_parser().parse_args([*ARGS, "--eval-repeats", "3"])
    task = TaskSpec("lasso_path", str(tmp_path), "init_program.py", "p.txt", lambda p: {})
    agent = CommandAgent(["python3", "-c", "{prompt}"])
    assert runner.launch_record(a, ARGS, task, agent, agent)["config"]["eval_repeats"] == 3


def test_an_env_wrapped_agent_reports_the_version_of_the_cli_it_wraps():
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_loop.py tests/test_run_dream_rsi.py`

Expected: collection of `tests/test_loop.py` fails with `ImportError: cannot import name 'repeated' from 'see.live'`. Then run `python -m pytest -q tests/test_run_dream_rsi.py`. Expected: `test_the_launch_record_carries_eval_repeats` fails with `KeyError: 'eval_repeats'`.

- [ ] **Step 3: Implement**

In `reconstruction/see/live.py`, replace:

````python
        for pipe in (p.stdout, p.stderr):
            if pipe is not None:
                pipe.close()


````

with:

````python
        for pipe in (p.stdout, p.stderr):
            if pipe is not None:
                pipe.close()


def repeated(task: TaskSpec, k: int) -> TaskSpec:
    """``task`` with an evaluator that runs up to ``k`` times (odd) and keeps the median run.

    The first run that reports an ``error`` is returned as it is, so a failure is never averaged
    away; otherwise the run with the median ``combined_score`` is. Either way ``repeat_scores``
    lists the scores of the runs made, in order. With ``k`` odd the median is one real run,
    whatever the task's direction.
    """
    inner = task.evaluate

    def evaluate(path: str) -> dict:
        runs = []
        for _ in range(k):
            runs.append(dict(inner(path)))
            if runs[-1].get("error") is not None:
                return {**runs[-1], "repeat_scores": [r["combined_score"] for r in runs]}
        middle = sorted(runs, key=lambda r: r["combined_score"])[k // 2]
        return {**middle, "repeat_scores": [r["combined_score"] for r in runs]}

    return dataclasses.replace(task, evaluate=evaluate)


````

In `reconstruction/see/loop.py`, replace:

````python
from see.live import LiveQuestion, TaskSpec, kill_process_group, oriented_score
````

with:

````python
from see.live import LiveQuestion, TaskSpec, kill_process_group, oriented_score, repeated
````

In `reconstruction/see/loop.py`, replace:

````python

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective {self.objective!r} is not one of {OBJECTIVES}")
````

with:

````python
    eval_repeats: int = 1  # evaluations per program, median kept; 1 is the paper's single one

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective {self.objective!r} is not one of {OBJECTIVES}")
        k = self.eval_repeats
        if not (isinstance(k, int) and k >= 1 and k % 2 == 1):  # odd: the median is one real run
            raise ValueError(f"eval_repeats {k!r} is not an odd count of at least 1")
````

In `reconstruction/see/loop.py`, replace:

````python
        self.c, self.task = config, task
        self.discovery_agent, self.policy_agent = discovery_agent, policy_agent
````

with:

````python
        self.c, self.task = config, task
        if config.eval_repeats > 1:  # the baseline and every attempt are scored the same way
            self.task = repeated(task, config.eval_repeats)
        self.discovery_agent, self.policy_agent = discovery_agent, policy_agent
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
            "agent_timeout": a.agent_timeout,
````

with:

````python
            "agent_timeout": a.agent_timeout,
            "eval_repeats": a.eval_repeats,
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
    ap.add_argument(
        "--eigen-include",
````

with:

````python
    ap.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="evaluations per program, odd; the median run is kept (default: 1, as the paper)",
    )
    ap.add_argument(
        "--eigen-include",
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
        hard_max_grid=tuple(a.hard_max),
        objective=a.objective,
    )
````

with:

````python
        hard_max_grid=tuple(a.hard_max),
        objective=a.objective,
        eval_repeats=a.eval_repeats,
    )
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_loop.py tests/test_run_dream_rsi.py`
Expected: all pass.

- [ ] **Step 5: Ledger**

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
`test_loop.py::test_score_json_keeps_the_agents_stderr_tail`) | fixed in `see/live.py` (`_run_attempt`) |
````

with:

````markdown
`test_loop.py::test_score_json_keeps_the_agents_stderr_tail`) | fixed in `see/live.py` (`_run_attempt`) |
| Evaluation repeats | not discussed | one evaluation per program by default; with k > 1 (k odd) each program is evaluated k times and the median run is kept, with every run's score in `repeat_scores`, and the first run that reports an error ends the repeats and is kept as it is, so a failure is never averaged away; the baseline is scored the same way, but once per workdir (`baseline_eval.json` is cached, so a restart with another k keeps the first baseline); an attempt's k runs hold the evaluation lock together. In the D2a run, eight evaluations of the unchanged baseline spread 14.0%, about the size of the run's improvement (§5, "D2a run"; pinned in `test_loop.py::test_repeats_score_the_median_run_and_stop_at_the_first_error`, `::test_an_even_zero_or_negative_repeat_count_is_refused_when_the_config_is_built` and `::test_the_baseline_is_scored_with_the_same_repeats_as_the_attempts`) | `LoopConfig.eval_repeats`; `run_dream_rsi.py --eval-repeats` |
````

- [ ] **Step 6: Run the gate**

Run the Global Constraints gate with `--expect 132`. Expected: `132 passed`, `junit: 132 tests, no skips`, ruff and pyright clean. Then run `python scripts/run_dream_rsi.py --help`: it must list `--eval-repeats`.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 6 files above. Then:

```bash
cat > /tmp/d2b-t2-msg.txt <<'EOF'
Add an evaluation repeat setting that keeps the median run

LoopConfig.eval_repeats (run_dream_rsi.py --eval-repeats) evaluates
each program k times, k odd, and keeps the median run with every
score in repeat_scores; the first failing run ends the repeats. The
default of 1 is the paper's single evaluation. DreamRSI wraps its task
once, so the baseline and the attempts are scored the same way, and
the launch record carries k.
EOF
git add reconstruction/see/live.py reconstruction/see/loop.py reconstruction/scripts/run_dream_rsi.py \
  reconstruction/tests/test_loop.py reconstruction/tests/test_run_dream_rsi.py reconstruction/GAPS.md
git commit -F /tmp/d2b-t2-msg.txt
```

---

### Task 3: Each replay episode records the plan the policy would run live

**Files:**
- Modify: `reconstruction/see/objective.py` (three `Episode` fields; the extra call in `run_episode`; `beyond_support` in `beta_sweep`'s report)
- Modify: `reconstruction/tests/test_ledger.py` (`_support_context`; two new tests)
- Modify: `reconstruction/tests/test_objective.py` (the rejected-plan comparison)
- Modify: `reconstruction/GAPS.md` (new §3 row "Live-plan signal in replay")

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `see.objective.Episode` gains three fields:
    - `live_plan: dict | None`, of the form `{"branch_count", "refine_count", "fallback"}`;
    - `beyond_support: bool`;
    - `live_plan_error: str | None`.
  - The executions file carries them through `dataclasses.asdict`.
  - `beta_sweep`'s report gains `"beyond_support": bool` beside `"out_of_support"`.

Why: Listing 2 lets `plan_grid` read the replay-only support fields `trace_branch_count`/`trace_refine_count`, and D2a's policies clamped their plans to them. No episode was ever clipped, even though the deployed version planned 4 x 4 live over 4 x 3 trees.

After the episode is scored, `run_episode`:
1. calls `plan_grid` again with both fields cleared (the context `online()` gives it live);
2. validates the plan as `online()` does;
3. records the grid.

The extra call never changes the episode's score, error or validity.

- [ ] **Step 1: Write the failing tests**

In `reconstruction/tests/test_ledger.py`, replace:

````python

def test_the_floor_is_reswept_for_reference_and_never_deployed(tmp_path, stub_prompts):
````

with:

````python

def _support_context() -> GridPlanningContext:
    """Live caps of 8 x 8 over a recorded 5 x 6 tree, as the loop's sweep passes them."""
    return GridPlanningContext(
        history=(),
        fallback_branch_count=5,
        fallback_refine_count=6,
        hard_max_branch_count=8,
        hard_max_refine_count=8,
        max_parallelism=5,
        trace_branch_count=5,
        trace_refine_count=6,
    )


def test_a_policy_that_clamps_in_replay_but_plans_wider_live_is_flagged_with_its_reward_unchanged():
    """D2a's deployed version clamped its plan to the replay-only trace fields, so none of its
    episodes was out of support, while live it planned deeper than any recorded tree. Each episode
    also records the plan the policy makes with those fields cleared, as online() gives it; the
    flag never changes what the episode scores."""
    trace = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)
    context = _support_context()

    class Clamps(ParallelRefine):
        def plan_grid(self, context):
            w, r = context.trace_branch_count, context.trace_refine_count
            if w is None or r is None:  # live: no recorded tree to stay inside
                return GridPlan(8, 8, reason="as wide and deep as the caps allow")
            return GridPlan(w, r, reason="clamped to the recorded tree")

    class Clipped(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(5, 6, reason="the tree's own grid")

    clamps = run_episode(Clamps(None), trace, context, record=True)
    clipped = run_episode(Clipped(None), trace, context, record=True)
    assert not clamps.out_of_support and (clamps.beyond_support, clipped.beyond_support) == (
        True,
        False,
    )
    assert clamps.live_plan == {"branch_count": 8, "refine_count": 8, "fallback": False}
    assert (clamps.probes, clamps.best, clamps.attainment, clamps.log) == (
        clipped.probes,
        clipped.best,
        clipped.attainment,
        clipped.log,
    )
    report, _ = beta_sweep(Clamps, [trace], context_for=lambda t: context, betas=(0.5,))
    assert (report["out_of_support"], report["beyond_support"], report["valid"]) == (
        False,
        True,
        True,
    )


def test_a_live_plan_that_raises_is_recorded_not_an_episode_error():
    """The extra plan_grid call runs after the replay episode is scored and cannot fail it."""
    trace = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)

    class RaisesLive(ParallelRefine):
        def plan_grid(self, context):
            if context.trace_branch_count is None:
                raise RuntimeError("no support fields")
            return GridPlan(5, 6, reason="the tree's own grid")

    episode = run_episode(RaisesLive(None), trace, _support_context())
    assert (episode.error, episode.live_plan, episode.beyond_support) == (None, None, False)
    assert "RuntimeError: no support fields" in (episode.live_plan_error or "")


def test_the_floor_is_reswept_for_reference_and_never_deployed(tmp_path, stub_prompts):
````

The rejected-plan test compares whole episodes. Its rejected plan is rejected live too, so its live plan is the fallback, marked as such:

In `reconstruction/tests/test_objective.py`, replace:

````python
    assert rejected == dataclasses.replace(explicit, plan=None)
````

with:

````python
    # rejected live too, so its live plan is the same fallback grid, marked as the fallback
    live = {"branch_count": 2, "refine_count": 2, "fallback": True}
    assert rejected == dataclasses.replace(explicit, plan=None, live_plan=live)
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_ledger.py tests/test_objective.py`

Expected: 3 failed.
- `test_ledger.py`:
  - `test_a_policy_that_clamps_in_replay_but_plans_wider_live_is_flagged_with_its_reward_unchanged`;
  - `test_a_live_plan_that_raises_is_recorded_not_an_episode_error`.
- `test_objective.py`: `test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace`.

- [ ] **Step 3: Implement**

In `reconstruction/see/objective.py`, replace:

````python
    out_of_support: bool
    error: str | None
    log: list


````

with:

````python
    out_of_support: bool
    error: str | None
    log: list
    live_plan: dict | None  # the grid the policy would run live, trace fields cleared
    beyond_support: bool  # that grid is wider or deeper than the recorded tree
    live_plan_error: str | None


````

In `reconstruction/see/objective.py`, replace:

````python
            q = ReplayQuestion(trace)
    probes = q.budget_spent
````

with:

````python
            q = ReplayQuestion(trace)
    live_plan, beyond_support, live_plan_error = None, False, None
    if use_plan:  # informational, after the episode is scored: what online() would run
        live = dataclasses.replace(context, trace_branch_count=None, trace_refine_count=None)
        try:
            asked = validate_plan(policy.plan_grid(live), live)
            b = asked.branch_count if asked else live.fallback_branch_count
            r = asked.refine_count if asked else live.fallback_refine_count
            live_plan = {"branch_count": b, "refine_count": r, "fallback": asked is None}
            beyond_support = b > trace.grid[0] or r > trace.grid[1]
        except Exception:
            live_plan_error = traceback.format_exc(limit=4)
    probes = q.budget_spent
````

In `reconstruction/see/objective.py`, replace:

````python
        out_of_support=q.out_of_support,
        error=error,
        log=q.episode,
    )

````

with:

````python
        out_of_support=q.out_of_support,
        error=error,
        log=q.episode,
        live_plan=live_plan,
        beyond_support=beyond_support,
        live_plan_error=live_plan_error,
    )

````

In `reconstruction/see/objective.py`, replace:

````python
        "out_of_support": any(e.out_of_support for e in executions),
````

with:

````python
        "out_of_support": any(e.out_of_support for e in executions),
        "beyond_support": any(e.beyond_support for e in executions),
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_ledger.py tests/test_objective.py`
Expected: all pass.

- [ ] **Step 5: Ledger**

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
`LoopConfig.eval_repeats`; `run_dream_rsi.py --eval-repeats` |
````

with:

````markdown
`LoopConfig.eval_repeats`; `run_dream_rsi.py --eval-repeats` |
| Live-plan signal in replay | Listing 2 lets `plan_grid` read the replay support fields `trace_branch_count`/`trace_refine_count`, and says a plan beyond them "cannot earn replay reward" | after each replay episode is scored, `run_episode` calls `plan_grid` again with both trace fields cleared, the context `online()` gives it live, and validates the plan as `online()` does (a rejected or absent plan becomes the fallback grid); the episode records that grid as `live_plan`, `beyond_support` when it is wider or deeper than the recorded tree, and `live_plan_error` if the call raised; informational only, it never changes the episode's score, error or validity; `beta_sweep.json` carries `beyond_support` beside `out_of_support`; it means something only under live caps, as the loop's sweep and `see sweep` pass them, and is always false under `default_context`, whose caps are the trace's own grid. In the D2a run no version replayed on a clipped episode, because the policies clamped their plans to the trace fields, while the deployed version planned 4 x 4 live over 4 x 3 trees (§5, "D2a run"; pinned in `test_ledger.py::test_a_policy_that_clamps_in_replay_but_plans_wider_live_is_flagged_with_its_reward_unchanged` and `::test_a_live_plan_that_raises_is_recorded_not_an_episode_error`) | fixed in `see/objective.py` (`run_episode`, `beta_sweep`) |
````

- [ ] **Step 6: Run the gate**

Run the Global Constraints gate with `--expect 134`. Expected: `134 passed`, `junit: 134 tests, no skips`, ruff and pyright clean.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 4 files above. Then:

```bash
cat > /tmp/d2b-t3-msg.txt <<'EOF'
Record the plan each replayed policy would run live

After a replay episode is scored, run_episode asks plan_grid again
with the trace support fields cleared, as online() does, and records
the validated grid, whether it goes beyond the recorded tree, and any
error from the call. The signal is informational and never changes a
score. D2a's policies clamped to those fields in replay, so clipping
alone could not show that the deployed one planned deeper live.
EOF
git add reconstruction/see/objective.py reconstruction/tests/test_ledger.py \
  reconstruction/tests/test_objective.py reconstruction/GAPS.md
git commit -F /tmp/d2b-t3-msg.txt
```

---

### Task 4: The restart guard runs before a launch is recorded

**Files:**
- Modify: `reconstruction/see/loop.py` (`online()`'s refusals move into a new `DreamRSI.check_iteration`, which `online()` calls first)
- Modify: `reconstruction/scripts/run_dream_rsi.py` (`main`: build the loop, check, then record)
- Modify: `reconstruction/tests/test_run_dream_rsi.py` (imports; one new test)
- Modify: `reconstruction/GAPS.md` (§3 rows "Interrupted iteration handling" and "Deploy-time code integrity")

**Interfaces:**
- Consumes: nothing new.
- Produces: `DreamRSI.check_iteration(t: int) -> None`. It raises the same `RuntimeError`s `online()` raised, with the same messages:
  - "… exist(s): iteration t was interrupted or already ran; …";
  - "… changed since it was deployed …".

Why: `report_run.py` takes the caps from the last line of `launches.jsonl`. A launch the loop refuses must therefore append nothing, and in D2a the launch was recorded before the guard ran.

- [ ] **Step 1: Write the failing test**

In `reconstruction/tests/test_run_dream_rsi.py`, replace:

````python
from see.live import CommandAgent, TaskSpec
from see.loader import load_module_from_path
````

with:

````python
import pytest

from see.live import CommandAgent, TaskSpec
from see.loader import load_module_from_path
from see.toy import make_task
````

In `reconstruction/tests/test_run_dream_rsi.py`, replace:

````python

def test_a_restart_appends_its_launch_and_keeps_the_first(tmp_path):
````

with:

````python

def test_a_refused_restart_records_no_launch(tmp_path, monkeypatch):
    """report_run.py takes the caps from the last launch line, so a launch the loop refuses must
    leave none: the restart guard runs before the launch is recorded."""
    workdir = tmp_path / "w"
    (workdir / "runs" / "iter0001").mkdir(parents=True)  # an interrupted first iteration
    monkeypatch.setattr(runner, "install_signal_handlers", lambda: None)  # keep pytest's own
    monkeypatch.setattr(runner, "simpletes_task", lambda *a, **kw: make_task(str(tmp_path)))
    argv = [str(workdir) if a == "unused" else a for a in ARGS]
    with pytest.raises(RuntimeError, match="iteration 1 was interrupted or already ran"):
        runner.main(argv)
    assert not (workdir / "launches.jsonl").exists()


def test_a_restart_appends_its_launch_and_keeps_the_first(tmp_path):
````

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest -q tests/test_run_dream_rsi.py`
Expected: `test_a_refused_restart_records_no_launch` fails, because `launches.jsonl` exists.

- [ ] **Step 3: Implement**

The first `loop.py` block turns the head of `online()` into `check_iteration`, whose body is the existing guard. The second closes the guard and reopens `online()` right after it, calling `check_iteration` first:

In `reconstruction/see/loop.py`, replace:

````python
    def online(self, t: int):
        """Stage 1: the deployed policy drives discovery; the tree is frozen into the pool."""
````

with:

````python
    def check_iteration(self, t: int):
        """Refuse iteration t before anything runs: its directories exist, or the deployed policy
        changed since it was deployed. run_dream_rsi.py calls it before recording a launch, so a
        refused restart leaves no launch line."""
````

In `reconstruction/see/loop.py`, replace:

````python
                )
        policy = load_policy(self.state["deployed"])(None)  # baked-in default beta
````

with:

````python
                )

    def online(self, t: int):
        """Stage 1: the deployed policy drives discovery; the tree is frozen into the pool."""
        self.check_iteration(t)
        run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")
        out = os.path.join(self.pool, f"iter{t:04d}")
        policy = load_policy(self.state["deployed"])(None)  # baked-in default beta
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
    record_launch(
        a.workdir,
        launch_record(a, sys.argv if argv is None else list(argv), task, discovery, policy),
    )
    loop = DreamRSI(cfg, task, discovery, policy)
````

with:

````python
    loop = DreamRSI(cfg, task, discovery, policy)
    loop.check_iteration(loop.state["iteration"] + 1)  # a refused restart records no launch
    record_launch(
        a.workdir,
        launch_record(a, sys.argv if argv is None else list(argv), task, discovery, policy),
    )
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_run_dream_rsi.py tests/test_loop.py`
Expected: all pass. The loop's restart tests pin the refusals' messages, which are unchanged.

- [ ] **Step 5: Ledger**

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
starting with `r{round+1}_tNN_m0` (see the archive-name guard row) | fixed in `see/loop.py` (`online`) |
````

with:

````markdown
starting with `r{round+1}_tNN_m0` (see the archive-name guard row); `run_dream_rsi.py` runs the same check before it appends to `launches.jsonl`, so a refused restart records no launch and the report's caps still come from the launch that ran (pinned in `test_run_dream_rsi.py::test_a_refused_restart_records_no_launch`) | fixed in `see/loop.py` (`check_iteration`, which `online` calls first) |
````

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
from ordering | fixed in `see/loop.py` (`_archive`, `_deploy`, `online`) |
````

with:

````markdown
from ordering | fixed in `see/loop.py` (`_archive`, `_deploy`, `check_iteration`) |
````

- [ ] **Step 6: Run the gate**

Run the Global Constraints gate with `--expect 135`. Expected: `135 passed`, `junit: 135 tests, no skips`, ruff and pyright clean.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 4 files above. Then:

```bash
cat > /tmp/d2b-t4-msg.txt <<'EOF'
Check the restart guard before recording a launch

online()'s two refusals, leftover directories of the iteration and a
deployed policy that changed, move into DreamRSI.check_iteration, which
online() still calls first. run_dream_rsi.py calls it before appending
to launches.jsonl, so a refused restart records no launch and
report_run.py's caps keep coming from the launch that ran.
EOF
git add reconstruction/see/loop.py reconstruction/scripts/run_dream_rsi.py \
  reconstruction/tests/test_run_dream_rsi.py reconstruction/GAPS.md
git commit -F /tmp/d2b-t4-msg.txt
```

---

### Task 5: Redaction at copy time, and in the report itself

**Files:**
- Modify: `reconstruction/scripts/report_run.py`:
  - imports `re`, `tempfile`, `typing.Any`;
  - `SOURCE_LINE`, `ADDRESS`, `REDACTIONS`;
  - new `redact`, `redact_json`, `_redacted_copy`;
  - `copy_evidence`;
  - the "Evidence" section of `markdown`;
  - `main` redacts the report before writing it.
- Modify: `reconstruction/tests/test_report_run.py`:
  - import `tempfile`;
  - the symlink and evidence tests compare the fields they are about;
  - three new tests.
- Modify: `.claude/CLAUDE.md` (the `evidence/` convention)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `redact(text: str, counts: collections.Counter) -> str`. It replaces four things, in this order, and adds each rule's count to `counts`:
    - `SOURCE_LINE` (`^ *\d+ \| .*$`, multiline) becomes `[source line withheld]`;
    - `tempfile.gettempdir() + os.sep` becomes `$TMPDIR/`;
    - the home directory + `os.sep` becomes `~/`;
    - `ADDRESS` becomes `[address withheld]`.
  - `redact_json(value: Any, counts) -> Any`, applied to keys and values.
  - `copy_evidence(workdir, dest) -> {"copied", "withheld", "redacted": {rule: {"matches", "files"}}}`, with rules `source_lines`, `temp_paths`, `home_paths` and `addresses`. A file no rule matches is copied byte for byte.
  - `main` writes the report after `redact_json`.

Why: D2a's evidence commit had to do all of this by hand. It withheld a compiler-quoted source line in five files (inside JSON strings too) and rewrote the home prefix; it left a `/var/folders/…` temp path in seven files.

- [ ] **Step 1: Write the failing tests**

In `reconstruction/tests/test_report_run.py`, replace:

````python
import os
import shutil

import pytest
````

with:

````python
import os
import shutil
import tempfile

import pytest
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
    assert result == {
        "copied": 0,
        "withheld": ["runs/iter0001/tree/attempt_b000_a000/proposal.md"],
    }
````

with:

````python
    assert (result["copied"], result["withheld"]) == (
        0,
        ["runs/iter0001/tree/attempt_b000_a000/proposal.md"],
    )
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
    assert report["evidence"] == {
        "copied": 44,
        "withheld": ["runs/iter0002/tree/attempt_b000_a001/proposal.md"],
    }
````

with:

````python
    evidence = report["evidence"]
    assert (evidence["copied"], evidence["withheld"]) == (
        44,
        ["runs/iter0002/tree/attempt_b000_a001/proposal.md"],
    )
    # the toy run quotes no source and no address; its paths lie under the temp directory
    assert (
        evidence["redacted"]["source_lines"]
        == evidence["redacted"]["addresses"]
        == {
            "matches": 0,
            "files": 0,
        }
    )


def _attempt_dir(workdir):
    node = workdir / "runs" / "iter0001" / "tree" / "attempt_b000_a000"
    (node / "eval").mkdir(parents=True)
    return node


def test_quoted_source_lines_inside_json_strings_are_withheld_and_the_json_stays_valid(tmp_path):
    """gcc quotes the failing line of the program as "   79 |   code"; inside score.json that line
    sits between escaped newlines. D2a's evidence commit had to withhold it by hand."""
    node = _attempt_dir(tmp_path / "w")
    error = "x.cpp:79:27: error: expected ';'\n   79 |   double y = x\n      |              ^\n"
    (node / "eval" / "score.json").write_text(json.dumps({"error": error, "combined_score": 0.0}))
    (node / "error.txt").write_text(error)
    (node / "proposal.md").write_text("one step up\n")
    result = report_run.copy_evidence(str(tmp_path / "w"), str(tmp_path / "out"))
    out = tmp_path / "out" / "runs" / "iter0001" / "tree" / "attempt_b000_a000"
    withheld = "x.cpp:79:27: error: expected ';'\n[source line withheld]\n      |              ^\n"
    assert json.loads((out / "eval" / "score.json").read_text())["error"] == withheld
    assert (out / "error.txt").read_text() == withheld
    assert (out / "proposal.md").read_bytes() == b"one step up\n"  # untouched: copied as is
    assert result["redacted"]["source_lines"] == {"matches": 2, "files": 2}


def test_home_temp_and_address_are_rewritten_and_counted(tmp_path, monkeypatch):
    home, temp = tmp_path / "home" / "someone", tmp_path / "tmp"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))
    node = _attempt_dir(tmp_path / "w")
    (node / "error.txt").write_text(
        f"cache {temp}/lasso_cache/a.cpp\nlogged in as someone@example.com\nrun from {home}/runs\n"
    )
    result = report_run.copy_evidence(str(tmp_path / "w"), str(tmp_path / "out"))
    out = tmp_path / "out" / "runs" / "iter0001" / "tree" / "attempt_b000_a000" / "error.txt"
    assert out.read_text() == (
        "cache $TMPDIR/lasso_cache/a.cpp\nlogged in as [address withheld]\nrun from ~/runs\n"
    )
    assert result["redacted"] == {
        "source_lines": {"matches": 0, "files": 0},
        "temp_paths": {"matches": 1, "files": 1},
        "home_paths": {"matches": 1, "files": 1},
        "addresses": {"matches": 1, "files": 1},
    }


def test_the_published_report_is_redacted_like_the_evidence(toy_run):
    """report.json and report.md are published beside the evidence and quote stderr tails."""
    report, out = toy_run
    temp = tempfile.gettempdir() + os.sep
    assert temp not in (out / "report.json").read_text()
    assert temp not in (out / "report.md").read_text()
    assert "$TMPDIR" + os.sep in report["workdir"]
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_report_run.py`

Expected: 4 failed.
- `test_the_evidence_subset_carries_no_program_and_withholds_a_quoted_one` (`KeyError: 'redacted'`);
- `test_quoted_source_lines_inside_json_strings_are_withheld_and_the_json_stays_valid`;
- `test_home_temp_and_address_are_rewritten_and_counted`;
- `test_the_published_report_is_redacted_like_the_evidence`.

- [ ] **Step 3: Implement**

In `reconstruction/scripts/report_run.py`, replace:

````python
import json
import os
import shutil

````

with:

````python
import json
import os
import re
import shutil
import tempfile
from typing import Any

````

In `reconstruction/scripts/report_run.py`, replace:

````python
    "CPP_CODE"  # opens every SimpleTES Lasso program (AGPL); a file quoting it stays out
)

````

with:

````python
    "CPP_CODE"  # opens every SimpleTES Lasso program (AGPL); a file quoting it stays out
)
# Rewritten in every copied file and in the report itself (D2b spec section 6.2): what D2a's
# evidence commit had to redact by hand.
SOURCE_LINE = re.compile(r"^ *\d+ \| .*$", re.M)  # a compiler quoting source: "   79 |   y = x;"
ADDRESS = re.compile(r"[\w.%+-]+@[\w-]+(?:\.[\w-]+)+")
REDACTIONS = {
    "source_lines": "source lines withheld",
    "temp_paths": "temp paths shortened to $TMPDIR",
    "home_paths": "home paths shortened to ~",
    "addresses": "addresses withheld",
}

````

In `reconstruction/scripts/report_run.py`, replace:

````python
def copy_evidence(workdir: str, dest: str) -> dict:
    """Copy the EVIDENCE subset under ``dest``, withholding any file that quotes a program."""
    copied, withheld = 0, []
````

with:

````python
def redact(text: str, counts: collections.Counter) -> str:
    """``text`` without quoted program source, this host's temp and home paths, or email
    addresses; each rule's replacements are added to ``counts``."""
    text, n = SOURCE_LINE.subn("[source line withheld]", text)
    counts["source_lines"] += n
    for rule, prefix, short in (  # the temp prefix first: it may lie under the home directory
        ("temp_paths", tempfile.gettempdir() + os.sep, "$TMPDIR" + os.sep),
        ("home_paths", os.path.expanduser("~") + os.sep, "~" + os.sep),
    ):
        counts[rule] += text.count(prefix)
        text = text.replace(prefix, short)
    text, n = ADDRESS.subn("[address withheld]", text)
    counts["addresses"] += n
    return text


def redact_json(value: Any, counts: collections.Counter) -> Any:
    """``redact`` applied to every string in a parsed JSON value, keys included."""
    if isinstance(value, str):
        return redact(value, counts)
    if isinstance(value, list):
        return [redact_json(v, counts) for v in value]
    if isinstance(value, dict):
        return {redact(k, counts): redact_json(v, counts) for k, v in value.items()}
    return value


def _redacted_copy(path: str, text: str, counts: collections.Counter) -> str:
    """JSON is redacted value by value, so a quoted line between escaped newlines is caught and
    the copy stays valid JSON."""
    if path.endswith(".jsonl"):
        lines = [json.loads(line) for line in text.splitlines() if line.strip()]
        return "".join(json.dumps(redact_json(v, counts)) + "\n" for v in lines)
    if path.endswith(".json"):
        return json.dumps(redact_json(json.loads(text), counts), indent=1) + "\n"
    return redact(text, counts)


def copy_evidence(workdir: str, dest: str) -> dict:
    """Copy the EVIDENCE subset under ``dest``, withholding any file that quotes a program and
    redacting the rest; a file no rule touches is copied byte for byte."""
    copied, withheld = 0, []
    matches, files = collections.Counter(), collections.Counter()
````

In `reconstruction/scripts/report_run.py`, replace:

````python
                if PROGRAM_MARKER in f.read():
                    withheld.append(rel)
                    continue
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
    return {"copied": copied, "withheld": withheld}
````

with:

````python
                text = f.read()
            if PROGRAM_MARKER in text:
                withheld.append(rel)
                continue
            counts = collections.Counter()
            published = _redacted_copy(path, text, counts)
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if sum(counts.values()):
                with open(target, "w") as f:
                    f.write(published)
            else:
                shutil.copyfile(path, target)
            matches.update(counts)
            files.update(rule for rule, n in counts.items() if n)
            copied += 1
    redacted = {rule: {"matches": matches[rule], "files": files[rule]} for rule in REDACTIONS}
    return {"copied": copied, "withheld": withheld, "redacted": redacted}
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        out.append(f"- {role} agent: `{json.dumps(a.get('argv'))}`, version {a.get('version')}")
````

with:

````python
        out.append(f"- {role} agent: `{json.dumps(a.get('argv'))}`, version {a.get('version')}")
    if r.get("evidence"):
        ev = r["evidence"]
        done = [
            f"{n['matches']} {REDACTIONS[rule]} in {n['files']} file(s)"
            for rule, n in ev["redacted"].items()
            if n["matches"]
        ]
        out += [
            "",
            "## Evidence",
            "",
            f"Copied {ev['copied']} file(s); withheld {len(ev['withheld'])} that quote a program "
            "or lead outside the workdir.",
            f"Redacted at copy time: {'; '.join(done) or 'nothing'}.",
        ]
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        )
    with open(os.path.join(a.out, "report.json"), "w") as f:
````

with:

````python
        )
    report = redact_json(report, collections.Counter())  # published beside the evidence
    with open(os.path.join(a.out, "report.json"), "w") as f:
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_report_run.py`
Expected: all pass.

- [ ] **Step 5: Docs**

In `.claude/CLAUDE.md`, replace:

````markdown
  (SimpleTES is AGPL): `report_run.py --copy-evidence` copies an allowlist and withholds any
  file quoting `CPP_CODE`. It is excluded from ruff and the whitespace hooks, kept as written.
````

with:

````markdown
  (SimpleTES is AGPL): `report_run.py --copy-evidence` copies an allowlist, withholds any
  file quoting `CPP_CODE`, and in the rest (and in the report itself) replaces compiler-quoted
  source lines, this host's home and temp paths, and email addresses. It is excluded from ruff
  and the whitespace hooks, kept as written.
````

- [ ] **Step 6: Run the gate**

Run the Global Constraints gate with `--expect 138`. Expected: `138 passed`, `junit: 138 tests, no skips`, ruff and pyright clean.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 3 files above. Then:

```bash
cat > /tmp/d2b-t5-msg.txt <<'EOF'
Redact the evidence copy and the report as they are written

report_run.py --copy-evidence now replaces compiler-quoted source
lines, this host's home and temp paths, and email addresses in every
file it copies, value by value in JSON so a quoted line between
escaped newlines is caught and the copy stays valid; a file no rule
touches is copied byte for byte. The report itself goes through the
same rules and states how many of each it replaced. D2a's evidence
commit had to do this by hand.
EOF
git add reconstruction/scripts/report_run.py reconstruction/tests/test_report_run.py .claude/CLAUDE.md
git commit -F /tmp/d2b-t5-msg.txt
```

---

### Task 6: The report reads the live-plan signal, the loop's untouched flag and the repeats

**Files:**
- Modify: `reconstruction/scripts/report_run.py`:
  - import `statistics`;
  - `analyse_iteration`;
  - `analyse_version`;
  - `build_report`;
  - new `_measured`;
  - `markdown` sections 2, 3 and Health.
- Modify: `reconstruction/tests/test_report_run.py`:
  - imports;
  - `_launch(workdir, eval_repeats=1)`;
  - the untouched and clipped tests;
  - four new tests.

**Interfaces:**
- Consumes:
  - Task 1's `untouched` and `agent_stderr` in `score.json`;
  - Task 2's `repeat_scores` and the launch record's `eval_repeats`;
  - Task 3's `beyond_support` and `live_plan` in the executions file.
- Produces, in `build_report`'s dict:
  - `"eval_repeats"` (None when the launch predates it);
  - on each iteration row, `"loop_untouched"` (a count, or None) and `"repeat_spread"` (the median over attempts of (max − min) / median of `repeat_scores`, or None);
  - on each version, `"beyond_support"` (a count, or None when the sweep predates it) and `"live_asked"`;
  - in `"out_of_support"`: `"beyond_support"`, `"beyond_support_deployed"` and `"measured"`;
  - in `"untouched"`, `"loop_flagged"`;
  - on each untouched case:
    - `"byte_identical"` (bool, or None when the program is gone);
    - `"loop_untouched"`;
    - `"agent_stderr"` (the last line).

  A case is listed when the bytes match or the loop flagged it, so a disagreement shows.

Why: the next run's report must answer what D2a's could not, and a field the run's code did not record must read "not measured", never 0 or False.

- [ ] **Step 1: Write the failing tests**

In `reconstruction/tests/test_report_run.py`, replace:

````python
"""

import glob
import json
import os
````

with:

````python
"""

import dataclasses
import glob
import itertools
import json
import os
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
def _launch(workdir) -> None:
    """The line scripts/run_dream_rsi.py appends to launches.jsonl, cut to what the report reads."""
    line = {
        "task": {"name": "toy", "eval_program": PROGRAM},
        "config": {"fallback_grid": [2, 1], "hard_max_grid": [3, 2]},
````

with:

````python
def _launch(workdir, eval_repeats=1) -> None:
    """The line scripts/run_dream_rsi.py appends to launches.jsonl, cut to what the report reads."""
    line = {
        "task": {"name": "toy", "eval_program": PROGRAM},
        "config": {"fallback_grid": [2, 1], "hard_max_grid": [3, 2], "eval_repeats": eval_repeats},
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
    assert report["out_of_support"] == {
        "flagged": ["r0002_t01_m1", "r0005_t02_m1"],
        "flagged_deployed": [],  # clipped to the tree, WIDE ties m0, and ties keep the earlier
    }
````

with:

````python
    oos = report["out_of_support"]
    assert (oos["flagged"], oos["flagged_deployed"]) == (
        ["r0002_t01_m1", "r0005_t02_m1"],
        [],  # clipped to the tree, WIDE ties m0, and ties keep the earlier
    )
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
        "cases": [
            {"iteration": 1, "cell": "b0a0", "source": "baseline", "fail_class": "no_program"}
            | {"agent_timed_out": False, "agent_returncode": 0}
            | {"evaluated": False, "score": 0.0, "source_score": 1.0},
            {"iteration": 1, "cell": "b1a1", "source": "parent", "fail_class": "no_program"}
            | {"agent_timed_out": True, "agent_returncode": None}
            | {"evaluated": False, "score": 0.0, "source_score": 1.1},
        ],
    }
````

with:

````python
        "loop_flagged": 2,
        "cases": [
            {"iteration": 1, "cell": "b0a0", "source": "baseline", "fail_class": "no_program"}
            | {"agent_timed_out": False, "agent_returncode": 0}
            | {"evaluated": False, "score": 0.0, "source_score": 1.0}
            | {"byte_identical": True, "loop_untouched": True, "agent_stderr": ""},
            {"iteration": 1, "cell": "b1a1", "source": "parent", "fail_class": "no_program"}
            | {"agent_timed_out": True, "agent_returncode": None}
            | {"evaluated": False, "score": 0.0, "source_score": 1.1}
            | {"byte_identical": True, "loop_untouched": True}
            | {"agent_stderr": "agent timed out after 900s"},
        ],
    }


def test_versions_that_would_plan_beyond_the_tree_live_are_counted_per_version(toy_run):
    """WIDE asks for 3 x 1 live too, within the 3 x 2 caps and beyond every 2 x 1 tree, so each
    of its episodes is beyond support; m0 plans the fallback and EMPTY leaves it to the fallback."""
    report, _ = toy_run
    rows = [(v["version"], v["beyond_support"], v["live_asked"]) for v in report["versions"]]
    assert rows == [
        ("r0001_t01_m0", 0, []),
        ("r0002_t01_m1", 12, [[3, 1]]),
        ("r0003_t01_m2", 0, []),
        ("r0004_t02_m0", 0, []),
        ("r0005_t02_m1", 24, [[3, 1]]),
        ("r0006_t02_m2", 0, []),
    ]
    oos = report["out_of_support"]
    assert (oos["beyond_support"], oos["beyond_support_deployed"], oos["measured"]) == (
        ["r0002_t01_m1", "r0005_t02_m1"],
        [],
        True,
    )
````

In `reconstruction/tests/test_report_run.py`, replace:

````python
    assert case["agent_returncode"] == 1
````

with:

````python
    assert case["agent_returncode"] == 1


def test_the_loops_untouched_flag_and_the_byte_check_are_both_shown_and_disagreement_flagged(
    tmp_path, stub_prompts
):
    """The loop decides when the agent returns; the report compares bytes after the run. Agents
    run in the whole tree, so one that later rewrites the source splits the two: show both."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), _NonZeroExit(), ScriptedPolicyAgent()).run()
    source = workdir / "runs" / "iter0001" / "tree" / "attempt_b000_a000" / PROGRAM
    source.write_text(json.dumps({"x": 9.0}))  # b0a1's source, rewritten after b0a1 returned
    report = report_run.build_report(str(workdir))
    (case,) = report["untouched"]["cases"]
    assert (case["cell"], case["loop_untouched"], case["byte_identical"]) == ("b0a1", True, False)
    md = report_run.markdown(report)
    assert "The loop marked 1 untouched when their agent returned; 1 disagree" in md
    assert "| True (disagrees) |" in md


def test_a_workdir_from_before_d2b_reads_not_measured():
    """D2a's committed evidence predates the live-plan signal, the loop's untouched flag and the
    repeat setting: the report says so, rather than reading 0 or False."""
    report = report_run.build_report(os.path.join(RECON, "evidence", "d2a-lasso", "workdir"))
    assert [v["beyond_support"] for v in report["versions"]] == [None, None, None]
    assert (report["eval_repeats"], report["untouched"]["loop_flagged"]) == (None, None)
    md = report_run.markdown(report)
    assert "Live plans beyond the recorded tree: not measured." in md
    assert "The loop's own untouched flag: not measured." in md
    assert "Evaluation repeats in force: not measured." in md


def test_the_health_table_reports_the_repeats_in_force_and_their_median_spread(
    tmp_path, stub_prompts
):
    """With k = 3 each program is evaluated three times; an attempt's spread is (max - min) /
    median of its runs, and each iteration reports the median over its attempts. Every
    evaluation here is scaled by 1.0, 1.1 and 0.9 in turn, so every spread is 0.2."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir, eval_repeats=3)
    task = make_task(str(workdir))
    factors = itertools.cycle([1.0, 1.1, 0.9])

    def jittered(path):
        result = task.evaluate(path)
        return {**result, "combined_score": result["combined_score"] * next(factors)}

    cfg = _config(workdir, iterations=1, versions=1, eval_repeats=3)
    noisy = dataclasses.replace(task, evaluate=jittered)
    DreamRSI(cfg, noisy, ScriptedDiscoveryAgent(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(workdir))
    (row,) = report["iterations"]
    assert (report["eval_repeats"], row["repeat_spread"]) == (3, pytest.approx(0.2))
    assert "Evaluation repeats in force: 3." in report_run.markdown(report)
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_report_run.py`

Expected: 5 failed.
- `test_an_untouched_attempt_is_reported_with_its_source_and_both_scores`;
- `test_versions_that_would_plan_beyond_the_tree_live_are_counted_per_version`;
- `test_the_loops_untouched_flag_and_the_byte_check_are_both_shown_and_disagreement_flagged`;
- `test_a_workdir_from_before_d2b_reads_not_measured`;
- `test_the_health_table_reports_the_repeats_in_force_and_their_median_spread`.

- [ ] **Step 3: Implement**

In `reconstruction/scripts/report_run.py`, replace:

````python
import re
import shutil
import tempfile
````

with:

````python
import re
import shutil
import statistics
import tempfile
````

In `reconstruction/scripts/report_run.py`, replace:

````python
    timeouts = crashed = failures = unchecked = 0
    untouched = []
````

with:

````python
    timeouts = crashed = failures = unchecked = flagged = 0
    flag_recorded, spreads, untouched = False, [], []
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        mine = os.path.join(node, program)
        source, source_attempt = resume_source(tree, baseline_dir, program, c.branch, c.attempt)
        if not os.path.exists(mine):
````

with:

````python
        loop_says = score.get("untouched")  # the loop's own check; absent before D2b
        flag_recorded = flag_recorded or "untouched" in score
        flagged += loop_says is True
        runs = score.get("repeat_scores") or []
        if len(runs) > 1 and statistics.median(runs) > 0:
            spreads.append((max(runs) - min(runs)) / statistics.median(runs))
        mine = os.path.join(node, program)
        source, source_attempt = resume_source(tree, baseline_dir, program, c.branch, c.attempt)
        if os.path.exists(mine):
            identical = _bytes(mine) == _bytes(source)
        else:
````

In `reconstruction/scripts/report_run.py`, replace:

````python
            continue
        if _bytes(mine) != _bytes(source):
````

with:

````python
            identical = None
        if not identical and not loop_says:
````

In `reconstruction/scripts/report_run.py`, replace:

````python
                "score": c.score,
                "source_score": source_score,
            }
````

with:

````python
                "score": c.score,
                "source_score": source_score,
                "byte_identical": identical,
                "loop_untouched": loop_says,
                "agent_stderr": _last_line(score.get("agent_stderr") or ""),
            }
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        "untouched_unchecked": unchecked,
````

with:

````python
        "untouched_unchecked": unchecked,
        "loop_untouched": flagged if flag_recorded else None,
        "repeat_spread": statistics.median(spreads) if spreads else None,
````

In `reconstruction/scripts/report_run.py`, replace:

````python
    clipped = [e for e in episodes if e["out_of_support"]]
````

with:

````python
    clipped = [e for e in episodes if e["out_of_support"]]
    measured = bool(episodes) and all("beyond_support" in e for e in episodes)  # D2b onwards
    beyond = [e for e in episodes if e.get("beyond_support")]
    live_asked = {(e["live_plan"]["branch_count"], e["live_plan"]["refine_count"]) for e in beyond}
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        "recorded": [list(g) for g in sorted(recorded)],
````

with:

````python
        "recorded": [list(g) for g in sorted(recorded)],
        "beyond_support": len(beyond) if measured else None,
        "live_asked": [list(g) for g in sorted(live_asked)],
````

In `reconstruction/scripts/report_run.py`, replace:

````python
    flagged = [v for v in versions if v["clipped"]]
````

with:

````python
    flagged = [v for v in versions if v["clipped"]]
    beyond = [v for v in versions if v["beyond_support"]]
    flags = [r["loop_untouched"] for r in rows]
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        },
        "archive": {"file": os.path.basename(archive), "sha256": sha256_of(archive)}
````

with:

````python
        },
        "eval_repeats": launch["config"].get("eval_repeats"),  # absent before D2b
        "archive": {"file": os.path.basename(archive), "sha256": sha256_of(archive)}
````

In `reconstruction/scripts/report_run.py`, replace:

````python
            "flagged_deployed": [v["version"] for v in flagged if v["deployed"]],
````

with:

````python
            "flagged_deployed": [v["version"] for v in flagged if v["deployed"]],
            "beyond_support": [v["version"] for v in beyond],
            "beyond_support_deployed": [v["version"] for v in beyond if v["deployed"]],
            "measured": any(v["beyond_support"] is not None for v in versions),
````

In `reconstruction/scripts/report_run.py`, replace:

````python
            "unchecked": sum(r["untouched_unchecked"] for r in rows),
````

with:

````python
            "unchecked": sum(r["untouched_unchecked"] for r in rows),
            "loop_flagged": sum(flags) if flags and None not in flags else None,
````

In `reconstruction/scripts/report_run.py`, replace:

````python
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, float) else str(x)
````

with:

````python
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, float) else str(x)


def _measured(x) -> str:
    """A field the run's code did not yet record reads as such, never as 0 or False."""
    return "not measured" if x is None else str(x)
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        f"{len(oos['flagged_deployed'])} of them deployed.",
        "",
        "| Version | Episodes | Clipped | Asked | Recorded | Deployed |",
        "|---|---|---|---|---|---|",
    ]
    for v in r["versions"]:
        asked = ", ".join(_grid(a) for a in v["asked"]) or "-"
        recorded = ", ".join(_grid(g) for g in v["recorded"]) or "-"
        out.append(
            f"| {v['version']} | {v['episodes']} | {v['clipped']} | {asked} | {recorded} | "
            f"{v['deployed']} |"
````

with:

````python
        f"{len(oos['flagged_deployed'])} of them deployed. "
        + (
            f"{len(oos['beyond_support'])} version(s) would plan beyond the recorded tree live; "
            f"{len(oos['beyond_support_deployed'])} of them deployed."
            if oos["measured"]
            else "Live plans beyond the recorded tree: not measured."
        ),
        "",
        "| Version | Episodes | Clipped | Asked | Recorded | Beyond support live | Live asked "
        "| Deployed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in r["versions"]:
        asked = ", ".join(_grid(a) for a in v["asked"]) or "-"
        recorded = ", ".join(_grid(g) for g in v["recorded"]) or "-"
        live = ", ".join(_grid(g) for g in v["live_asked"]) or "-"
        out.append(
            f"| {v['version']} | {v['episodes']} | {v['clipped']} | {asked} | {recorded} | "
            f"{_measured(v['beyond_support'])} | {live} | {v['deployed']} |"
````

In `reconstruction/scripts/report_run.py`, replace:

````python
    out += [
        "",
        "## 3. Untouched programs",
        "",
        f"{len(u['cases'])} of {checked} attempts left their resume source byte for byte. "
        + coverage,
        "",
        "| Iteration | Cell | Source | Fail class | Agent timed out | Agent returncode | "
        "Evaluated | Score | Source score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in u["cases"]:
        out.append(
            f"| {c['iteration']} | {c['cell']} | {c['source']} | {c['fail_class']} | "
            f"{c['agent_timed_out']} | {_num(c['agent_returncode'])} | {c['evaluated']} | "
            f"{_num(c['score'])} | {_num(c['source_score'])} |"
````

with:

````python
    split = [c for c in u["cases"] if None not in (c["loop_untouched"], c["byte_identical"])]
    split = [c for c in split if c["loop_untouched"] != c["byte_identical"]]
    loop = (
        f"The loop marked {u['loop_flagged']} untouched when their agent returned; "
        f"{len(split)} disagree with the byte check."
        if u["loop_flagged"] is not None
        else "The loop's own untouched flag: not measured."
    )
    identical = sum(1 for c in u["cases"] if c["byte_identical"])
    out += [
        "",
        "## 3. Untouched programs",
        "",
        f"{identical} of {checked} attempts left their resume source byte for byte. "
        + coverage
        + " "
        + loop,
        "",
        "| Iteration | Cell | Source | Fail class | Agent timed out | Agent returncode | "
        "Evaluated | Score | Source score | Byte identical | Loop flag "
        "| Agent stderr (last line) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in u["cases"]:
        flag = _measured(c["loop_untouched"]) + (" (disagrees)" if c in split else "")
        out.append(
            f"| {c['iteration']} | {c['cell']} | {c['source']} | {c['fail_class']} | "
            f"{c['agent_timed_out']} | {_num(c['agent_returncode'])} | {c['evaluated']} | "
            f"{_num(c['score'])} | {_num(c['source_score'])} | {_measured(c['byte_identical'])} "
            f"| {flag} | {_cell(c['agent_stderr']) or '-'} |"
````

In `reconstruction/scripts/report_run.py`, replace:

````python
        "| Iteration | Attempts | Successes | Fail classes | Agent timeouts | Agent failures | "
        "No program | Evaluator crashed | Baseline | Best | Error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
````

with:

````python
        f"Evaluation repeats in force: {_measured(r['eval_repeats'])}.",
        "",
        "| Iteration | Attempts | Successes | Fail classes | Agent timeouts | Agent failures | "
        "No program | Evaluator crashed | Baseline | Best | Repeat spread | Error |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
````

In `reconstruction/scripts/report_run.py`, replace:

````python
            f"{i['evaluator_crashed']} | {_num(i['baseline_score'])} | {_num(i['best_score'])} | "
            f"{_cell(i['error'] or '')} |"
````

with:

````python
            f"{i['evaluator_crashed']} | {_num(i['baseline_score'])} | {_num(i['best_score'])} | "
            f"{_num(i['repeat_spread'])} | {_cell(i['error'] or '')} |"
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_report_run.py`
Expected: all pass.

- [ ] **Step 5: Run the gate**

Run the Global Constraints gate with `--expect 142`. Expected: `142 passed`, `junit: 142 tests, no skips`, ruff and pyright clean.

- [ ] **Step 6: Commit**

Run `git diff --stat`. It should show exactly the 2 files above. Then:

```bash
cat > /tmp/d2b-t6-msg.txt <<'EOF'
Report the live-plan signal, the loop's untouched flag and repeats

report_run.py counts, per version, the episodes whose live plan goes
beyond the recorded tree and the grid it asked for; shows the loop's
own untouched flag beside the byte check and flags a disagreement,
with each case's last stderr line; and gives the repeats in force and
each iteration's median in-attempt spread. On a workdir from before
these fields, such as D2a's evidence, each reads "not measured".
EOF
git add reconstruction/scripts/report_run.py reconstruction/tests/test_report_run.py
git commit -F /tmp/d2b-t6-msg.txt
```

---

### Task 7: The verdicts in the ledger; docs; the CI pin

**Files:**
- Modify: `reconstruction/GAPS.md`:
  - §3 rows "Per-round call budget", "`probe_batch([])`" and "Trace ceiling";
  - §6, the off-policy bullet.
- Modify: `reconstruction/tests/test_ledger.py` (a docstring that still called zero versus clip "a D2 decision")
- Modify: `.claude/CLAUDE.md` ("One `question` API, two backends")
- Modify: `reconstruction/README.md`:
  - status rows for replay and online rollout;
  - the run paragraph (`--eval-repeats`);
  - the report paragraph (`--copy-evidence`).
- Modify: `.github/workflows/ci.yml` (`--expect 123` → `--expect 142`)

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: no code.

Why: the spec's section 3 verdicts. Budget, zero versus clip and the empty batch were not observed in D2a, so they stay as they are, and a complete rerun (D2c) revisits them. Afterwards, no "is D2" remains in GAPS.

- [ ] **Step 1: Ledger**

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
and no budget is enforced, so a plan may be (32, 0) or (1, 19); the demo hardcodes (4, 5) and (8, 8) (pinned in `test_ledger.py::test_the_default_fallback_grid_is_the_papers_110_call_round` and `test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper`) | `LoopConfig.fallback_grid`/`hard_max_grid`; `run_dream_rsi.py --grid/--hard-max`; `see sweep --fallback/--hard-max`; a budget knob is D2 |
````

with:

````markdown
and no budget is enforced, so a plan may be (32, 0) or (1, 19), and the D2a run left it so: no plan pushed on a budget there (the largest, the deployed version's live 4 x 4, was 20 calls under the run's 6 x 4 caps of 30), and the caps already bound a round at W × (R + 1); a complete rerun (D2c) revisits it; the demo hardcodes (4, 5) and (8, 8) (pinned in `test_ledger.py::test_the_default_fallback_grid_is_the_papers_110_call_round` and `test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper`) | `LoopConfig.fallback_grid`/`hard_max_grid`; `run_dream_rsi.py --grid/--hard-max`; `see sweep --fallback/--hard-max` |
````

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
−∞ like any other exception, live it propagates to the policy (pinned in `test_world.py::test_illegal_batches_are_refused`, `test_objective.py::test_crashing_or_illegal_policy_is_scored_minus_infinity`) | fixed in `see/world.py` (`probe_batch`); ending the episode instead is D2 |
````

with:

````markdown
−∞ like any other exception, live it propagates to the policy; the D2a run left it so, as no version and no live batch there left a batch empty, and D2c revisits it (pinned in `test_world.py::test_illegal_batches_are_refused`, `test_objective.py::test_crashing_or_illegal_policy_is_scored_minus_infinity`) | fixed in `see/world.py` (`probe_batch`) |
````

In `reconstruction/GAPS.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
clipped plan, with `out_of_support` reported for information (pinned in `test_ledger.py::test_the_trace_ceiling_counts_only_successful_cells` and `::test_an_out_of_support_plan_is_flagged_and_scored_as_its_clipped_grid`) | fixed in `see/world.py` (`Trace.ceiling`, `ReplayQuestion`); zero-versus-clip is D2 |
````

with:

````markdown
clipped plan, with `out_of_support` reported for information; the D2a run left it so, as none of its episodes was clipped (its policies clamped their plans to the trace fields; see the live-plan signal row), and D2c revisits it (pinned in `test_ledger.py::test_the_trace_ceiling_counts_only_successful_cells` and `::test_an_out_of_support_plan_is_flagged_and_scored_as_its_clipped_grid`) | fixed in `see/world.py` (`Trace.ceiling`, `ReplayQuestion`) |
````

In `reconstruction/GAPS.md`, replace:

````markdown
  favours policies close to the ones that generated the pool. The paper does not
  discuss this.
````

with:

````markdown
  favours policies close to the ones that generated the pool. The paper does not
  discuss this. Listing 2 also lets a policy read the recorded grid (`trace_*`), in replay
  only, so a policy can clamp its plan there and plan wider live, and replay then scores a
  plan the policy would not run; each replay episode records the live plan for that reason
  (§3, "Live-plan signal in replay").
````

In `reconstruction/tests/test_ledger.py`, replace:

````python
    clipped plan; the flag is informational (zero-versus-clip is a D2 decision, GAPS §3)."""
````

with:

````python
    clipped plan; the flag is informational (zero versus clip, kept after the D2a run, GAPS §3)."""
````

- [ ] **Step 2: Docs**

In `.claude/CLAUDE.md`, replace:

````markdown
real agent in `runs/iterNNNN/tree/attempt_*/` workspaces. The policy cannot tell them apart, which
is what makes replay evaluation valid. Replay only reveals what the recorded tree contains — a
````

with:

````markdown
real agent in `runs/iterNNNN/tree/attempt_*/` workspaces. Through the `Question` the policy cannot
tell them apart, which is what makes replay evaluation valid. Its `GridPlanningContext` can: the
`trace_*` fields are set only in replay, so each replay episode also records the plan made with
them cleared (GAPS.md §3, "Live-plan signal in replay"). Replay only reveals what the recorded
tree contains — a
````

In `reconstruction/README.md`, replace:

````markdown
| Replay simulator and policy API (`see.policy.api`, `see.policy.observation_signal`) | Sec. 3, Listing 2 | implemented and tested; same import paths as the paper's prompt |
````

with:

````markdown
| Replay simulator and policy API (`see.policy.api`, `see.policy.observation_signal`) | Sec. 3, Listing 2 | implemented and tested; same import paths as the paper's prompt; each replay episode also records the plan the policy would run live, with the replay-only `trace_*` fields cleared |
````

In `reconstruction/README.md`, replace (**mid-line**: this span begins partway through a long line):

````markdown
the residues); a timed-out attempt is scored as the program it left but recorded as a `timeout` failure |
````

with:

````markdown
the residues); a timed-out attempt is scored as the program it changed but recorded as a `timeout` failure; an attempt whose agent changed nothing is `no_program` and never evaluated; each attempt keeps its agent's exit code and stderr tail |
````

In `reconstruction/README.md`, replace:

````markdown
budget: about 110 discovery calls per round at the paper's 3.1-Pro setting. On macOS add
`--eigen-include` as above.
````

with:

````markdown
budget: about 110 discovery calls per round at the paper's 3.1-Pro setting. On macOS add
`--eigen-include` as above. `--eval-repeats 3` evaluates each program three times and keeps the
median run (default 1, the paper's single evaluation); in the D2a run, eight evaluations of one
unchanged program spread 14%.
````

In `reconstruction/README.md`, replace:

````markdown
out-of-support replay, untouched programs, empty batches), as `report.md` and `report.json`:
````

with:

````markdown
out-of-support replay, untouched programs, empty batches), as `report.md` and `report.json`;
`--copy-evidence` also writes the license-safe subset of the workdir, redacted:
````

- [ ] **Step 3: CI pin**

In `.github/workflows/ci.yml`, replace:

````yaml
      - run: python tools/check_junit.py report.xml --expect 123
````

with:

````yaml
      - run: python tools/check_junit.py report.xml --expect 142
````

- [ ] **Step 4: Run the gate and the demo**

Run the Global Constraints gate with `--expect 142`. Expected: `142 passed`, `junit: 142 tests, no skips`, ruff and pyright clean. Then:
- `grep -c 'is D2' GAPS.md` prints `0`;
- `python -m see demo --workdir /tmp/drsi-d2b` runs to its three `iter` lines.

- [ ] **Step 5: Commit**

Run `git diff --stat`. It should show exactly the 5 files above. Then:

```bash
cat > /tmp/d2b-t7-msg.txt <<'EOF'
Record D2's verdicts in the ledger and bring the docs up to date

The three ledger rows that deferred to D2 now say what the D2a run
showed and that a complete rerun revisits them: no budget, the clip
kept for out-of-support replay, and the empty batch left at minus
infinity. GAPS section 6 and CLAUDE.md say that a policy can tell
replay from live through the trace fields, and the README describes
the untouched rule, the repeat setting and the redacted evidence copy.
CI pins the suite at 142 tests.
EOF
git add reconstruction/GAPS.md reconstruction/tests/test_ledger.py .claude/CLAUDE.md \
  reconstruction/README.md .github/workflows/ci.yml
git commit -F /tmp/d2b-t7-msg.txt
```

---

After Task 7: the whole-branch review against the spec (sections 1–13) and this plan, then the finishing menu. The PR targets `main`.
