import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

import pytest

import see.live
import see.prompts
from see.live import LiveQuestion, repeated
from see.loader import load_policy
from see.loop import DreamRSI, LoopConfig, archive_name, install_signal_handlers
from see.objective import run_episode
from see.pool import context_factory, load_pool
from see.toy import PROGRAM, ScriptedDiscoveryAgent, ScriptedPolicyAgent, evaluate, make_task
from see.world import Trace


@pytest.fixture(scope="module")
def finished_loop(tmp_path_factory, stub_prompts):
    work = str(tmp_path_factory.mktemp("loop"))
    cfg = LoopConfig(
        workdir=work,
        iterations=3,
        versions=3,
        max_parallelism=4,
        fallback_grid=(4, 5),
        hard_max_grid=(8, 8),
        betas=(0.0, 0.5, 1.0),
    )
    loop = DreamRSI(
        cfg,
        make_task(work),
        ScriptedDiscoveryAgent(seed=7),
        ScriptedPolicyAgent(betas=(0.3, 0.9), broken_every=2),
    )
    loop.run()
    return loop


def test_loop_writes_the_layout_listing_2_refers_to(finished_loop):
    w = finished_loop.w
    for t in (1, 2, 3):
        for name in ("trace.json", "live_cycle_manifest.json", "live_episode.jsonl"):
            assert os.path.exists(os.path.join(w, "trace_pool", f"iter{t:04d}", name))
    assert os.readlink(os.path.join(w, "trace_pool", "_current")) == "iter0003"
    base = os.path.join(w, "policy_dev", "history", "baseline", "proposal_results")
    assert os.path.exists(os.path.join(base, "beta_sweep.json"))
    rounds = sorted(d for d in os.listdir(os.path.join(w, "policy_dev", "history")) if d[0] == "r")
    assert len(rounds) == 9  # 3 iterations x M = 3 versions
    # one replay episode per (trace, beta) plus one per trace at the default beta
    first = os.path.join(
        w, "policy_dev", "history", rounds[0], "proposal_results", "policy_execution_traces.jsonl"
    )
    with open(first) as f:
        assert sum(1 for _ in f) == 1 * (3 + 1)


def test_broken_versions_score_minus_infinity_and_are_never_deployed(finished_loop):
    for entry in finished_loop.state["log"]:
        scores = {c["round"]: c for c in entry["offline"]}
        broken = [c for c in scores.values() if not c["valid"]]
        assert broken, "the scripted agent breaks every second revision"
        assert all(c["score"] == float("-inf") for c in broken)
        chosen = scores[entry["selected"]]
        assert chosen["valid"]
        assert chosen["score"] == max(c["score"] for c in scores.values())
        m0 = next(c for c in scores.values() if c["m"] == 0)
        assert chosen["score"] >= m0["score"]  # Sec. 3: never worse than pi_t on H_t


def test_on_policy_replay_reproduces_the_live_episode(finished_loop):
    """Replaying the deployed policy on its own tree must repeat its live batches."""
    w, c = finished_loop.w, finished_loop.c
    pool = load_pool(os.path.join(w, "trace_pool"))
    context_for = context_factory(pool, c.fallback_grid, c.hard_max_grid)
    for t, (trace, manifest) in enumerate(pool, start=1):
        cls = load_policy(os.path.join(w, "deployed", f"iter{t:04d}.py"))
        replay = run_episode(cls(None), trace, context_for(trace), record=True)
        with open(os.path.join(w, "trace_pool", f"iter{t:04d}", "live_episode.jsonl")) as f:
            live = [json.loads(line) for line in f]
        assert [s["batch"] for s in replay.log] == [s["batch"] for s in live]
        assert replay.probes == manifest["probes"] and replay.best == manifest["best_score"]


def test_plan_contexts_only_see_earlier_cycles(finished_loop):
    pool = load_pool(os.path.join(finished_loop.w, "trace_pool"))
    context_for = context_factory(pool, (4, 5), (8, 8))
    for i, (trace, _) in enumerate(pool):
        assert [m["iteration"] for m in context_for(trace).history] == list(range(1, i + 1))


def test_agent_crash_is_a_failed_attempt_not_a_failed_episode(tmp_path, stub_prompts):
    task = make_task(str(tmp_path))

    def crashes_on_roots(prompt, *, cwd, target):
        if target.endswith("_a000"):
            raise RuntimeError("agent process died")
        path = os.path.join(target, PROGRAM)
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:  # re-saved: a real edit that keeps the resumed x
            json.dump({"x": x}, f, indent=1)
        return {"returncode": 0}

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        task,
        crashes_on_roots,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=1,
    )
    obs = q.probe_batch(q.legal_roots())
    assert [o.fail_class for o in obs] == ["no_program", "no_program"]
    assert not obs[0].evaluated
    assert (tree / "attempt_b000_a000" / "error.txt").exists()
    assert q.legal_actions() == ["b0a1", "b1a1"]  # the episode goes on
    # children of a crashed attempt resume from the nearest program: here the baseline
    obs = q.probe_batch(q.legal_actions())
    assert [o.fail_class for o in obs] == ["ok", "ok"] and obs[0].score == 1.0
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


def test_repeats_score_the_median_run_and_stop_at_the_first_error(tmp_path):
    """D2a's eight evaluations of one unchanged program spread 14%, about the size of the
    improvement. With k repeats a program scores its median run; a run that fails ends the repeats
    and is returned as it is, so a failure is never averaged away."""
    runs = iter(
        [
            {"combined_score": 0.5, "run": 1},
            {"combined_score": 0.3, "run": 2},
            {"combined_score": 0.1, "run": 3},
            {"combined_score": 0.4, "run": 4},
            {"combined_score": 0.2, "run": 5},
            {"combined_score": 0.4, "run": 6},
            {"combined_score": 0.0, "error": "ValueError: boom", "run": 7},
            {"combined_score": 0.9, "run": 8},
        ]
    )
    calls = []

    def evaluate(path):
        calls.append(path)
        return next(runs)

    # the median, 0.3, is not the first, last or middle run submitted, nor sorted(...)[1]
    five = repeated(dataclasses.replace(make_task(str(tmp_path)), evaluate=evaluate), 5)
    assert five.evaluate("p") == {
        "combined_score": 0.3,
        "run": 2,
        "repeat_scores": [0.5, 0.3, 0.1, 0.4, 0.2],
    }
    assert five.evaluate("p") == {
        "combined_score": 0.0,
        "error": "ValueError: boom",
        "run": 7,
        "repeat_scores": [0.4, 0.0],
    }
    assert len(calls) == 7  # the eighth run never happened


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


def test_one_malformed_evaluator_result_fails_one_cell_not_the_batch(tmp_path, stub_prompts):
    """Garbage an evaluator returns (not raises) costs its own cell, not its siblings'."""
    task = make_task(str(tmp_path))
    real_evaluate = task.evaluate
    malformed = {  # each of these used to abort the whole batch it was evaluated in
        "attempt_b001_a000": {"combined_score": "n/a", "validity": 1.0},
        "attempt_b001_a001": {"combined_score": 3.0, "error": 42},
        "attempt_b000_a002": {"combined_score": 3.0, "error": []},  # falsy, but not a string
        "attempt_b002_a002": {"error": None},  # combined_score missing entirely
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
        "attempt_b000_a002": 8.0,
        "attempt_b001_a002": 9.0,
        "attempt_b002_a002": 10.0,
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
        refine_count=2,
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
    obs = q.probe_batch(  # round 3: branch 0's error is falsy, branch 2's combined_score is missing
        q.legal_actions()
    )
    assert [o.cell_id for o in obs] == ["b0a2", "b1a2", "b2a2"]
    assert [(o.score, o.evaluated) for o in obs] == [(0.0, False), (9.0, True), (0.0, False)]
    assert obs[1].fail_class == "ok"  # the well-formed sibling keeps its result
    for i in (0, 2):
        assert obs[i].error is not None
        assert obs[i].error.startswith("ValueError: malformed evaluator result")
    with open(tree / "attempt_b000_a002" / "eval" / "score.json") as f:
        assert json.load(f)["evaluator_crashed"] is True
    with open(tree / "attempt_b002_a002" / "eval" / "score.json") as f:
        assert json.load(f)["evaluator_crashed"] is True
    assert len(q.frozen("iter0001", {})) == 9  # every cell of all three rounds reaches the tree


def test_a_fault_while_recording_a_batch_keeps_none_of_it(tmp_path, stub_prompts, monkeypatch):
    """The cells of a batch reach the tree together or not at all, so neither a fault while
    recording nor an interrupt between two cells leaves half a batch in the tree."""
    real, seen = see.live.observation_for, []

    def fails_on_the_second(cell, parent, baseline):
        seen.append(cell.id)
        if len(seen) == 2:
            raise RuntimeError("fault while recording the batch")
        return real(cell, parent, baseline)

    monkeypatch.setattr(see.live, "observation_for", fails_on_the_second)
    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        lambda prompt, *, cwd, target: {"returncode": 0},  # leaves the parent's program
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=0,
    )
    with pytest.raises(RuntimeError, match=r"fault while recording"):
        q.probe_batch(q.legal_roots())
    assert seen == ["b0a0", "b1a0"] and q.cells == {}


SWALLOWS = '''"""A policy that hides the batch's fault: for the manifest test only."""

from see.policy.api import GridPlan, LLMDesignedMethod, SimResult, finalize_result

NAME = "Swallows"


class Swallows(LLMDesignedMethod):
    NAME = NAME

    def solve(self, question, budget=None):
        question.reset()
        for _ in range(2):  # keeps probing after the first batch is abandoned
            try:
                question.probe_batch(question.legal_roots()[: question.max_parallelism])
            except Exception:
                pass
        return finalize_result(question, SimResult())

    def plan_grid(self, context):
        return GridPlan(context.fallback_branch_count, context.fallback_refine_count, reason="t")
'''


def test_a_fault_the_policy_swallows_still_reaches_the_manifest(
    tmp_path, stub_prompts, monkeypatch
):
    """A policy that catches the batch's exception and returns cannot produce a clean-looking
    truncated cycle: the manifest names the fault, and the cycle is frozen with that error just
    as when the fault propagates. The fault here is a missing prompt file, raised in the worker
    before any agent call; the policy then probes the remaining roots, and the manifest still
    names the first fault, not the refusal of the second batch."""
    policy = tmp_path / "swallows.py"
    policy.write_text(SWALLOWS)
    cfg = LoopConfig(
        workdir=str(tmp_path),
        max_parallelism=2,
        fallback_grid=(4, 1),
        hard_max_grid=(4, 1),
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


def test_a_deployed_policy_edited_since_deploy_is_refused_before_it_runs(tmp_path, stub_prompts):
    """The digest state.json records at deploy time is checked again when the policy is loaded,
    so an edited deployed/iterNNNN.py never drives a rollout."""
    agent = _RecordingAgent()
    loop = _interruptible_loop(str(tmp_path), agent)
    loop.online(1)
    loop.offline(1)
    deployed = loop.state["deployed"]
    assert deployed.endswith("iter0002.py") and loop.state["deployed_sha256"] == _sha256(deployed)
    with open(deployed, "a") as f:
        f.write("\n# edited after it was deployed\n")
    calls_before = len(agent.targets)
    state = (tmp_path / "state.json").read_text()
    with pytest.raises(RuntimeError, match=r"iter0002\.py changed since it was deployed"):
        loop.online(2)
    assert len(agent.targets) == calls_before  # refused before any agent call
    assert not (tmp_path / "runs" / "iter0002").exists()
    assert (tmp_path / "state.json").read_text() == state


def test_a_sweep_that_scored_other_bytes_than_the_archive_is_refused(tmp_path, monkeypatch):
    """beta_sweep.json carries the sha256 the subprocess hashed before loading the method; a
    score for other bytes than the archived ones is an integrity failure, not a score."""
    loop = _interruptible_loop(str(tmp_path), _RecordingAgent())
    method = tmp_path / "method.py"
    method.write_text("# a policy\n")
    invalid = {"valid": False, "errors": ["x"], "pareto": {"reward": -1e999}, "eq1": {"V": -1e999}}
    monkeypatch.setattr(
        loop, "_sweep", lambda archived, rdir: {**invalid, "valid": True, "sha256": "0" * 64}
    )
    with pytest.raises(RuntimeError, match=r"did not score the archived bytes"):
        loop._archive(str(method), 1, 0)
    # a crashed or timed-out sweep writes no digest, so there is nothing to compare
    monkeypatch.setattr(loop, "_sweep", lambda archived, rdir: invalid)
    assert loop._archive(str(method), 1, 1)["score"] == float("-inf")


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


def test_sweep_timeout_kills_the_subprocess_group_and_scores_invalid(tmp_path, process_gone):
    """A replay subprocess that hangs is killed with everything it forked, and the version
    scores invalid exactly as a crashed one does."""
    work = str(tmp_path)
    # 2 s covers interpreter startup, importing see and hashing before the "policy" forks on a
    # loaded runner; the test's own bound below is what keeps a regression visible
    cfg = LoopConfig(workdir=work, sweep_timeout=2.0, kill_grace=0.2)
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
    assert time.time() - started < 5.0  # the "policy" sleeps 30 s
    assert report["valid"] is False
    assert report["errors"] == ["RuntimeError: sweep timed out after 2.0s"]
    assert report["pareto"]["reward"] == float("-inf")
    assert process_gone(int(pid_file.read_text()))
    with open(rdir / "proposal_results" / "beta_sweep.json") as f:
        assert json.load(f)["valid"] is False


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


def test_restart_refuses_when_any_archive_of_the_iteration_exists(tmp_path, stub_prompts):
    """An iteration runs once per workdir, so any policy_dev/history/r*_tNN_m* entry for it is a
    leftover of an aborted or completed run, not only the r{round+1}_tNN_m0 a restart would
    recreate: every one of them is named, whatever round or version number it carries."""
    agent = _RecordingAgent()
    loop = _interruptible_loop(str(tmp_path), agent)
    history = tmp_path / "policy_dev" / "history"
    for name in (archive_name(7, 1, 2), archive_name(1, 2, 0), "baseline"):  # only t01 matters
        (history / name).mkdir(parents=True)
    with pytest.raises(RuntimeError, match=r"history/r0007_t01_m2 exists: .*delete it"):
        loop.online(1)
    assert agent.targets == []
    (history / archive_name(8, 1, 0)).mkdir()
    with pytest.raises(RuntimeError, match=r"r0007_t01_m2 and .*r0008_t01_m0 exist: .*delete them"):
        loop.online(1)
    assert agent.targets == []


def test_restart_guard_and_manifests_survive_glob_characters_in_the_workdir(tmp_path, stub_prompts):
    """A workdir such as run[1] must neither make the guard's glob match nothing, which would let
    a restart overwrite the aborted archive, nor hide the pool's manifests from planning."""
    work = tmp_path / "run[1]"
    agent = _RecordingAgent()
    loop = _interruptible_loop(str(work), agent)
    (work / "policy_dev" / "history" / archive_name(1, 1, 0)).mkdir(parents=True)
    pool_entry = work / "trace_pool" / "iter0001"
    pool_entry.mkdir()
    (pool_entry / "live_cycle_manifest.json").write_text('{"iteration": 1}\n')
    assert loop.manifests() == [{"iteration": 1}]
    with pytest.raises(RuntimeError, match=r"trace_pool/iter0001 and .*r0001_t01_m0 exist"):
        loop.online(1)
    assert agent.targets == []


def test_sigterm_takes_the_same_path_as_ctrl_c():
    """`kill <pid>` raises KeyboardInterrupt in the main thread, so a run freezes its partial
    tree and refuses on restart like Ctrl-C does, instead of exiting at once."""
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        install_signal_handlers()
        with pytest.raises(KeyboardInterrupt, match=r"signal 15"):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(0.5)  # the handler runs at the next bytecode boundary
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def test_sighup_takes_the_same_path_as_ctrl_c_unless_inherited_ignored():
    """A hangup (the terminal or SSH session going away) freezes and refuses like Ctrl-C, since
    the agents and the sweep run in their own sessions and never see the hangup themselves; a run
    launched under nohup, which ignores SIGHUP, keeps ignoring it."""
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        install_signal_handlers()
        with pytest.raises(KeyboardInterrupt, match=r"signal 1$"):
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(0.5)  # the handler runs at the next bytecode boundary
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        install_signal_handlers()
        assert signal.getsignal(signal.SIGHUP) == signal.SIG_IGN
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def test_the_demo_entry_point_maps_sigterm_to_the_partial_freeze(tmp_path, stub_prompts):
    """`kill <pid>` of a real `python -m see demo` run ends it the way Ctrl-C does: the process
    dies by KeyboardInterrupt (re-raised as SIGINT, not killed by SIGTERM), the partial tree is
    frozen under runs/, nothing reaches the pool and state.json is untouched. The entry point runs
    in a subprocess with the stub prompts and the scripted agent slowed to 0.2 s per attempt, so
    the signal lands while the first batch is in flight."""
    work = tmp_path / "work"
    program = (
        "import sys, time\n"
        "import see.prompts, see.toy\n"
        f"see.prompts.GENERATED = {see.prompts.GENERATED!r}\n"
        "_call = see.toy.ScriptedDiscoveryAgent.__call__\n"
        "see.toy.ScriptedDiscoveryAgent.__call__ = "
        "lambda self, *a, **k: (time.sleep(0.2), _call(self, *a, **k))[1]\n"
        "from see.__main__ import main\n"
        "main(['demo', '--workdir', sys.argv[1], '--iterations', '1'])\n"
    )
    p = subprocess.Popen(
        [sys.executable, "-c", program, str(work)],
        cwd=str(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    tree = work / "runs" / "iter0001" / "tree"
    deadline = time.time() + 20.0
    while not any(tree.glob("attempt_*")) and time.time() < deadline:
        time.sleep(0.05)  # until the first attempt is in flight
    assert any(tree.glob("attempt_*")), p.communicate(timeout=20.0)[1][-2000:]
    state = (work / "state.json").read_text()
    p.send_signal(signal.SIGTERM)
    _, err = p.communicate(timeout=20.0)
    assert p.returncode == -signal.SIGINT, err[-2000:]
    assert "KeyboardInterrupt: signal 15" in err
    partial = work / "runs" / "iter0001" / "partial"
    assert Trace.load(str(partial / "trace.json")).trace_id == "iter0001-partial"
    with open(partial / "live_cycle_manifest.json") as f:
        manifest = json.load(f)
    assert manifest["partial"] is True and manifest["error"] == "KeyboardInterrupt: signal 15"
    assert os.listdir(work / "trace_pool") == []
    assert (work / "state.json").read_text() == state


def test_interrupt_during_the_baseline_evaluation_leaves_nothing_under_runs(tmp_path, stub_prompts):
    """An interrupt before the first attempt (here: during the baseline evaluation on a fresh
    workdir) leaves no runs/iterNNNN at all, so a restart needs no cleanup and is not refused."""
    work = str(tmp_path)
    calls: list = []

    def interrupting_evaluate(path):
        calls.append(path)
        if len(calls) == 1:
            raise _Interrupted("killed during the baseline evaluation")
        return evaluate(path)

    cfg = LoopConfig(workdir=work, max_parallelism=1, fallback_grid=(2, 1), hard_max_grid=(2, 1))
    task = dataclasses.replace(make_task(work), evaluate=interrupting_evaluate)
    agent = _RecordingAgent()
    loop = DreamRSI(cfg, task, agent, ScriptedPolicyAgent())
    state = (tmp_path / "state.json").read_text()
    with pytest.raises(_Interrupted):
        loop.online(1)
    assert os.listdir(tmp_path / "runs") == []
    assert not (tmp_path / "baseline_eval.json").exists()
    assert agent.targets == []
    assert (tmp_path / "state.json").read_text() == state
    assert loop.online(1)["probes"] == 4  # the restart runs the whole 2 x (1 + 1) grid
