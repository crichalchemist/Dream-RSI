"""Pins for scripts/report_run.py, the D2a evidence report (spec section 5).

One scripted run on the toy task stands in for the real-agent run: a 2 x 1 fallback grid (two
branches of two attempts), two iterations, three policy versions per iteration, two workers. The
discovery agent adds 0.1 to the program's x except where SCRIPT says otherwise; the policy agent
alternates a version that plans one branch wider than any recorded tree (WIDE) and one that ends
on an empty batch (EMPTY). Every expected number below is counted by hand from that script.
"""

import dataclasses
import glob
import itertools
import json
import os
import shutil
import tempfile

import pytest

from see.live import LiveQuestion
from see.loader import load_module_from_path
from see.loop import DreamRSI, LoopConfig
from see.toy import PROGRAM, ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

RECON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report_run = load_module_from_path(
    "report_run_under_test", os.path.join(RECON, "scripts", "report_run.py")
)
WIDE = """from see.policies.parallel_refine import ParallelRefine
from see.policy.api import GridPlan

NAME = "WiderRefine"


class WiderRefine(ParallelRefine):
    NAME = NAME

    def plan_grid(self, context):
        return GridPlan(
            context.fallback_branch_count + 1, context.fallback_refine_count, reason="one wider"
        )
"""
EMPTY = """from see.policy.api import LLMDesignedMethod

NAME = "StopsWithAnEmptyBatch"


class StopsWithAnEmptyBatch(LLMDesignedMethod):
    NAME = NAME

    def solve(self, question, budget=None):
        question.reset()
        question.probe_batch([])
"""
SCRIPT = {
    ("iter0001", "attempt_b000_a000"): "untouched",  # identical to the baseline: no_program
    ("iter0001", "attempt_b001_a001"): "timeout",  # identical to its parent, timed out: no_program
    ("iter0002", "attempt_b001_a000"): "delete",  # no program; its child resumes the baseline
    ("iter0002", "attempt_b000_a001"): "paste",  # its proposal quotes a program: withheld
}


class ScriptedDiscovery:
    def __call__(self, prompt, *, cwd, target):
        iteration = os.path.basename(os.path.dirname(os.path.dirname(target)))
        what = SCRIPT.get((iteration, os.path.basename(target)))
        with open(os.path.join(target, "proposal.md"), "w") as f:
            f.write("pasted CPP_CODE = ...\n" if what == "paste" else "one step up\n")
        path = os.path.join(target, PROGRAM)
        if what == "untouched":
            return {"returncode": 0}
        if what == "timeout":
            return {"returncode": None, "timed_out": True, "stderr": "agent timed out after 900s"}
        if what == "delete":
            os.remove(path)
            return {"returncode": 0}
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:
            json.dump({"x": round(x + 0.1, 6)}, f)
        return {"returncode": 0}


class AlternatingPolicy:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, *, cwd, target):
        self.calls += 1
        with open(target, "w") as f:
            f.write(WIDE if self.calls % 2 else EMPTY)
        return {"returncode": 0}


def _launch(workdir, eval_repeats=1) -> None:
    """The line scripts/run_dream_rsi.py appends to launches.jsonl, cut to what the report reads."""
    line = {
        "task": {"name": "toy", "eval_program": PROGRAM},
        "config": {"fallback_grid": [2, 1], "hard_max_grid": [3, 2], "eval_repeats": eval_repeats},
    }
    (workdir / "launches.jsonl").write_text(json.dumps(line) + "\n")


def _config(workdir, **kw) -> LoopConfig:
    return LoopConfig(
        workdir=str(workdir), max_parallelism=2, fallback_grid=(2, 1), hard_max_grid=(3, 2), **kw
    )


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory, stub_prompts):
    root = tmp_path_factory.mktemp("d2a")
    workdir = root / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=2, versions=3)
    DreamRSI(cfg, make_task(str(workdir)), ScriptedDiscovery(), AlternatingPolicy()).run()
    (root / "archive.bin").write_bytes(b"archive")
    out = root / "out"
    args = ["--workdir", str(workdir), "--out", str(out), "--archive", str(root / "archive.bin")]
    return report_run.main([*args, "--copy-evidence"]), out


def test_each_rounds_plan_and_spend_come_from_its_manifest(toy_run):
    report, _ = toy_run
    budget = [
        {k: i[k] for k in ("iteration", "policy_round", "planned_grid", "used_fallback")}
        | {k: i[k] for k in ("effective_grid", "grid_calls", "probes", "decision_rounds")}
        for i in report["iterations"]
    ]
    assert budget == [  # parallel refine plans the fallback, and m0 is deployed both times
        {"iteration": 1, "policy_round": "initial", "planned_grid": [2, 1], "used_fallback": False}
        | {"effective_grid": [2, 1], "grid_calls": 4, "probes": 4, "decision_rounds": 2},
        {"iteration": 2, "policy_round": "r0001_t01_m0", "planned_grid": [2, 1]}
        | {"used_fallback": False, "effective_grid": [2, 1], "grid_calls": 4, "probes": 4}
        | {"decision_rounds": 2},
    ]
    assert (
        report["launches"],
        report["caps"]["fallback_calls"],
        report["caps"]["hard_max_calls"],
    ) == (
        1,
        4,
        9,
    )


def test_versions_that_plan_wider_than_the_tree_are_flagged_with_their_clipped_episodes(toy_run):
    """WIDE asks for 3 x 1 on 2 x 1 trees: every episode is clipped, 11 betas plus the default
    episode per trace, so 12 in iteration 1 and 24 over the two trees of iteration 2."""
    report, _ = toy_run
    rows = [
        (v["version"], v["episodes"], v["clipped"], v["asked"], v["recorded"], v["deployed"])
        for v in report["versions"]
    ]
    assert rows == [
        ("r0001_t01_m0", 12, 0, [], [], True),
        ("r0002_t01_m1", 12, 12, [[3, 1]], [[2, 1]], False),
        ("r0003_t01_m2", 12, 0, [], [], False),
        ("r0004_t02_m0", 24, 0, [], [], True),
        ("r0005_t02_m1", 24, 24, [[3, 1]], [[2, 1]], False),
        ("r0006_t02_m2", 24, 0, [], [], False),
    ]
    oos = report["out_of_support"]
    assert (oos["flagged"], oos["flagged_deployed"]) == (
        ["r0002_t01_m1", "r0005_t02_m1"],
        [],  # clipped to the tree, WIDE ties m0, and ties keep the earlier
    )


def test_an_untouched_attempt_is_reported_with_its_source_and_both_scores(toy_run):
    report, _ = toy_run
    assert report["untouched"] == {
        "attempts": 8,
        "unchecked": 0,
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


def test_versions_that_plan_beyond_a_tree_without_its_trace_fields_are_counted_per_version(toy_run):
    """WIDE asks for 3 x 1 with the trace fields cleared too, within the 3 x 2 caps and beyond
    every 2 x 1 tree, so each of its episodes is beyond support; m0 plans the fallback and EMPTY
    leaves it to the fallback."""
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
    md = report_run.markdown(report)
    assert "2 version(s) planned beyond a replayed tree with the trace fields cleared; 0 of" in md
    assert "| r0001_t01_m0 | 12 | 0 | - | - | 0 | - |" in md  # measured, and nothing asked beyond


def test_versions_whose_next_live_plan_no_recorded_tree_covers_are_counted(toy_run):
    """WIDE plans 3 x 1 for the next live cycle too, and no recorded 2 x 1 tree covers it; m0
    plans the fallback and EMPTY leaves the grid to it (GAPS §3, "Next live plan")."""
    report, _ = toy_run
    rows = [
        (v["version"], v["next_live_plan"], v["next_beyond_support"], v["next_live_plan_error"])
        for v in report["versions"]
    ]
    assert rows == [
        ("r0001_t01_m0", [2, 1], False, None),
        ("r0002_t01_m1", [3, 1], True, None),
        ("r0003_t01_m2", [2, 1], False, None),
        ("r0004_t02_m0", [2, 1], False, None),
        ("r0005_t02_m1", [3, 1], True, None),
        ("r0006_t02_m2", [2, 1], False, None),
    ]
    oos = report["out_of_support"]
    assert (oos["next_beyond_support"], oos["next_beyond_support_deployed"]) == (
        ["r0002_t01_m1", "r0005_t02_m1"],
        [],
    )
    assert (oos["next_live_plan_errors"], oos["next_measured"]) == ([], True)
    md = report_run.markdown(report)
    assert "2 version(s) would plan a next live cycle that no recorded tree covers; 0 of" in md
    assert "| r0001_t01_m0 | 12 | 0 | - | - | 0 | - | 2 x 1 | True |" in md
    assert "| 12 | 3 x 1 | 3 x 1, beyond support | False |" in md


def test_a_next_live_plan_that_raised_is_reported_with_its_error(tmp_path):
    """online() would stop on a deployed version whose plan_grid raises, so the report names the
    error where the grid would be, never "not measured". D2a's evidence, with m2's sweep given
    such an error, stands in for a run that recorded one."""
    workdir = tmp_path / "w"
    shutil.copytree(os.path.join(RECON, "evidence", "d2a-lasso", "workdir"), workdir)
    path = workdir / "policy_dev" / "history" / "r0003_t01_m2" / "proposal_results"
    sweep = json.loads((path / "beta_sweep.json").read_text())
    error = 'Traceback (most recent call last):\n  File "method.py"\nRuntimeError: no plan\n'
    sweep["next_live_plan"] = {"error": error}
    (path / "beta_sweep.json").write_text(json.dumps(sweep))
    report = report_run.build_report(str(workdir))
    m2 = report["versions"][-1]
    assert (m2["next_live_plan"], m2["next_beyond_support"], m2["next_live_plan_error"]) == (
        None,
        None,
        "RuntimeError: no plan",
    )
    assert report["out_of_support"]["next_live_plan_errors"] == ["r0003_t01_m2"]
    md = report_run.markdown(report)
    assert "no recorded tree covers; 0 of them deployed. 1 version(s) raised planning it." in md
    assert "| raised: RuntimeError: no plan | True |" in md


def test_versions_that_end_on_an_empty_batch_are_named_with_their_cause(toy_run):
    report, _ = toy_run
    empty = [
        (v["version"], v["valid"], v["cause"], v["episode_errors"], v["first_error"])
        for v in report["versions"]
        if v["cause"]
    ]
    line = "see.world.IllegalBatch: empty batch: stop by not probing"
    assert empty == [
        ("r0003_t01_m2", False, "empty_batch", 12, line),
        ("r0006_t02_m2", False, "empty_batch", 24, line),
    ]
    assert report["empty_batches"] == {"versions": ["r0003_t01_m2", "r0006_t02_m2"], "live": []}


def test_a_live_batch_left_empty_is_reported_from_its_manifest(tmp_path, stub_prompts):
    policy = tmp_path / "empty.py"
    policy.write_text(EMPTY)
    _launch(tmp_path)
    cfg = _config(tmp_path, iterations=1, versions=1, initial_policy=str(policy))
    DreamRSI(cfg, make_task(str(tmp_path)), ScriptedDiscoveryAgent(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(tmp_path))
    (row,) = report["iterations"]
    assert (row["probes"], row["error"], row["error_cause"]) == (
        0,
        "IllegalBatch: empty batch: stop by not probing",
        "empty_batch",
    )
    assert report["empty_batches"] == {"versions": ["r0001_t01_m0"], "live": [1]}


class _Interrupting:
    """Adds 0.1, and raises KeyboardInterrupt on attempt b1a1, as Ctrl-C would mid-batch."""

    def __call__(self, prompt, *, cwd, target):
        if os.path.basename(target) == "attempt_b001_a001":
            raise KeyboardInterrupt("Ctrl-C")
        path = os.path.join(target, PROGRAM)
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:
            json.dump({"x": round(x + 0.1, 6)}, f)
        return {"returncode": 0}


def test_an_interrupted_iteration_is_reported_as_partial(tmp_path, stub_prompts):
    """The second batch is abandoned whole, so the partial tree holds the first batch only."""
    _launch(tmp_path)
    cfg = _config(tmp_path, iterations=1, versions=1)
    with pytest.raises(KeyboardInterrupt):
        DreamRSI(cfg, make_task(str(tmp_path)), _Interrupting(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(tmp_path))
    (row,) = report["iterations"]
    assert (row["partial"], row["probes"], row["attempts"], row["error"]) == (
        True,
        2,
        2,
        "KeyboardInterrupt: Ctrl-C",
    )
    assert report["versions"] == []  # offline never ran


def test_the_health_table_counts_each_iterations_outcomes(toy_run):
    report, _ = toy_run
    keys = ("attempts", "successes", "fail_classes", "agent_timeouts", "no_program")
    keys += ("evaluator_crashed", "baseline_score", "best_score", "error")
    assert [{k: i[k] for k in keys} for i in report["iterations"]] == [
        {"attempts": 4, "successes": 2, "fail_classes": {"no_program": 2, "ok": 2}}
        | {"agent_timeouts": 1, "no_program": 2, "evaluator_crashed": 0}
        | {"baseline_score": 1.0, "best_score": 1.1, "error": None},
        {"attempts": 4, "successes": 3, "fail_classes": {"no_program": 1, "ok": 3}}
        | {"agent_timeouts": 0, "no_program": 1, "evaluator_crashed": 0}
        | {"baseline_score": 1.0, "best_score": 1.2, "error": None},
    ]


class _NonZeroExit:
    """Adds 0.1 to x, except b0a1 whose agent exits 1 without touching the program."""

    def __call__(self, prompt, *, cwd, target):
        if os.path.basename(target) == "attempt_b000_a001":
            return {"returncode": 1}
        path = os.path.join(target, PROGRAM)
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:
            json.dump({"x": round(x + 0.1, 6)}, f)
        return {"returncode": 0}


def test_a_nonzero_agent_exit_is_counted_as_a_failure_not_a_silent_success(tmp_path, stub_prompts):
    """b0a1's program is byte-identical to its parent because the agent never touched it, not
    because it chose to; the health table must not read that as an ordinary "ok" success."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), _NonZeroExit(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(workdir))
    (row,) = report["iterations"]
    assert row["agent_failures"] == 1
    (case,) = [c for c in report["untouched"]["cases"] if c["cell"] == "b0a1"]
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
    """D2a's committed evidence predates the live-plan signal, the next live plan, the loop's
    untouched flag and the repeat setting: the report says so, rather than reading 0, False,
    "n/a" or "-"."""
    report = report_run.build_report(os.path.join(RECON, "evidence", "d2a-lasso", "workdir"))
    assert [v["beyond_support"] for v in report["versions"]] == [None, None, None]
    assert [v["next_live_plan"] for v in report["versions"]] == [None, None, None]
    assert (report["eval_repeats"], report["untouched"]["loop_flagged"]) == (None, None)
    md = report_run.markdown(report)
    assert "Plans beyond a replayed tree with the trace fields cleared: not measured." in md
    assert "Next live plans: not measured." in md
    assert "The loop's own untouched flag: not measured." in md
    assert "Evaluation repeats in force: not measured." in md
    assert md.count("| 12 | 0 | - | - | not measured | not measured | not measured |") == 3
    assert "| 0.0134266 | 0.0167488 | not measured |" in md  # the repeat spread


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


def test_a_program_missing_from_disk_cannot_be_checked_for_untouched(tmp_path, stub_prompts):
    """report_run must say what the untouched check covered, never render "0 of N" as if it had
    checked every attempt when the program files themselves are gone (e.g. a stripped copy)."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), ScriptedDiscoveryAgent(), ScriptedPolicyAgent()).run()
    copy_dir = tmp_path / "copy"
    shutil.copytree(workdir, copy_dir)
    for path in glob.glob(
        os.path.join(str(copy_dir), "runs", "iter*", "tree", "attempt_*", PROGRAM)
    ):
        os.remove(path)
    report = report_run.build_report(str(copy_dir))
    (row,) = report["iterations"]
    assert row["attempts"] == 4  # the scripted agent always writes a program: no "no_program" cell
    assert row["untouched_unchecked"] == 4
    assert report["untouched"]["unchecked"] == 4
    assert report["untouched"]["cases"] == []
    md = report_run.markdown(report)
    assert "Untouched check covers 0 of 4 attempts; 4 had no program file on disk." in md
    assert "0 of 4 attempts left" not in md  # never claim full coverage found nothing untouched


def test_a_workdir_without_the_loops_flag_still_lists_what_the_byte_check_finds(
    tmp_path, stub_prompts
):
    """D2a's untouched attempts were found by comparing bytes on a live workdir that predates the
    loop's flag; with the flag absent, the byte check alone must keep listing them."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), _NonZeroExit(), ScriptedPolicyAgent()).run()
    pattern = workdir / "runs" / "iter*" / "tree" / "attempt_*" / "eval" / "score.json"
    for path in glob.glob(str(pattern)):
        with open(path) as f:
            score = json.load(f)
        del score["untouched"]
        with open(path, "w") as f:
            json.dump(score, f)
    report = report_run.build_report(str(workdir))
    (case,) = report["untouched"]["cases"]
    assert (case["cell"], case["loop_untouched"], case["byte_identical"]) == ("b0a1", None, True)
    assert report["untouched"]["loop_flagged"] is None


def test_a_stderr_tail_the_loop_never_recorded_reads_not_measured_not_empty(tmp_path, stub_prompts):
    """A score.json from before D2b has no agent_stderr, while an agent that printed nothing has
    one with nothing in it; the report must not show both as "-"."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), _NonZeroExit(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(workdir))
    (case,) = report["untouched"]["cases"]
    assert (case["cell"], case["agent_stderr"]) == ("b0a1", "")  # recorded, and empty
    assert "| True | True | - |" in report_run.markdown(report)
    pattern = workdir / "runs" / "iter*" / "tree" / "attempt_*" / "eval" / "score.json"
    for path in glob.glob(str(pattern)):
        with open(path) as f:
            score = json.load(f)
        del score["agent_stderr"]
        with open(path, "w") as f:
            json.dump(score, f)
    report = report_run.build_report(str(workdir))
    (case,) = report["untouched"]["cases"]
    assert (case["cell"], case["agent_stderr"]) == ("b0a1", None)
    assert "| True | True | not measured |" in report_run.markdown(report)


def test_an_untouched_attempt_whose_program_is_gone_is_not_counted_as_checked(
    tmp_path, stub_prompts
):
    """An untouched attempt is "no_program", yet its program was on disk; once a stripped copy
    drops it, the byte check has not covered it, even though the loop's flag still lists it."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), _NonZeroExit(), ScriptedPolicyAgent()).run()
    for path in glob.glob(str(workdir / "runs" / "iter*" / "tree" / "attempt_*" / PROGRAM)):
        os.remove(path)
    report = report_run.build_report(str(workdir))
    (row,) = report["iterations"]
    assert row["untouched_unchecked"] == row["attempts"]
    (case,) = report["untouched"]["cases"]
    assert (case["cell"], case["loop_untouched"], case["byte_identical"]) == ("b0a1", True, None)


def test_repeats_cut_short_by_a_failed_run_are_not_counted_as_evaluation_noise(
    tmp_path, stub_prompts
):
    """repeated() stops at the first run that fails and keeps the scores so far; the gap between
    a good run and a failed one is a failure, not the spread of a noisy evaluator."""
    workdir = tmp_path / "w"
    workdir.mkdir()
    _launch(workdir, eval_repeats=3)
    cfg = _config(workdir, iterations=1, versions=1)
    DreamRSI(cfg, make_task(str(workdir)), ScriptedDiscoveryAgent(), ScriptedPolicyAgent()).run()
    pattern = workdir / "runs" / "iter0001" / "tree" / "attempt_*" / "eval" / "score.json"
    for path in glob.glob(str(pattern)):
        with open(path) as f:
            score = json.load(f)
        score.update(error="ValueError: boom", repeat_scores=[1.0, 0.0])
        with open(path, "w") as f:
            json.dump(score, f)
    (row,) = report_run.build_report(str(workdir))["iterations"]
    assert row["repeat_spread"] is None


def test_a_version_whose_live_plan_raised_is_counted_not_read_as_within_support(tmp_path):
    """A version whose plan_grid raised when asked without the trace fields, given that replay's
    history, has no plan to compare with the tree; its episodes must not read as simply within
    support."""
    rdir = tmp_path / "r0001_t01_m1"
    (rdir / "proposal_results").mkdir(parents=True)
    episode = {"out_of_support": False, "error": None, "beyond_support": False}
    fallback = {"branch_count": 2, "refine_count": 1, "fallback": True}
    episodes = [
        episode | {"live_plan": None, "live_plan_error": "TypeError: '>' not supported"},
        episode | {"live_plan": fallback, "live_plan_error": None},
    ]
    lines = "".join(json.dumps(e) + "\n" for e in episodes)
    (rdir / "proposal_results" / "policy_execution_traces.jsonl").write_text(lines)
    version = report_run.analyse_version(str(rdir), {}, (2, 1), set())
    assert (version["beyond_support"], version["live_plan_errors"]) == (0, 1)


def test_copy_evidence_does_not_follow_a_symlink_out_of_the_workdir(tmp_path):
    workdir = tmp_path / "w"
    outside = tmp_path / "outside.md"
    outside.write_text("not part of the workdir")
    node = workdir / "runs" / "iter0001" / "tree" / "attempt_b000_a000"
    node.mkdir(parents=True)
    (node / "proposal.md").symlink_to(outside)
    dest = tmp_path / "out"
    result = report_run.copy_evidence(str(workdir), str(dest))
    assert (result["copied"], result["withheld"]) == (
        0,
        ["runs/iter0001/tree/attempt_b000_a000/proposal.md"],
    )
    assert not (dest / "runs" / "iter0001" / "tree" / "attempt_b000_a000" / "proposal.md").exists()


def test_the_evidence_subset_carries_no_program_and_withholds_a_quoted_one(toy_run):
    """44 files: state.json and launches.jsonl; three per frozen iteration (6); eight score.json;
    three error.txt (the two untouched attempts, the deleted program); seven of eight proposals;
    method.py and two sweep files for each of six versions (18). SimpleTES programs are AGPL and
    stay out."""
    report, out = toy_run
    copied = [f for _, _, files in os.walk(out / "workdir") for f in files]
    assert PROGRAM not in copied
    assert len(copied) == 44
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


def test_the_home_and_temp_directories_themselves_are_redacted_not_only_paths_under_them(
    tmp_path, monkeypatch
):
    """An environment dump or a working directory can name the directory itself, with no
    separator after it; a longer name that merely starts with it is someone else's."""
    home, temp = tmp_path / "home" / "someone", tmp_path / "tmp"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp))
    node = _attempt_dir(tmp_path / "w")
    (node / "error.txt").write_text(f"HOME={home}\nTMPDIR={temp}\nran in {home}.\n{home}2/x\n")
    result = report_run.copy_evidence(str(tmp_path / "w"), str(tmp_path / "out"))
    out = tmp_path / "out" / "runs" / "iter0001" / "tree" / "attempt_b000_a000" / "error.txt"
    assert out.read_text() == f"HOME=~\nTMPDIR=$TMPDIR\nran in ~.\n{home}2/x\n"
    assert result["redacted"]["home_paths"] == {"matches": 2, "files": 1}
    assert result["redacted"]["temp_paths"] == {"matches": 1, "files": 1}


def test_a_json_file_a_killed_run_cut_short_is_withheld_and_the_copy_goes_on(tmp_path):
    """A SIGKILL can leave score.json half-written. Unparsed it cannot be checked for a quoted
    program, so it is withheld, and the rest of the evidence is still copied."""
    node = _attempt_dir(tmp_path / "w")
    (node / "eval" / "score.json").write_text('{"error": "x.cpp:79:27: error\\n   79 |   dou')
    (node / "error.txt").write_text("agent timed out\n")
    result = report_run.copy_evidence(str(tmp_path / "w"), str(tmp_path / "out"))
    out = tmp_path / "out" / "runs" / "iter0001" / "tree" / "attempt_b000_a000"
    rel = os.path.join("runs", "iter0001", "tree", "attempt_b000_a000", "eval", "score.json")
    assert result["withheld"] == [rel] and result["copied"] == 1
    assert not (out / "eval" / "score.json").exists()
    assert (out / "error.txt").read_text() == "agent timed out\n"


def test_the_published_report_is_redacted_like_the_evidence(toy_run):
    """report.json and report.md are published beside the evidence and quote stderr tails."""
    report, out = toy_run
    temp = tempfile.gettempdir() + os.sep
    assert temp not in (out / "report.json").read_text()
    assert temp not in (out / "report.md").read_text()
    assert "$TMPDIR" + os.sep in report["workdir"]


def test_the_archive_digest_is_recorded(toy_run):
    report, _ = toy_run
    assert report["archive"] == {
        "file": "archive.bin",
        "sha256": "0eb3e36bfb24dcd9bb1d1bece1531216b59539a8fde17ee80224af0653c92aa3",
    }


def test_the_reports_resume_source_rule_is_the_loops(tmp_path):
    """Programs exist at b0a0, b0a2 and b2a1 only; report and loop must agree on every cell."""
    task = make_task(str(tmp_path))
    tree = tmp_path / "tree"
    for node in ("attempt_b000_a000", "attempt_b000_a002", "attempt_b002_a001"):
        (tree / node).mkdir(parents=True)
        (tree / node / PROGRAM).write_text("{}")
    q = LiveQuestion(task, ScriptedDiscoveryAgent(), str(tree), str(tmp_path / "h"), 1.0, 2, 3, 3)
    for b in range(3):
        for a in range(4):
            path, _ = report_run.resume_source(str(tree), task.baseline_dir, PROGRAM, b, a)
            assert path == q._resume_from(b, a), (b, a)
    source = {
        (b, a): report_run.resume_source(str(tree), task.baseline_dir, PROGRAM, b, a)[1]
        for b, a in ((0, 0), (0, 2), (0, 3), (2, 1), (2, 3))
    }
    assert source == {(0, 0): None, (0, 2): 0, (0, 3): 2, (2, 1): None, (2, 3): 1}


def test_the_smoke_tests_noise_is_the_relative_spread_of_its_repeats(tmp_path):
    runs = [
        {"combined_score": 0.5, "geo_mean_sol_ms": 2.0},
        {"combined_score": 0.625, "geo_mean_sol_ms": 1.6},
    ]
    (tmp_path / "host.json").write_text(json.dumps({"labels": [], "results": {"seed": runs}}))
    assert report_run.noise(str(tmp_path / "host.json")) == {
        "seed": {"scores": [0.5, 0.625], "geo_mean_ms": [2.0, 1.6], "spread": 0.25}
    }


def test_free_text_in_a_table_cell_cannot_break_the_markdown_table():
    assert report_run._cell("line one\nline two | still one cell") == (
        "line one line two \\| still one cell"
    )


def test_the_markdown_report_answers_the_four_questions(toy_run):
    _, out = toy_run
    md = (out / "report.md").read_text()
    for answer in (
        "## 1. Per-round call budget",
        "fallback 2 x 1 = 4 calls, hard max 3 x 2 = 9 calls",
        "2 version(s) replayed on clipped episodes; 0 of them deployed.",
        "2 of 8 attempts left their resume source byte for byte.",
        "Versions scored minus infinity for an empty batch: r0003_t01_m2, r0006_t02_m2.",
    ):
        assert answer in md
