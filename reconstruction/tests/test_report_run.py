"""Pins for scripts/report_run.py, the D2a evidence report (spec section 5).

One scripted run on the toy task stands in for the real-agent run: a 2 x 1 fallback grid (two
branches of two attempts), two iterations, three policy versions per iteration, two workers. The
discovery agent adds 0.1 to the program's x except where SCRIPT says otherwise; the policy agent
alternates a version that plans one branch wider than any recorded tree (WIDE) and one that ends
on an empty batch (EMPTY). Every expected number below is counted by hand from that script.
"""

import json
import os

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
    ("iter0001", "attempt_b000_a000"): "untouched",  # identical to the baseline
    ("iter0001", "attempt_b001_a001"): "timeout",  # identical to its parent, agent timed out
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


def _launch(workdir) -> None:
    """The line scripts/run_dream_rsi.py appends to launches.jsonl, cut to what the report reads."""
    line = {
        "task": {"name": "toy", "eval_program": PROGRAM},
        "config": {"fallback_grid": [2, 1], "hard_max_grid": [3, 2]},
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
    assert report["out_of_support"] == {
        "flagged": ["r0002_t01_m1", "r0005_t02_m1"],
        "flagged_deployed": [],  # clipped to the tree, WIDE ties m0, and ties keep the earlier
    }


def test_an_untouched_attempt_is_reported_with_its_source_and_both_scores(toy_run):
    report, _ = toy_run
    assert report["untouched"] == {
        "attempts": 8,
        "cases": [
            {"iteration": 1, "cell": "b0a0", "source": "baseline", "fail_class": "ok"}
            | {"agent_timed_out": False, "evaluated": True, "score": 1.0, "source_score": 1.0},
            {"iteration": 1, "cell": "b1a1", "source": "parent", "fail_class": "timeout"}
            | {"agent_timed_out": True, "evaluated": True, "score": 1.1, "source_score": 1.1},
        ],
    }


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
        {"attempts": 4, "successes": 3, "fail_classes": {"ok": 3, "timeout": 1}}
        | {"agent_timeouts": 1, "no_program": 0, "evaluator_crashed": 0}
        | {"baseline_score": 1.0, "best_score": 1.1, "error": None},
        {"attempts": 4, "successes": 3, "fail_classes": {"no_program": 1, "ok": 3}}
        | {"agent_timeouts": 0, "no_program": 1, "evaluator_crashed": 0}
        | {"baseline_score": 1.0, "best_score": 1.2, "error": None},
    ]


def test_the_evidence_subset_carries_no_program_and_withholds_a_quoted_one(toy_run):
    """43 files: state.json and launches.jsonl; three per frozen iteration (6); eight score.json;
    two error.txt (the timeout, the deleted program); seven of eight proposals; method.py and two
    sweep files for each of six versions (18). SimpleTES programs are AGPL and stay out."""
    report, out = toy_run
    copied = [f for _, _, files in os.walk(out / "workdir") for f in files]
    assert PROGRAM not in copied
    assert len(copied) == 43
    assert report["evidence"] == {
        "copied": 43,
        "withheld": ["runs/iter0002/tree/attempt_b000_a001/proposal.md"],
    }


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
