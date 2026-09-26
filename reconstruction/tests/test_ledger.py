"""Pins for the reconstruction choices GAPS.md §3 records.

Each test is a hand-computed case for one ruling, named for the claim it protects, so the
ledger cannot drift from the code without a test going red.
"""

import json
import os

import pytest

from see.__main__ import main
from see.live import LiveQuestion
from see.loop import BASELINE_POLICY, DreamRSI, LoopConfig
from see.objective import attainment, beta_sweep, eq1_value, next_live_plan, run_episode
from see.policies.parallel_refine import ParallelRefine
from see.policies.portfolio import OptimalPolicy
from see.policy.api import GridPlan, GridPlanningContext
from see.pool import context_factory, next_context
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


class _FailsAtBetaOne(_DefaultBeta045):
    def solve(self, question, budget=None):
        if self.beta == 1.0:
            raise RuntimeError("only the grid episode at beta 1.0 fails")
        return super().solve(question, budget)


def test_an_error_in_either_episode_set_invalidates_both_objectives():
    cases = ((_FailsAtItsDefault, [False, False, True]), (_FailsAtBetaOne, [False, True, False]))
    for policy_cls, failing in cases:
        report, execs = beta_sweep(policy_cls, [synthetic_trace(0)], betas=(0.0, 1.0))
        assert [bool(e["error"]) for e in execs] == failing
        assert not report["valid"]
        assert report["pareto"]["reward"] == report["eq1"]["V"] == float("-inf")


def test_an_out_of_support_plan_is_flagged_and_scored_as_its_clipped_grid():
    """Replay clips a plan wider than the recorded tree to the tree and scores it like the
    clipped plan; the flag is informational (zero versus clip, kept after the D2a run, GAPS §3)."""
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
    assert wide.plan == {
        "branch_count": 8,
        "refine_count": 8,
        "reason": "wider and deeper than the tree",
    }
    report, _ = beta_sweep(Wide, [trace], context_for=lambda t: context, betas=(0.5,))
    assert report["out_of_support"] is True and report["valid"]


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


def test_a_policy_that_clamps_to_the_trace_fields_is_flagged_with_its_reward_unchanged():
    """A policy that clamps its plan to the replay-only trace fields is never out of support in
    replay. Each episode also records the plan the policy makes with those fields cleared, which
    flags it; the flag never changes what the episode scores. D2a's deployed version is not caught
    this way: its clamp bound only once history was present, and a one-trace pool's replay has
    none (GAPS §3, "Live-plan signal in replay")."""
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


class _OneRoot(ParallelRefine):
    """Probes one root and stops, so cells stay legal after solve returns."""

    def solve(self, question, budget=None):
        question.reset()
        question.probe_batch(question.legal_roots()[:1])


class _ProbesFromPlanGrid(_OneRoot):
    """Keeps the question from solve and probes it again whenever plan_grid is asked later."""

    def solve(self, question, budget=None):
        self.kept = question
        return super().solve(question, budget)

    def plan_grid(self, context):
        kept = getattr(self, "kept", None)
        if kept is not None:
            kept.probe_batch(kept.legal_actions()[: kept.max_parallelism])
        return super().plan_grid(context)


def test_a_policy_that_probes_its_kept_question_from_plan_grid_cannot_change_its_own_score():
    """The extra plan_grid call comes after the episode is scored, so a policy that kept the
    question from solve and probes it there scores exactly what the same policy scores without."""
    trace = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)
    probing = _ProbesFromPlanGrid(None)
    kept = run_episode(probing, trace, _support_context(), record=True)
    quiet = run_episode(_OneRoot(None), trace, _support_context(), record=True)
    fields = ("probes", "best", "rounds", "effective_rounds", "attainment", "penalty", "eq1", "log")
    assert [getattr(kept, f) for f in fields] == [getattr(quiet, f) for f in fields]
    assert probing.kept.budget_spent > kept.probes  # the extra call did probe


class _WidensWithHistory(ParallelRefine):
    """One branch more for every recorded live cycle: a plan only history makes wider."""

    def plan_grid(self, context):
        return GridPlan(2 + len(context.history), 2, reason="one branch per recorded cycle")


def test_a_plan_that_widens_with_history_is_flagged_only_as_the_next_live_plan():
    """Replay gives each tree only the cycles before it, so this policy plans every recorded
    tree's own grid there and no episode is beyond support. online() plans with every cycle, one
    branch wider than any tree, and only the next live plan sees that (GAPS §3, "Next live
    plan")."""
    pool = [
        (synthetic_trace(i, branches=2 + i, refine=2, max_parallelism=4), {"iteration": i + 1})
        for i in range(2)
    ]
    report, _ = beta_sweep(
        _WidensWithHistory,
        [t for t, _ in pool],
        context_for=context_factory(pool, (2, 2), (8, 8)),
        betas=(0.5,),
    )
    assert (report["out_of_support"], report["beyond_support"], report["valid"]) == (
        False,
        False,
        True,
    )
    context = next_context(pool, (2, 2), (8, 8), 4)
    assert next_live_plan(_WidensWithHistory, context, [t.grid for t, _ in pool]) == {
        "branch_count": 4,
        "refine_count": 2,
        "fallback": False,
        "beyond_support": True,
    }


class _Plans5x5(ParallelRefine):
    def plan_grid(self, context):
        return GridPlan(5, 5, reason="five by five")


def test_a_next_live_plan_is_covered_only_by_one_tree_at_least_as_wide_and_deep():
    """A plan inside an older, larger tree is within support however small the newest tree is;
    two trees that cover its width and its depth only between them do not cover it."""
    context = GridPlanningContext((), 2, 2, 8, 8, 4)
    beyond = [
        next_live_plan(_Plans5x5, context, grids)["beyond_support"]
        for grids in ([(6, 6), (2, 2)], [(5, 5)], [(6, 2), (2, 6)])
    ]
    assert beyond == [False, False, True]


class _RejectedAtItsDefault(_DefaultBeta045):
    """Plans past the hard maximum at its baked-in default beta and 3 x 2 at any other."""

    def plan_grid(self, context):
        if self.beta == 0.45:
            return GridPlan(9, 9, reason="past the caps")
        return GridPlan(3, 2, reason="a swept beta")


def test_the_next_live_plan_is_made_at_the_default_beta_and_falls_back_as_online_does():
    """online() plans with a fresh instance at the baked-in default beta and runs the fallback
    grid when validate_plan rejects the plan; the next live plan does the same."""
    context = GridPlanningContext((), 2, 1, 8, 8, 4)
    assert next_live_plan(_RejectedAtItsDefault, context, [(4, 4)]) == {
        "branch_count": 2,
        "refine_count": 1,
        "fallback": True,
        "beyond_support": False,
    }


D2A = os.path.join(os.path.dirname(__file__), "..", "evidence", "d2a-lasso", "workdir")
D2A_M2 = os.path.join(D2A, "policy_dev", "history", "r0003_t01_m2")
D2A_CAPS = ("--fallback", "4", "3", "--hard-max", "6", "4")  # launches.jsonl's config


def _sweep_d2a(method: str, out, *extra: str) -> dict:
    pool = os.path.join(D2A, "trace_pool")
    main(["sweep", "--method", method, "--pool", pool, "--out", str(out), *D2A_CAPS, *extra])
    with open(os.path.join(out, "beta_sweep.json")) as f:
        return json.load(f)


def test_d2as_deployed_version_plans_a_next_cycle_its_one_tree_does_not_cover(tmp_path):
    """D2a deployed m2, and its stopped iteration 2 planned 4 x 4 over a 4 x 3 tree. Replay gives
    that one tree no history, so no m2 episode is beyond support; the next live plan, made with
    iteration 1's manifest, is 4 x 4 and flagged, and the reward is the one D2a recorded (GAPS §3,
    "Next live plan")."""
    report = _sweep_d2a(os.path.join(D2A_M2, "method.py"), tmp_path)
    with open(os.path.join(D2A_M2, "proposal_results", "beta_sweep.json")) as f:
        recorded = json.load(f)
    assert report["next_live_plan"] == {
        "branch_count": 4,
        "refine_count": 4,
        "fallback": False,
        "beyond_support": True,
    }
    assert (report["beyond_support"], report["valid"]) == (False, True)
    assert report["pareto"]["reward"] == recorded["pareto"]["reward"]


RAISES_WITH_HISTORY = """
from see.policies.parallel_refine import ParallelRefine

NAME = "RaisesWithHistory"


class RaisesWithHistory(ParallelRefine):
    def plan_grid(self, context):
        if context.history:
            raise RuntimeError("no plan with history")
        return super().plan_grid(context)
"""

PARALLEL_REFINE = """
from see.policies.parallel_refine import ParallelRefine

NAME = "ParallelRefine"
"""


def test_a_next_live_plan_that_raises_is_recorded_and_the_sweep_scores_as_without_it(tmp_path):
    """online() would stop on a plan_grid that raises with the full history; the sweep records
    the error as the version's next live plan and scores the version exactly as its twin without
    the raise, since the measure is reported and never selects."""
    for name, source in (("raises", RAISES_WITH_HISTORY), ("twin", PARALLEL_REFINE)):
        (tmp_path / f"{name}.py").write_text(source)
    raises = _sweep_d2a(str(tmp_path / "raises.py"), tmp_path / "raises")
    twin = _sweep_d2a(str(tmp_path / "twin.py"), tmp_path / "twin")
    assert "RuntimeError: no plan with history" in raises["next_live_plan"]["error"]
    assert "error" not in twin["next_live_plan"]
    assert (raises["valid"], raises["errors"], raises["pareto"]) == (
        twin["valid"],
        twin["errors"],
        twin["pareto"],
    )
    assert raises["valid"]


PLANS_ITS_PARALLELISM = """
from see.policies.parallel_refine import ParallelRefine
from see.policy.api import GridPlan

NAME = "PlansItsParallelism"


class PlansItsParallelism(ParallelRefine):
    def plan_grid(self, context):
        return GridPlan(context.max_parallelism, 1, reason="one branch per worker")
"""


def _synthetic_sweep(tmp_path, source: str, widths, *extra: str) -> dict:
    """Sweep ``source`` over one synthetic 3 x 2 tree per width, oldest first, each with a
    manifest, under a 2 x 1 fallback and 8 x 8 caps; the report's next live plan."""
    pool = tmp_path / "trace_pool"
    for i, width in enumerate(widths, start=1):
        cycle = pool / f"iter{i:04d}"
        cycle.mkdir(parents=True, exist_ok=True)
        synthetic_trace(i, branches=3, refine=2, max_parallelism=width).save(
            str(cycle / "trace.json")
        )
        (cycle / "live_cycle_manifest.json").write_text(json.dumps({"iteration": i}))
    method = tmp_path / "method.py"
    method.write_text(source)
    out = tmp_path / f"out{len(list(tmp_path.glob('out*')))}"
    caps = ("--fallback", "2", "1", "--hard-max", "8", "8")
    main(["sweep", "--method", str(method), "--pool", str(pool), "--out", str(out), *caps, *extra])
    with open(out / "beta_sweep.json") as f:
        return json.load(f)["next_live_plan"]


def test_the_next_live_plan_uses_the_given_parallelism_else_the_newest_trees(tmp_path):
    """online() plans with LoopConfig.max_parallelism, which the loop's sweep passes; a sweep run
    by hand without it falls back to the newest recorded tree's: 2 here, where the older and
    wider tree's is 3."""
    default = _synthetic_sweep(tmp_path, PLANS_ITS_PARALLELISM, (3, 2))
    given = _synthetic_sweep(tmp_path, PLANS_ITS_PARALLELISM, (3, 2), "--max-parallelism", "5")
    assert (default["branch_count"], given["branch_count"]) == (2, 5)


TOUCHES_ITS_HISTORY = """
from see.policies.parallel_refine import ParallelRefine
from see.policy.api import GridPlan

NAME = "TouchesItsHistory"


class TouchesItsHistory(ParallelRefine):
    def plan_grid(self, context):
        if any(m.get("touched") for m in context.history):  # edited by an earlier call
            return GridPlan(3, 1, reason="history already touched")
        for m in context.history:
            m["touched"] = True
        return GridPlan(2, 1, reason="history as recorded")
"""


def test_the_next_live_plan_reads_the_manifests_afresh_as_online_does(tmp_path):
    """online() reads every manifest from disk, so a manifest that a replay episode's plan_grid
    edited in place cannot reach the next live plan: 2 x 1 here, not 3 x 1."""
    plan = _synthetic_sweep(tmp_path, TOUCHES_ITS_HISTORY, (2, 2))
    assert (plan["branch_count"], plan["refine_count"]) == (2, 1)


NUMPY_WITH_HISTORY = """
import numpy as np
from see.policies.parallel_refine import ParallelRefine
from see.policy.api import GridPlan

NAME = "NumpyWithHistory"


class NumpyWithHistory(ParallelRefine):
    def plan_grid(self, context):
        if context.history:  # np.clip over past grids, say, returns numpy integers
            return GridPlan(np.int64(4), np.int64(3), reason="numpy counts")
        return super().plan_grid(context)
"""


def test_a_next_live_plan_in_numpy_integers_is_recorded_and_the_sweep_scores_as_without_it(
    tmp_path,
):
    """A numpy integer is an integer: the next live plan records it as one, and the version scores
    as its twin. On a one-tree pool only this planning call has history, so a plan the sweep
    could not write would have turned a valid version into minus infinity."""
    for name, source in (("numpy", NUMPY_WITH_HISTORY), ("twin", PARALLEL_REFINE)):
        (tmp_path / f"{name}.py").write_text(source)
    numpy_plan = _sweep_d2a(str(tmp_path / "numpy.py"), tmp_path / "numpy")
    twin = _sweep_d2a(str(tmp_path / "twin.py"), tmp_path / "twin")
    assert (numpy_plan["valid"], numpy_plan["errors"], numpy_plan["pareto"]) == (
        twin["valid"],
        twin["errors"],
        twin["pareto"],
    )
    assert numpy_plan["next_live_plan"] == {
        "branch_count": 4,
        "refine_count": 3,
        "fallback": False,
        "beyond_support": False,
    }


PLANS_FROM_SOLVED_STATE = """
from see.policies.parallel_refine import ParallelRefine
from see.policy.api import GridPlan

NAME = "PlansFromSolvedState"


class PlansFromSolvedState(ParallelRefine):
    solved = False

    def solve(self, question, budget=None):
        type(self).solved = True
        return super().solve(question, budget)

    def plan_grid(self, context):
        if type(self).solved:  # only in a process that has already run solve
            return GridPlan(4, 4, reason="after a solve")
        return GridPlan(4, 3, reason="freshly loaded")
"""


def test_the_next_live_plan_is_made_by_a_freshly_loaded_policy_as_online_makes_it(tmp_path):
    """online() loads the deployed file afresh in its own process, so what a policy's class kept
    from the sweep's episodes cannot reach its next live plan: 4 x 3 here, not 4 x 4."""
    (tmp_path / "method.py").write_text(PLANS_FROM_SOLVED_STATE)
    report = _sweep_d2a(str(tmp_path / "method.py"), tmp_path / "out")
    plan = report["next_live_plan"]
    assert (plan["branch_count"], plan["refine_count"]) == (4, 3)


STATE_FROM_PLANNING = """
import os

from see.policies.parallel_refine import ParallelRefine

NAME = "StateFromPlanning"
MARKER = {marker!r}  # outside the module, so loading the file afresh keeps it


class StateFromPlanning(ParallelRefine):
    def plan_grid(self, context):
        if context.history:
            open(MARKER, "w").close()
        return super().plan_grid(context)

    def solve(self, question, budget=None):
        if os.path.exists(MARKER):  # probe one root and stop
            question.reset()
            question.probe_batch(question.legal_roots()[:1])
            return None
        return super().solve(question, budget)
"""


def test_what_planning_the_next_cycle_changes_cannot_change_the_versions_score(tmp_path):
    """A plan_grid that leaves state outside its module once history is present (a file here),
    which solve reads, scores exactly as its twin: loading the file afresh cannot undo such
    state, so it is the next live plan being made after the sweep is scored that keeps it out."""
    marker = tmp_path / "planned_with_history"
    (tmp_path / "state.py").write_text(STATE_FROM_PLANNING.format(marker=str(marker)))
    (tmp_path / "twin.py").write_text(PARALLEL_REFINE)
    state = _sweep_d2a(str(tmp_path / "state.py"), tmp_path / "state")
    twin = _sweep_d2a(str(tmp_path / "twin.py"), tmp_path / "twin")
    assert (state["valid"], state["errors"], state["pareto"]) == (
        twin["valid"],
        twin["errors"],
        twin["pareto"],
    )
    assert marker.exists()  # the next live plan did plan with history


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
    with open(floor / "beta_sweep.json") as f:
        report = json.load(f)
    # re-swept on the current pool: a valid sweep of pi_1 over the one recorded tree
    assert (report["valid"], report["policy"], report["n_traces"]) == (True, "ParallelRefine", 1)
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
