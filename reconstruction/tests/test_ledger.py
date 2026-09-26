"""Pins for the reconstruction choices GAPS.md §3 records.

Each test is a hand-computed case for one ruling, named for the claim it protects, so the
ledger cannot drift from the code without a test going red.
"""

import json
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
