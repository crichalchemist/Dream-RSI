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


def test_eq1_matches_the_paper_formula():
    # V = max s - b1 * N + b2 * N / max(1, k)
    assert eq1_value(2.0, 1.0, probes=12, rounds=3, beta1=0.01, beta2=0.05) == pytest.approx(
        2.0 - 0.12 + 0.05 * 4
    )
    assert eq1_value(None, 1.0, probes=0, rounds=0, beta1=0.5, beta2=0.5) == 1.0  # root only


def test_attainment_normalises_between_root_and_ceiling():
    t = Trace([Cell(0, 0, 0, 1.5), Cell(0, 1, 1, 3.0)], baseline_score=1.0, max_parallelism=1)
    assert attainment(None, t) == 0.0
    assert attainment(2.0, t) == pytest.approx(0.5)
    assert attainment(3.0, t) == 1.0
    flat = Trace([Cell(0, 0, 0, 0.5)], baseline_score=1.0, max_parallelism=1)
    assert attainment(None, flat) == 1.0


def test_pareto_auc_uses_only_non_dominated_points():
    auc, frontier = pareto_auc([(0.5, 0.5), (0.6, 0.4), (1.0, 1.0)])
    assert frontier == [(0.5, 0.5), (1.0, 1.0)]
    assert auc == pytest.approx(0.5 * 0.5 / 2 + 0.5 * (0.5 + 1.0) / 2)
    auc_single, _ = pareto_auc([(0.25, 1.0)])
    assert auc_single == pytest.approx(0.25 / 2 + 0.75)


def test_parallel_refine_reveals_everything_with_full_batches():
    traces = [synthetic_trace(s, branches=6, refine=4, max_parallelism=6) for s in range(3)]
    report, execs = beta_sweep(ParallelRefine, traces, betas=(0.0, 1.0))
    assert report["valid"]
    for p in report["per_beta"]:
        assert p["work"] == 1.0 and p["attainment"] == 1.0
    # beta is ignored: the sweep is degenerate, a single frontier point
    assert len(report["pareto"]["frontier"]) == 1
    for e in execs:
        assert e["penalty"] <= 1.0 and e["effective_rounds"] == e["rounds"]


def test_serial_policy_pays_the_full_parallel_penalty():
    class Serial(LLMDesignedMethod):
        NAME = "Serial"

        def solve(self, question, budget=None):
            question.reset()
            while question.legal_actions():
                question.probe_batch(question.legal_actions()[:1])
            return finalize_result(question, SimResult())

    traces = [synthetic_trace(1, branches=4, refine=3, max_parallelism=4)]
    serial, _ = beta_sweep(Serial, traces, betas=(0.5,))
    batched, _ = beta_sweep(ParallelRefine, traces, betas=(0.5,))
    assert serial["pareto"]["parallel_penalty"] == 1.0
    assert batched["pareto"]["parallel_penalty"] < 0.5
    assert score_of(batched, "pareto") > score_of(serial, "pareto")


def test_crashing_or_illegal_policy_is_scored_minus_infinity():
    class Illegal(LLMDesignedMethod):
        NAME = "Illegal"

        def solve(self, question, budget=None) -> SimResult:
            question.reset()
            question.probe_batch(["b0a5"])
            raise AssertionError("unreachable: the batch above is illegal and must raise")

    report, execs = beta_sweep(Illegal, [synthetic_trace(0)], betas=(0.5,))
    assert not report["valid"] and report["pareto"]["reward"] == float("-inf")
    assert "IllegalBatch" in execs[0]["error"]


def test_plan_grid_restricts_replay_and_flags_out_of_support():
    class Wide(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(context.hard_max_branch_count, 1, reason="test")

    t = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)
    _, execs = beta_sweep(Wide, [t], betas=(0.5,))
    assert all(e["plan"]["refine_count"] == 1 for e in execs)
    assert all(e["probes"] <= 5 * 2 for e in execs)


def test_adaptive_policy_beats_the_parallel_refine_floor_on_synthetic_traces():
    traces = [synthetic_trace(s) for s in range(12)]
    floor, _ = beta_sweep(ParallelRefine, traces)
    adaptive, _ = beta_sweep(OptimalPolicy, traces)
    assert adaptive["valid"]
    works = [p["work"] for p in adaptive["per_beta"]]
    assert works == sorted(works) and works[0] < works[-1]  # beta trades work for attainment
    assert score_of(adaptive, "pareto") > score_of(floor, "pareto")


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
