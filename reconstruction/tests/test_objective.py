import pytest

from see.objective import attainment, beta_sweep, eq1_value, pareto_auc, score_of
from see.policies.parallel_refine import ParallelRefine
from see.policies.portfolio import OptimalPolicy
from see.policy.api import GridPlan, LLMDesignedMethod, SimResult, finalize_result
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

        def solve(self, question, budget=None):
            question.reset()
            question.probe_batch(["b0a5"])

    report, execs = beta_sweep(Illegal, [synthetic_trace(0)], betas=(0.5,))
    assert not report["valid"] and report["pareto"]["reward"] == float("-inf")
    assert "IllegalBatch" in execs[0]["error"]


def test_plan_grid_restricts_replay_and_flags_out_of_support():
    class Wide(ParallelRefine):
        def plan_grid(self, context):
            return GridPlan(context.hard_max_branch_count, 1, reason="test")

    t = synthetic_trace(0, branches=5, refine=6, max_parallelism=5)
    report, execs = beta_sweep(Wide, [t], betas=(0.5,))
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
