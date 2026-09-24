"""Replay objectives: the paper's Eq. (1) and Listing 2's beta-sweep reward.

The paper gives two objectives that do not agree. Sec. 3, Eq. (1), scores one
episode per replay world i:

    V_i = max revealed score - beta1 * N_i + beta2 * N_i / max(1, k_i)

and selects the policy version with the highest mean V. Listing 2, the prompt
the policy-development agent actually receives, sweeps the policy's beta knob
and scores the whole sweep:

    pareto.reward = pareto.auc - lambda * parallel_penalty

Listing 2 pins the parallel penalty exactly (mean over the sweep of
effective_sequential_rounds / total_probes, one effective round per
ceil(k / W) cells). It does not give the attainment normalisation, the AUC
construction, the beta grid, lambda, or beta1/beta2; the choices below are
inferred and exposed as parameters (see GAPS.md).
"""
from __future__ import annotations

import dataclasses
import statistics
import traceback
from typing import Callable, Optional, Sequence

from see.policy.api import GridPlan, GridPlanningContext
from see.world import ReplayQuestion, Trace

DEFAULT_BETAS = tuple(round(0.1 * i, 1) for i in range(11))  # Listing 2 clamps beta to [0, 1]
DEFAULT_LAMBDA = 0.1


def attainment(best: Optional[float], trace: Trace) -> float:
    """Best revealed score normalised between the root (baseline) and the trace's ceiling."""
    floor, ceiling = trace.baseline_score, trace.ceiling()
    if ceiling <= floor:
        return 1.0  # nothing recorded beats the root: stopping at once loses nothing
    if best is None:
        return 0.0
    return min(1.0, max(0.0, (best - floor) / (ceiling - floor)))


def eq1_value(best: Optional[float], root_score: float, probes: int, rounds: int,
              beta1: float, beta2: float) -> float:
    """Paper Eq. (1); the max runs over the revealed subtree, root included."""
    quality = root_score if best is None else max(best, root_score)
    return quality - beta1 * probes + beta2 * probes / max(1, rounds)


def pareto_auc(points: Sequence[tuple]) -> tuple:
    """Area under the (work, attainment) Pareto frontier on work in [0, 1].

    Frontier points are joined linearly from (0, 0) and held flat to work = 1.
    Returns (auc, frontier).
    """
    frontier = []
    for work, att in sorted(points, key=lambda p: (p[0], -p[1])):
        if not frontier or att > frontier[-1][1]:
            frontier.append((work, att))
    auc, prev = 0.0, (0.0, 0.0)
    for work, att in frontier:
        auc += (work - prev[0]) * (att + prev[1]) / 2
        prev = (work, att)
    auc += (1.0 - prev[0]) * prev[1]
    return auc, frontier


def validate_plan(plan: Optional[GridPlan], context: GridPlanningContext) -> Optional[GridPlan]:
    """The runner's check (Listing 2 lines 208-210); None means use the fallback grid."""
    if plan is None:
        return None
    if not (1 <= plan.branch_count <= context.hard_max_branch_count
            and 0 <= plan.refine_count <= context.hard_max_refine_count):
        return None
    return plan


def default_context(trace: Trace) -> GridPlanningContext:
    b, r = trace.grid
    return GridPlanningContext(history=(), fallback_branch_count=b, fallback_refine_count=r,
                               hard_max_branch_count=b, hard_max_refine_count=r,
                               max_parallelism=trace.max_parallelism,
                               trace_branch_count=b, trace_refine_count=r)


@dataclasses.dataclass
class Episode:
    trace_id: str
    beta: Optional[float]
    best: Optional[float]
    probes: int
    rounds: int
    effective_rounds: int
    work: float
    attainment: float
    penalty: float
    eq1: float
    plan: Optional[dict]
    out_of_support: bool
    error: Optional[str]
    log: list


def run_episode(policy, trace: Trace, context: GridPlanningContext, *, beta=None,
                use_plan: bool = True, max_rounds: Optional[int] = None,
                beta1: float = 0.0, beta2: float = 0.0, record: bool = False) -> Episode:
    plan, error, q = None, None, None
    try:
        if use_plan:
            plan = validate_plan(policy.plan_grid(context), context)
        q = ReplayQuestion(trace, branch_count=plan.branch_count if plan else None,
                           refine_count=plan.refine_count if plan else None,
                           max_rounds=max_rounds, record_episode=record)
        policy.solve(q, budget=None)
    except Exception:
        error = traceback.format_exc(limit=4)
        if q is None:
            q = ReplayQuestion(trace)
    probes = q.budget_spent
    return Episode(
        trace_id=trace.trace_id, beta=beta, best=q.best_so_far, probes=probes,
        rounds=q.decision_rounds, effective_rounds=q.effective_sequential_rounds,
        work=probes / max(1, len(trace)), attainment=attainment(q.best_so_far, trace),
        penalty=q.effective_sequential_rounds / probes if probes else 1.0,
        eq1=eq1_value(q.best_so_far, trace.baseline_score, probes, q.decision_rounds, beta1, beta2),
        plan=dataclasses.asdict(plan) if plan else None, out_of_support=q.out_of_support,
        error=error, log=q.episode)


def _mean(xs):
    xs = list(xs)
    return statistics.fmean(xs) if xs else float("nan")


def beta_sweep(policy_cls: Callable, traces: Sequence[Trace], *,
               context_for: Callable[[Trace], GridPlanningContext] = default_context,
               betas: Sequence[float] = DEFAULT_BETAS, lam: float = DEFAULT_LAMBDA,
               beta1: float = 0.0, beta2: float = 0.0, use_plan: bool = True,
               max_rounds: Optional[int] = None, record: bool = True) -> tuple:
    """Evaluate one policy version on every replay world.

    Returns (report, executions): ``report`` is what goes to
    proposal_results/beta_sweep.json; ``executions`` holds one replay episode
    per (frozen trace, beta), for policy_execution_traces.jsonl.
    """
    kw = dict(use_plan=use_plan, max_rounds=max_rounds, beta1=beta1, beta2=beta2, record=record)
    per_beta, executions = [], []
    for beta in betas:
        eps = [run_episode(policy_cls({"beta": beta}), t, context_for(t), beta=beta, **kw)
               for t in traces]
        executions += eps
        per_beta.append({
            "beta": beta, "work": _mean(e.work for e in eps),
            "attainment": _mean(e.attainment for e in eps),
            "penalty": _mean(e.penalty for e in eps), "probes": _mean(e.probes for e in eps),
            "rounds": _mean(e.rounds for e in eps), "eq1": _mean(e.eq1 for e in eps),
        })
    shipped = policy_cls(None)  # the baked-in default beta is what a live episode uses
    default_eps = [run_episode(policy_cls(None), t, context_for(t), beta=None, **kw) for t in traces]
    executions += default_eps
    auc, frontier = pareto_auc([(p["work"], p["attainment"]) for p in per_beta])
    penalty = _mean(p["penalty"] for p in per_beta)
    errors = [e.error for e in executions if e.error]
    report = {
        "policy": getattr(policy_cls, "NAME", policy_cls.__name__),
        "default_beta": getattr(shipped, "beta", None),
        "n_traces": len(traces),
        "valid": not errors,
        "errors": errors[:3],
        "pareto": {"reward": auc - lam * penalty if not errors else float("-inf"),
                   "auc": auc, "parallel_penalty": penalty, "lambda": lam,
                   "frontier": [list(p) for p in frontier]},
        "per_beta": per_beta,
        "eq1": {"beta1": beta1, "beta2": beta2,
                "V": _mean(e.eq1 for e in default_eps) if not errors else float("-inf"),
                "per_trace": {e.trace_id: e.eq1 for e in default_eps}},
        "default_episode": {"attainment": _mean(e.attainment for e in default_eps),
                            "work": _mean(e.work for e in default_eps),
                            "penalty": _mean(e.penalty for e in default_eps)},
        "out_of_support": any(e.out_of_support for e in executions),
    }
    return report, [dataclasses.asdict(e) for e in executions]


def score_of(report: dict, objective: str) -> float:
    """Selection statistic: ``"pareto"`` (Listing 2) or ``"eq1"`` (Sec. 3)."""
    return report["pareto"]["reward"] if objective == "pareto" else report["eq1"]["V"]
