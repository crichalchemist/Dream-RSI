"""Policy-facing API, reconstructed from the paper's Listing 2.

Every public name here is one Listing 2 tells the policy-development agent to
import or call. Field lists follow the prompt verbatim; semantics the prompt
leaves open are marked "inferred" and recorded in GAPS.md.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional


@dataclasses.dataclass(frozen=True)
class Observation:
    """What a revealed cell exposes (Listing 2, lines 40-42)."""

    cell_id: str
    branch: int
    attempt: int
    score: Optional[float]
    evaluated: bool
    valid: bool
    fail_class: str
    error: Optional[str]
    delta_vs_baseline: Optional[float]
    delta_vs_parent: Optional[float]
    n_valid: Optional[int]
    n_total: Optional[int]


@dataclasses.dataclass(frozen=True)
class CellMeta:
    """Structural metadata: ``.branch .attempt .parent_id .seq .tags`` (line 35)."""

    cell_id: str
    branch: int
    attempt: int
    parent_id: Optional[str]
    seq: int
    tags: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class GridPlan:
    """``GridPlan(branch_count=W, refine_count=R, reason=...)`` (lines 206-212).

    Branches 0..W-1, attempts 0..R per branch; R counts refinements after the root.
    """

    branch_count: int
    refine_count: int
    reason: str = ""


@dataclasses.dataclass(frozen=True)
class GridPlanningContext:
    """Prefix-safe facts available to ``plan_grid`` (lines 214-220).

    ``history`` holds completed live-cycle manifests, oldest first. The two
    ``trace_*`` fields are set only when the plan is being scored in replay.
    """

    history: tuple
    fallback_branch_count: int
    fallback_refine_count: int
    hard_max_branch_count: int
    hard_max_refine_count: int
    max_parallelism: int
    trace_branch_count: Optional[int] = None
    trace_refine_count: Optional[int] = None


@dataclasses.dataclass
class SimResult:
    """Accumulator returned by ``solve``.

    ``curve`` gets one point per revealed cell via ``_record_curve``; the
    totals are filled by ``finalize_result``.
    """

    curve: list = dataclasses.field(default_factory=list)
    best_score: Optional[float] = None
    total_probes: int = 0
    decision_rounds: int = 0
    effective_sequential_rounds: int = 0


def _budget_done(question, budget) -> bool:
    """Probe budget check. Replay passes ``budget=None``: never exhausted."""
    return budget is not None and question.budget_spent >= budget


def _record_curve(res: SimResult, question) -> None:
    res.curve.append({
        "probes": question.budget_spent,
        "rounds": question.decision_rounds,
        "best": question.best_so_far,
    })


def finalize_result(question, res: SimResult) -> SimResult:
    res.best_score = question.best_so_far
    res.total_probes = question.budget_spent
    res.decision_rounds = question.decision_rounds
    res.effective_sequential_rounds = question.effective_sequential_rounds
    return res


class LLMDesignedMethod:
    """Base class for exploration policies (``class OptimalPolicy(LLMDesignedMethod)``).

    Subclasses read their one scalar in ``__init__`` with
    ``float(self.config.get("beta", default))``. The replay evaluator sweeps
    beta through ``config``; a live episode passes no config, so the policy's
    baked-in default applies.
    """

    NAME = "LLMDesignedMethod"

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})

    def solve(self, question, budget=None) -> SimResult:
        raise NotImplementedError

    def plan_grid(self, context: GridPlanningContext) -> Optional[GridPlan]:
        """Template stub; returning None hands the grid to the runner's fallback."""
        return None
