"""The paper's initial exploration policy pi_1 (Sec. 4), the "parallel-refine floor".

"launches multiple independent exploration workspaces in parallel, with each
workspace maintaining its own local discovery trajectory and repeatedly
refining its current candidate": open every root at once, then refine every
open branch each round until its attempts run out. Beta is ignored, so its
beta sweep is degenerate by construction.
"""
from see.policy.api import (
    GridPlan, LLMDesignedMethod, SimResult, _budget_done, _record_curve, finalize_result,
)

NAME = "ParallelRefine"


class ParallelRefine(LLMDesignedMethod):
    NAME = NAME

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        while not _budget_done(question, budget):
            roots = question.legal_roots()
            frontiers = [c for c in question.legal_actions() if c not in set(roots)]
            batch = (frontiers + roots)[: question.max_parallelism]
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context):
        return GridPlan(context.fallback_branch_count, context.fallback_refine_count,
                        reason="fixed parallel-refine grid (the runner's configured default)")
