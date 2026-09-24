"""Example adaptive policy written against Listing 2's requirements.

Not from the paper: it stands in for a policy-development agent's output so
the loop runs end to end without an LLM, and shows the API is sufficient.

Prefix signals: each branch's trajectory rebuilt from observed(): successful
anchor and the attempt that set it, the trailing failure episode and its
fail classes, and the branch's strength relative to the best anchor.
Batch rule: one portfolio per round, up to max_parallelism: exploration
(unopened roots while active width is below target, one more when every
active branch has stalled), at most one recovery (best-anchored branch whose
latest probe is a repairable failure, only into a slot no refinement needs),
then exploitation (non-stalled branches by relative strength), then stalled
branches kept in reserve when the schedule is patient enough.
Beta schedule: _schedule(beta) sets patience, repair budget, target width,
the keep threshold and whether reserve branches may fill idle workers.
Default beta 0.6, Listing 2's value when live history is insufficient.
Safeguards: closure is recomputed from whole trajectories every round, so a
later success reopens a branch; a zero-valid or compile failure closes a
branch only after the repair budget; roots are taken in creation order,
never by branch id; stopping happens only when no role has a candidate.
Grid rule: see plan_grid.
"""

import math

from see.policy.api import (
    GridPlan,
    LLMDesignedMethod,
    SimResult,
    _budget_done,
    _record_curve,
    finalize_result,
)
from see.policy.observation_signal import HARD_FAIL_CLASSES, is_repairable_failure, is_success

NAME = "OptimalPolicy"


class OptimalPolicy(LLMDesignedMethod):
    NAME = NAME

    def __init__(self, config=None):
        super().__init__(config)
        self.beta = float(self.config.get("beta", 0.6))

    @staticmethod
    def _schedule(beta):
        return {
            "patience": 1 + round(3 * beta),  # probes after the anchor before "stalled"
            "repairs": 1 + round(2 * beta),  # consecutive repairable failures tolerated
            "width": 0.3 + 0.7 * beta,  # target share of branch slots kept active
            "keep": 0.5 * (1.0 - beta),  # stalled branches below this strength close
            "reserve": beta >= 0.5,  # stalled-but-kept branches may fill idle workers
        }

    # -- prefix analysis ------------------------------------------------------
    @staticmethod
    def _trajectories(prefix, baseline):
        by_branch = {}
        for obs in prefix.values():
            by_branch.setdefault(obs.branch, []).append(obs)
        states = {}
        for b, obs_list in by_branch.items():
            traj = sorted(obs_list, key=lambda o: o.attempt)
            anchor = anchor_at = None
            for o in traj:
                if is_success(o) and o.score is not None and (anchor is None or o.score > anchor):
                    anchor, anchor_at = o.score, o.attempt
            trailing = []
            for o in reversed(traj):
                if is_success(o):
                    break
                trailing.append(o)
            states[b] = {
                "last": traj[-1],
                "anchor": anchor,
                "fails": len(trailing),
                "hard": any(o.fail_class in HARD_FAIL_CLASSES for o in trailing),
                "since": None if anchor_at is None else traj[-1].attempt - anchor_at,
            }
        anchors = [s["anchor"] for s in states.values() if s["anchor"] is not None]
        top = max(anchors, default=baseline)
        for s in states.values():
            if s["anchor"] is None:
                s["strength"] = 0.0
            elif top > baseline:
                s["strength"] = max(0.0, (s["anchor"] - baseline) / (top - baseline))
            else:
                s["strength"] = 1.0 if s["anchor"] >= top else 0.0
        return states

    def _closed(self, s, sched):
        if s["hard"] and (s["anchor"] is None or s["fails"] >= 2):
            return True  # environment/dependency failure episode: hard-unrecoverable
        if s["fails"] > sched["repairs"]:
            return True  # repair budget spent on this failure episode
        stalled = s["since"] is not None and s["since"] > sched["patience"]
        return stalled and s["strength"] < sched["keep"]  # repeatedly unpromising

    def _select_batch(self, question, sched):
        prefix = question.observed()
        states = self._trajectories(prefix, question.baseline_score)
        roots = question.legal_roots()
        frontier = {
            question.meta(c).branch: c for c in question.legal_actions() if c not in set(roots)
        }
        live = {b: s for b, s in states.items() if b in frontier and not self._closed(s, sched)}

        def stalled(s):
            return s["since"] is not None and s["since"] > sched["patience"]

        exploit = sorted(
            (b for b, s in live.items() if is_success(s["last"]) and not stalled(s)),
            key=lambda b: (-live[b]["strength"], -(live[b]["last"].delta_vs_parent or 0.0), b),
        )
        recover = sorted(
            (b for b, s in live.items() if is_repairable_failure(s["last"])),
            key=lambda b: (-live[b]["strength"], live[b]["fails"], b),
        )
        reserve = sorted(
            (b for b, s in live.items() if is_success(s["last"]) and stalled(s)),
            key=lambda b: (-live[b]["strength"], b),
        )

        slots = question.max_parallelism
        target = max(1, math.ceil(sched["width"] * (len(states) + len(roots))))
        n_roots = max(0, target - len(live))
        if live and all(stalled(s) for s in live.values()):
            n_roots += 1  # plateau: widen
        batch = roots[: min(n_roots, slots)]
        if recover and len(batch) + len(exploit) < slots:
            batch.append(frontier[recover[0]])
        batch += [frontier[b] for b in exploit][: slots - len(batch)]
        if sched["reserve"]:
            batch += [frontier[b] for b in reserve][: slots - len(batch)]
        return batch

    def solve(self, question, budget=None):
        question.reset()
        res = SimResult()
        sched = self._schedule(self.beta)
        while not _budget_done(question, budget):
            batch = self._select_batch(question, sched)
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context):
        """Width vs depth from completed live cycles (Listing 2 lines 222-233)."""
        cap = lambda w, r: (
            min(max(1, w), context.hard_max_branch_count),
            min(max(0, r), context.hard_max_refine_count),
        )
        hist = [h for h in context.history if h.get("best_score") is not None]
        if not hist:
            w, r = cap(context.fallback_branch_count, context.fallback_refine_count)
            return GridPlan(
                w, r, reason="no completed live cycle: bootstrap from the fallback grid"
            )
        last = hist[-1]
        w, r = last["effective_grid"]["branch_count"], last["effective_grid"]["refine_count"]
        depth = last.get("best_attempt")
        improved = len(hist) < 2 or last["best_score"] > hist[-2]["best_score"]
        if depth is None:
            w, r = cap(w - 1, r - 1)
            return GridPlan(w, r, reason="last cycle had no successful cell: shrink conservatively")
        if depth <= r // 3 and improved:
            w, r = cap(w + 2, r - 2)
            reason = f"best came early (attempt {depth} of {r}) and live best improved: widen"
        elif depth >= (2 * r) // 3:
            w, r = cap(w - 1, r + 2)
            reason = f"best came late (attempt {depth} of {r}): deepen"
        elif not improved:
            w, r = cap(w + 2, r)
            reason = "live best plateaued with mid-depth gains: widen"
        else:
            w, r = cap(w, r)
            reason = "no clear width/depth signal: hold the last grid"
        return GridPlan(w, r, reason=reason)
