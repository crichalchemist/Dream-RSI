"""OptimalPolicy: trajectory-ranked, portfolio-batched, beta-scheduled explorer.

Prefix signals (all from ``question.observed()``, ``baseline_score``, legal sets
and structural ``meta``):
  * Per-branch ordered trajectory: successful anchor (best score of an evaluated,
    ``error is None``, ``fail_class == "ok"`` result, regardless of ``valid``),
    latest successful score, parent->child gain / consecutive regressions, the
    current failure episode (weighted consecutive failures since the last
    success, hard count, repeated identical error = failed repair) and explored
    depth.
  * "Soft" evidence: a non-ok result that was still evaluated with fully valid
    output (e.g. an agent timeout whose candidate scored) counts as a discounted
    anchor and half a failure, never as a hard failure.
  * Scores are normalised relative to the prefix only: u = (x - baseline) / s,
    s = max(top - baseline, 0.1 |baseline|), so the best revealed result is ~1.
    Near-valid successes (>= 75% valid outputs but a collapsed score) are floored
    at -0.3 so one correctness slip does not bury a branch.

Failure interpretation: hard-unrecoverable only for environment / dependency
signals (fail_class or error text). ``branch_failed_hard`` and ``n_valid == 0``
are treated as signals, never as closure on their own; compile_other, timeouts,
mismatch, shared-memory/resource and shape/mask/layout/name errors are
repairable. A later success resets the failure episode and re-opens the branch.

Batch rule (each decision round): rank roots, exploitation frontiers (latest
success/soft), exploration candidates (roots and under-explored branches) and
recovery frontiers (latest repairable failure) in one deterministic priority
order. Eligible = within ``margin`` of the top priority. The batch takes the
best exploit, the best exploration and at most one recovery (if within
``recover_margin``), then fills the remaining workers by priority; singletons
are widened with the next credible candidate. A batch never holds a parent and
child (only legal cells, one frontier per branch).

Beta schedule (``_schedule``): higher beta -> more initial roots, more roots per
round, wider eligibility band, longer stagnation patience, softer failure
penalties and later closure; lower beta -> selective, earlier stagnation stop.
Beta is fixed within an episode.

Stop rule (portfolio level): after ``patience`` rounds without prefix progress,
only candidates (any role, including recoveries, under-explored branches and
unopened roots) whose priority reaches ``hp_thr`` may continue; stop only when
none does or nothing is legal. Otherwise every legal non-closed action competes.

Default beta 0.6: the only live cycle (iter0001) ran the beta-free baseline
(beta=null) and the sole archived sweep is degenerate, so history is
insufficient to justify moving away from a moderately exploratory default.

Grid planning (``plan_grid``): uses only completed live manifests. Gains at the
last allowed refinement -> hold width, deepen by 1; plateau with best found
early and depth unused -> widen by 1; dominant hard failures -> shrink both;
otherwise hold. Empty history -> the runner's fallback grid as a bootstrap.
The result is clamped to hard caps and, in replay, to the trace support.

Safeguards: no absolute score cutoffs; shallow weak branches get an
under-explored bonus; repairable failures with a strong anchor are only
deprioritised, not closed; closure needs cumulative evidence; batches are
filled to use workers whenever credible candidates exist.
"""

import math
import re

from see.policy.api import (
    GridPlan,
    GridPlanningContext,  # noqa: F401  (type of plan_grid's argument)
    LLMDesignedMethod,
    SimResult,
    _budget_done,
    _record_curve,
    finalize_result,
)

try:  # helper signal only; never used as unconditional closure
    from see.policy.observation_signal import branch_failed_hard as _sig_failed_hard
except Exception:  # pragma: no cover - optional helper
    _sig_failed_hard = None

NAME = "OptimalPolicy"
DEFAULT_BETA = 0.6

_HARD_CLASSES = {
    "env", "environment", "dependency", "missing_dependency", "infra",
    "infrastructure", "import", "setup", "system", "permission",
}
_HARD_PATTERNS = (
    "modulenotfounderror", "no module named", "importerror", "not installed",
    "dependency", "permission denied", "command not found", "no space left",
    "disk quota", "connection refused", "network is unreachable",
    "driver version", "no cuda-capable device", "environment error",
)
_REPAIRABLE_PATTERNS = (
    "compile", "compilation", "timeout", "timed out", "mismatch", "incorrect",
    "wrong", "shared memory", "shared_memory", "resource", "out of memory",
    "shape", "mask", "layout", "not declared", "undefined", "nameerror",
    "syntax", "index", "assert", "not finite", "nan", "typeerror",
    "valueerror", "runtimeerror", "segmentation",
)


def _g(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def _schedule(beta):
    """Every behavioural threshold, as a function of the fixed episode beta."""
    b = _clip(float(beta), 0.0, 1.0)
    return {
        "init_width_frac": 0.4 + 0.6 * b,     # share of workers opened as roots in round 1
        "roots_per_round": max(1, int(round(b * 4))),
        "margin": 0.35 + 1.0 * b,             # eligibility band below the top priority
        "wide_margin": 0.65 + 1.0 * b,        # band for widening a singleton batch
        "recover_margin": 0.25 + 1.0 * b,     # recovery must be within this of the top
        "patience": 1 + int(round(4 * b)),    # rounds without progress before stagnation
        "hp_thr": 0.85 - 0.5 * b,             # priority needed to continue when stagnant
        "fail_pen": 0.45 - 0.25 * b,          # per weighted consecutive failure
        "fail_close": 2.0 + 3.0 * b,          # weighted failures to close a hopeless branch
        "hard_close": 1 if b < 0.5 else 2,    # hard failures in an episode to close
        "decline_pen": 0.2 - 0.1 * b,         # per consecutive regression
        "unpromising_n": 3 + int(round(2 * b)),  # successes needed to call a branch unpromising
        "under_bonus": 0.3 + 0.4 * b,         # weak-but-under-explored protection
        "under_depth": 1 if b < 0.7 else 2,
        "explore_bonus": 0.2 + 0.5 * b,       # value of an unopened root over the root prior
    }


def _is_success(o):
    evaluated = _g(o, "evaluated", None)
    if evaluated is None:
        evaluated = _num(_g(o, "score")) is not None
    return bool(evaluated) and _g(o, "error") is None and _g(o, "fail_class") == "ok"


def _valid_frac(o):
    nv, nt = _num(_g(o, "n_valid")), _num(_g(o, "n_total"))
    if nv is None or nt is None or nt <= 0:
        return None
    return nv / nt


def _is_soft(o):
    """Non-ok result that was still evaluated with fully valid output."""
    if _is_success(o) or not _g(o, "evaluated", False) or _num(_g(o, "score")) is None:
        return False
    if _g(o, "valid") is not True:
        return False
    frac = _valid_frac(o)
    return frac is None or frac >= 1.0


def _error_text(o):
    return ("%s %s" % (_g(o, "fail_class") or "", _g(o, "error") or "")).lower()


def _classify_failure(o):
    """'soft' | 'hard' | 'repairable' for a non-success observation."""
    if _is_soft(o):
        return "soft"
    fc = str(_g(o, "fail_class") or "").lower()
    text = _error_text(o)
    if fc in _HARD_CLASSES or any(p in text for p in _HARD_PATTERNS):
        return "hard"
    helper_hard = False
    if _sig_failed_hard is not None:
        try:
            helper_hard = bool(_sig_failed_hard(o))
        except Exception:
            helper_hard = False
    # The helper (like n_valid == 0) is only a signal: a recognisably
    # implementation-level error overrides it.
    if helper_hard and not any(p in text for p in _REPAIRABLE_PATTERNS):
        return "hard"
    return "repairable"


def _error_signature(o):
    text = str(_g(o, "error") or "").lower()
    text = re.sub(r"/[^\s:'\"]+", "<p>", text)
    text = re.sub(r"[0-9a-f]{6,}", "<h>", text)
    text = re.sub(r"\d+", "<n>", text)
    return "%s|%s" % (_g(o, "fail_class"), text[:200])


def _trajectories(prefix):
    """Ordered per-branch prefix trajectories with episode statistics."""
    by_branch = {}
    for cid, o in prefix.items():
        by_branch.setdefault(_g(o, "branch"), []).append((_g(o, "attempt", 0), cid, o))
    trajs = {}
    for br, cells in by_branch.items():
        cells.sort(key=lambda t: (t[0], t[1]))
        t = {
            "cells": cells, "depth": len(cells), "succ": [], "soft": [],
            "fail_w": 0.0, "hard": 0, "n_fail": 0, "latest": None,
            "repairs_failed": 0, "first": cells[0][2],
        }
        last_sig = None
        for _, _, o in cells:
            if _is_success(o):
                t["succ"].append(o)
                t["fail_w"], t["hard"], t["n_fail"], last_sig = 0.0, 0, 0, None
                t["latest"] = "success"
                continue
            kind = _classify_failure(o)
            t["latest"] = kind
            t["n_fail"] += 1
            if kind == "soft":
                t["soft"].append(o)
                t["fail_w"] += 0.5
                continue
            w = 1.0
            sig = _error_signature(o)
            if sig == last_sig:  # the repair attempt reproduced the same error
                w += 0.5
                t["repairs_failed"] += 1
            last_sig = sig
            t["fail_w"] += w
            if kind == "hard":
                t["hard"] += 1
        trajs[br] = t
    return trajs


class _Norm:
    """Prefix-relative score scale: baseline -> 0, best revealed -> ~1."""

    def __init__(self, prefix, baseline):
        self.b = _num(baseline) or 0.0
        vals = [_num(_g(o, "score")) for o in prefix.values() if _is_success(o) or _is_soft(o)]
        vals = [v for v in vals if v is not None]
        top = max(vals) if vals else self.b
        self.s = max(top - self.b, 0.1 * abs(self.b), 1e-12)

    def u(self, x):
        x = _num(x)
        if x is None:
            return -1.5
        return _clip((x - self.b) / self.s, -1.5, 1.5)

    def eff(self, o):
        v = self.u(_g(o, "score"))
        frac = _valid_frac(o)
        if frac is not None and 0.75 <= frac < 1.0:
            v = max(v, -0.3)  # near-valid success: correctness slip, not a dead direction
        return v


def _branch_ref(t, norm):
    """Branch reference value: successful anchor, else discounted soft evidence."""
    refs = [norm.eff(o) for o in t["succ"]]
    refs += [0.7 * norm.u(_g(o, "score")) for o in t["soft"]]
    return max(refs) if refs else None


def _progress(prefix):
    vals = [_num(_g(o, "score")) for o in prefix.values() if _is_success(o) or _is_soft(o)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def update_closed(closed, prefix, question, sched):
    """Recompute closures from cumulative evidence (a later success re-opens)."""
    closed.clear()
    norm = _Norm(prefix, question.baseline_score)
    for br, t in _trajectories(prefix).items():
        ref = _branch_ref(t, norm)
        strong = ref is not None and ref >= 1.0 - sched["recover_margin"]
        if t["latest"] == "hard" and t["hard"] >= sched["hard_close"] and (
            ref is None or t["hard"] > sched["hard_close"]
        ):
            closed.add(br)  # hard-unrecoverable episode
        elif t["latest"] in ("repairable", "hard") and t["fail_w"] >= sched["fail_close"] and not strong:
            closed.add(br)  # repeated failed repairs without a credible anchor
        elif t["latest"] == "success" and len(t["succ"]) >= sched["unpromising_n"]:
            effs = [norm.eff(o) for o in t["succ"]]
            if max(effs) < 1.0 - sched["margin"] and effs[-1] <= effs[-2]:
                closed.add(br)  # repeatedly unpromising after sufficient valid evidence


def _candidates(prefix, question, closed, sched):
    """Deterministically scored candidates: (priority, role, tiebreak, cell)."""
    norm = _Norm(prefix, question.baseline_score)
    trajs = _trajectories(prefix)
    roots = list(question.legal_roots())
    root_set = set(roots)

    firsts = []
    for t in trajs.values():
        o = t["first"]
        if _is_success(o):
            firsts.append(norm.eff(o))
        elif _is_soft(o):
            firsts.append(0.7 * norm.u(_g(o, "score")))
        else:
            firsts.append(-0.5)
    root_prior = sum(_clip(v, -0.5, 1.0) for v in firsts) / len(firsts) if firsts else 0.5

    def tiebreak(cid):
        m = question.meta(cid)
        return (_g(m, "attempt", 0), _g(m, "seq", 0), str(cid))

    seen_dirs = set()
    for t in trajs.values():
        cid0 = t["cells"][0][1]
        try:
            seen_dirs.add(str(_g(_g(question.meta(cid0), "tags", {}) or {}, "direction", "")))
        except Exception:
            pass

    out = []
    for cid in question.legal_actions():
        if cid in root_set:
            try:
                d = str(_g(_g(question.meta(cid), "tags", {}) or {}, "direction", ""))
            except Exception:
                d = ""
            div = 0.1 if d and d not in seen_dirs else 0.0
            out.append((root_prior + sched["explore_bonus"] + div, "explore", tiebreak(cid), cid))
            continue
        br = _g(question.meta(cid), "branch")
        if br in closed or br not in trajs:
            continue
        t = trajs[br]
        ref = _branch_ref(t, norm)
        under = t["depth"] <= sched["under_depth"] and len(t["succ"]) <= 1
        if t["latest"] in ("success", "soft"):
            last = t["cells"][-1][2]
            l_eff = norm.eff(last) if t["latest"] == "success" else 0.7 * norm.u(_g(last, "score"))
            p = 0.55 * (ref if ref is not None else l_eff) + 0.45 * l_eff
            effs = [norm.eff(o) for o in t["succ"]]
            if t["latest"] == "success" and len(effs) >= 2:
                gain = effs[-1] - effs[-2]
                if gain > 0:
                    p += 0.3 * min(gain, 1.0)
                else:
                    declines = 0
                    for i in range(len(effs) - 1, 0, -1):
                        if effs[i] <= effs[i - 1]:
                            declines += 1
                        else:
                            break
                    p -= sched["decline_pen"] * declines
            p -= sched["fail_pen"] * t["fail_w"]
            if under:
                p += sched["under_bonus"]
            out.append((p, "explore" if under else "exploit", tiebreak(cid), cid))
        else:
            base = ref if ref is not None else root_prior
            p = base - sched["fail_pen"] * t["fail_w"]
            if under:
                p += 0.5 * sched["under_bonus"]
            out.append((p, "recovery", tiebreak(cid), cid))
    out.sort(key=lambda c: (-c[0], c[2]))
    return out


def select_batch(prefix, question, closed, sched, stagnant, budget_left):
    W = max(1, int(question.max_parallelism))
    cap = W if budget_left is None else max(0, min(W, budget_left))
    if cap == 0:
        return []
    roots = list(question.legal_roots())
    if not question.opened_branches() and not prefix:
        n0 = max(1, int(math.ceil(W * sched["init_width_frac"])))
        cands = _candidates(prefix, question, closed, sched)
        return [c[3] for c in cands if c[3] in set(roots)][: min(cap, n0)]

    cands = _candidates(prefix, question, closed, sched)
    if not cands:
        return []
    top = cands[0][0]
    if stagnant:
        eligible = [c for c in cands if c[0] >= sched["hp_thr"]]
        wide = sched["hp_thr"] - 0.3
    else:
        eligible = [c for c in cands if c[0] >= top - sched["margin"]]
        wide = top - sched["wide_margin"]
    if not eligible:
        return []  # portfolio-level stop: nothing credible remains

    batch, n_roots, n_rec = [], 0, 0
    root_set = set(roots)

    def take(c):
        nonlocal n_roots, n_rec
        if c[3] in batch or len(batch) >= cap:
            return False
        if c[3] in root_set and n_roots >= sched["roots_per_round"]:
            return False
        if c[1] == "recovery":
            if n_rec >= 1 or c[0] < top - sched["recover_margin"]:
                return False
            n_rec += 1
        if c[3] in root_set:
            n_roots += 1
        batch.append(c[3])
        return True

    # Portfolio: one of each role first (exploit, exploration, recovery) ...
    for role in ("exploit", "explore", "recovery"):
        for c in eligible:
            if c[1] == role and take(c):
                break
    # ... then remaining workers by priority.
    for c in eligible:
        take(c)
    # Avoid a serial singleton when another credible candidate exists.
    if len(batch) == 1 and cap > 1:
        for c in cands:
            if c[3] not in batch and c[0] >= wide and take(c):
                break
    return batch


class OptimalPolicy(LLMDesignedMethod):
    NAME = NAME

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = getattr(self, "config", None) or {}
        self.beta = float(config.get("beta", DEFAULT_BETA))
        self.sched = _schedule(self.beta)

    def solve(self, question, budget=None):
        question.reset()
        res, closed = SimResult(), set()
        last_level, since = None, 0
        while not _budget_done(question, budget):
            prefix = question.observed()
            level = _progress(prefix)
            if level is not None:
                if last_level is None or level > last_level + 1e-12 * max(1.0, abs(level)):
                    last_level, since = level, 0
                else:
                    since += 1
            update_closed(closed, prefix, question, self.sched)
            budget_left = None if budget is None else int(budget) - len(prefix)
            batch = select_batch(
                prefix, question, closed, self.sched,
                stagnant=since >= self.sched["patience"], budget_left=budget_left,
            )
            if not batch:
                break
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(res, question))
        return finalize_result(question, res)

    def plan_grid(self, context):
        def ctx(*names, default=None):
            for n in names:
                v = _g(context, n)
                if v is not None:
                    return v
            return default

        fb_w = int(ctx("fallback_branch_count", default=4))
        fb_r = int(ctx("fallback_refine_count", default=3))
        max_w = int(ctx("hard_max_branch_count", default=max(fb_w, 1)))
        max_r = int(ctx("hard_max_refine_count", default=max(fb_r, 0)))
        sup_w = ctx("trace_branch_count")
        sup_r = ctx("trace_refine_count")

        def clamp(w, r):
            w, r = _clip(int(w), 1, max(1, max_w)), _clip(int(r), 0, max(0, max_r))
            if sup_w is not None:
                w = min(w, max(1, int(sup_w)))
            if sup_r is not None:
                r = min(r, max(0, int(sup_r)))
            return w, r

        hist = [h for h in (ctx("history", default=[]) or []) if h is not None][-3:]
        if hist:
            try:
                return self._plan_from_history(hist, fb_w, fb_r, clamp)
            except Exception:
                pass  # malformed history: fall through to the bootstrap plan
        w, r = clamp(fb_w, fb_r)
        return GridPlan(branch_count=w, refine_count=r,
                        reason="insufficient usable live history: conservative bootstrap from the fallback grid")

    def _plan_from_history(self, hist, fb_w, fb_r, clamp):
        def grid(h):
            g = _g(h, "effective_grid") or _g(h, "planned_grid") or {}
            return int(_g(g, "branch_count", fb_w) or fb_w), int(_g(g, "refine_count", fb_r) or fb_r)

        last = hist[-1]
        w0, r0 = grid(last)
        best_att = _num(_g(last, "best_attempt"))
        fails = _g(last, "fail_classes", {}) or {}
        n_fail = sum(int(v) for k, v in fails.items() if k != "ok")
        n_hard = sum(int(v) for k, v in fails.items() if str(k).lower() in _HARD_CLASSES)
        n_all = sum(int(v) for v in fails.values()) or 1
        bests = [_num(_g(h, "best_score")) for h in hist]
        plateau = len(bests) >= 2 and None not in bests[-2:] and bests[-1] <= bests[-2]
        tag = "%d live cycle(s)" % len(hist)

        if n_hard >= 0.5 * n_all:
            w, r = w0 - 1, r0 - 1
            why = "hard failures dominate (%d/%d probes): shrink width and depth" % (n_hard, n_all)
        elif best_att is not None and best_att >= r0 and not plateau:
            w, r = w0, r0 + 1
            why = ("best found at the last allowed refinement (attempt %d of %d), %d non-ok "
                   "probes, none dominated by hard failures: hold width, deepen by 1" % (best_att, r0, n_fail))
        elif plateau and best_att is not None and best_att < r0:
            w, r = w0 + 1, r0
            why = "best score plateaued and was reached before max depth: widen by 1"
        elif best_att is not None and best_att <= 1 and r0 > 2:
            w, r = w0 + 1, r0 - 1
            why = "gains arrived early (attempt %d): widen, trim depth" % best_att
        else:
            w, r = w0, r0
            why = "no clear width/depth signal: hold the last effective grid"
        cw, cr = clamp(w, r)
        note = "" if (cw, cr) == (w, r) else " (clamped to caps/trace support)"
        if len(hist) < 2:
            note += "; evidence limited to one cycle"
        return GridPlan(branch_count=cw, refine_count=cr, reason="%s: %s%s" % (tag, why, note))
