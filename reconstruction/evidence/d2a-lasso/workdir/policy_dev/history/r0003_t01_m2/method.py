"""OptimalPolicy: trajectory-ranked, full-batch, beta-thresholded explorer.

Prefix signals (all from ``question.observed()``, ``baseline_score``, legal sets
and structural ``meta``):
  * Per-branch ordered trajectory: successful anchor (best score of an evaluated,
    ``error is None``, ``fail_class == "ok"`` result, regardless of ``valid``),
    the ordered sequence of valued results (parent->child gains, consecutive
    regressions), the current failure episode (weighted consecutive failures
    since the last success, hard count, repeated identical error = failed
    repair), whether an earlier repair succeeded, and explored depth.
  * "Soft" evidence: a non-ok result that was still evaluated with fully valid
    output (e.g. an agent timeout whose candidate scored) counts as a discounted
    anchor and half a failure. It is never counted as "beating the baseline".
  * Scores are normalised relative to the prefix only: u = (x - baseline) / s,
    s = max(top - baseline, 0.1 |baseline|), so the best revealed result is ~1
    and the baseline is 0. Near-valid successes (>= 75% valid outputs but a
    collapsed score) are floored at -0.3: a correctness slip, not a dead branch.

Failure interpretation: hard-unrecoverable only for environment / dependency
signals (fail_class or error text). ``branch_failed_hard`` and ``n_valid == 0``
are signals, never closure on their own; compile_other, timeouts, mismatch,
shared-memory/resource and shape/mask/layout/name errors are repairable. A later
success resets the failure episode and re-opens the branch.

Batch rule (each decision round): every legal non-closed action gets one
deterministic priority (exploit = latest success, explore = unopened root or
under-explored branch, recovery = latest repairable failure). Eligibility is a
*prefix-normalised* bar ``thr(beta)`` (0 = baseline, 1 = best revealed), not a
band below the current top, so the policy stops spending probes when only weak
actions remain instead of always chasing the relatively-best leftover. The
batch takes the best exploit, the best exploration, at most one justified
recovery, then fills every remaining worker from the eligible set by priority;
a lone candidate is widened with the next credible one. Round 1 opens a full
batch of roots. Until some *successful* result beats the baseline, all
non-closed actions are eligible (nothing has been attained yet, so stopping
cannot be justified). One frontier per branch, legal cells only, so a batch
never holds a parent and its child.

Beta schedule (``_schedule``): higher beta -> lower eligibility bar, more
roots per round, longer stagnation patience, softer failure / decline
penalties, later closure; lower beta -> skip weak/declining frontiers and
repeated failures, stop earlier on stagnation. Beta is fixed within an episode.

Stop rule (portfolio level): stop only when no legal non-closed candidate of
any role (active refinements, recoveries, under-explored branches, unopened
roots) clears the bar; after ``patience`` rounds without a new best success the
bar rises to ``stag_thr``.

Default beta 0.6: the only live cycle (iter0001) ran the beta-free baseline
(beta=null), so there is no live beta trend; the r0002 sweep only shows that
beta >= 0.3 reached the replay ceiling. History is insufficient, so the
moderately exploratory default is kept.

Grid planning (``plan_grid``): uses only completed live manifests. Best found
at the last allowed refinement without plateau -> hold width, deepen by 1;
plateau with best found before max depth -> widen by 1; gains only early ->
widen, trim depth; dominant hard failures -> shrink both; otherwise hold.
Empty history -> the runner's fallback grid as an explicit bootstrap. Always
clamped to hard caps and, in replay, to the trace support.

Safeguards: no absolute score cutoffs; shallow weak branches get an
under-explored bonus; a repairable failure keeps its branch's anchor as its
recovery value (only a per-failure penalty), so it is not starved; closure needs
cumulative evidence; batches fill workers whenever credible candidates exist.
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
        "thr": -0.35 - 1.2 * b,               # eligibility bar (0 = baseline, 1 = best)
        "stag_thr": 0.6 - 0.9 * b,            # raised bar once progress stagnates
        "patience": 1 + int(round(3 * b)),    # rounds without a new best before stagnation
        "roots_frac": 0.25 + 0.75 * b,        # share of workers that may open roots per round
        "recover_margin": 0.6 + 0.8 * b,      # recovery may take a slot within this of the top
        "wide_band": 0.25 + 0.5 * b,          # band below the bar for widening a singleton
        "fail_pen": 0.35 - 0.2 * b,           # per weighted consecutive failure
        "repair_bonus": 0.15,                 # branch already recovered from a failure once
        "fail_close": 2.0 + 3.0 * b,          # weighted failures to close a hopeless branch
        "hard_close": 1 if b < 0.5 else 2,    # hard failures in an episode to close
        "decline_pen": 0.2 - 0.1 * b,         # per consecutive regression
        "unpromising_n": 3 + int(round(2 * b)),  # successes needed to call a branch unpromising
        "unpromising_max": 0.3 - 0.6 * b,     # ... whose best normalised score stays below this
        "under_bonus": 0.3 + 0.4 * b,         # weak-but-under-explored protection
        "under_depth": 1 if b < 0.5 else 2,
        "explore_bonus": 0.2 + 0.6 * b,       # value of an unopened root over the root prior
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
            "repairs_failed": 0, "repaired": 0, "first": cells[0][2],
        }
        last_sig = None
        for _, _, o in cells:
            if _is_success(o):
                if t["latest"] in ("repairable", "hard"):
                    t["repaired"] += 1  # a success right after a failure
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

    def value(self, o):
        """Valued evidence of one cell: success eff, discounted soft, else None."""
        if _is_success(o):
            return self.eff(o)
        if _is_soft(o):
            return 0.7 * self.u(_g(o, "score"))
        return None


def _branch_ref(t, norm):
    """Branch reference value: successful anchor, else discounted soft evidence."""
    refs = [norm.value(o) for o in t["succ"] + t["soft"]]
    return max(refs) if refs else None


def _best_success(prefix):
    vals = [_num(_g(o, "score")) for o in prefix.values() if _is_success(o)]
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
            if max(effs) < sched["unpromising_max"] and effs[-1] <= effs[-2]:
                closed.add(br)  # repeatedly unpromising after sufficient valid evidence


def _candidates(prefix, question, closed, sched):
    """Deterministically scored candidates: (priority, role, tiebreak, cell)."""
    norm = _Norm(prefix, question.baseline_score)
    trajs = _trajectories(prefix)
    root_set = set(question.legal_roots())

    firsts = []
    for t in trajs.values():
        v = norm.value(t["first"])
        firsts.append(-0.5 if v is None else _clip(v, -0.5, 1.0))
    root_prior = sum(firsts) / len(firsts) if firsts else 0.5

    def tiebreak(cid):
        m = question.meta(cid)
        return (_g(m, "attempt", 0), _g(m, "seq", 0), str(cid))

    def direction(cid):
        try:
            return str(_g(_g(question.meta(cid), "tags", {}) or {}, "direction", "") or "")
        except Exception:
            return ""

    seen_dirs = {direction(t["cells"][0][1]) for t in trajs.values()}

    out = []
    for cid in question.legal_actions():
        if cid in root_set:
            d = direction(cid)
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
            vals = [v for v in (norm.value(o) for _, _, o in t["cells"]) if v is not None]
            last = vals[-1]
            p = 0.5 * (ref if ref is not None else last) + 0.5 * last
            if len(vals) >= 2:
                gain = vals[-1] - vals[-2]
                if gain > 0:
                    p += 0.3 * min(gain, 1.0)  # parent->child improvement
                else:
                    declines = 0
                    for i in range(len(vals) - 1, 0, -1):
                        if vals[i] <= vals[i - 1]:
                            declines += 1
                        else:
                            break
                    p -= sched["decline_pen"] * declines
            if len(t["cells"]) >= 2 and _classify_or_ok(t["cells"][-2][2]) in ("repairable", "hard") \
                    and t["latest"] == "success":
                p += sched["repair_bonus"]  # a repair just succeeded
            p -= sched["fail_pen"] * t["fail_w"]
            if under:
                p += sched["under_bonus"]
            out.append((p, "explore" if under else "exploit", tiebreak(cid), cid))
        else:
            # Recovery keeps the branch's historical anchor; only the current
            # failure episode is penalised, softened by earlier repair successes.
            base = ref if ref is not None else root_prior
            p = base - sched["fail_pen"] * t["fail_w"]
            if t["repaired"]:
                p += sched["repair_bonus"]
            if under:
                p += 0.5 * sched["under_bonus"]
            out.append((p, "recovery", tiebreak(cid), cid))
    out.sort(key=lambda c: (-c[0], c[2]))
    return out


def _classify_or_ok(o):
    return "success" if _is_success(o) else _classify_failure(o)


def select_batch(prefix, question, closed, sched, stagnant, budget_left):
    W = max(1, int(question.max_parallelism))
    cap = W if budget_left is None else max(0, min(W, budget_left))
    if cap == 0:
        return []
    cands = _candidates(prefix, question, closed, sched)
    if not cands:
        return []
    root_set = set(question.legal_roots())
    if not prefix:
        # Round 1: a full parallel batch of roots (nothing to rank on yet).
        return [c[3] for c in cands if c[3] in root_set][:cap]

    best = _best_success(prefix)
    base = _num(question.baseline_score) or 0.0
    if best is None or best <= base:
        thr = -math.inf  # nothing attained yet: every non-closed action stays eligible
    else:
        thr = max(sched["thr"], sched["stag_thr"]) if stagnant else sched["thr"]
    eligible = [c for c in cands if c[0] >= thr]
    if not eligible:
        return []  # portfolio-level stop: no candidate of any role clears the bar

    top = eligible[0][0]
    roots_cap = max(1, int(round(W * sched["roots_frac"])))
    batch, n_roots, n_rec = [], 0, 0

    def take(c):
        nonlocal n_roots, n_rec
        if c[3] in batch or len(batch) >= cap:
            return False
        if c[3] in root_set and n_roots >= roots_cap:
            return False
        if c[1] == "recovery":
            if n_rec >= 1:
                return False
            n_rec += 1
        if c[3] in root_set:
            n_roots += 1
        batch.append(c[3])
        return True

    # Portfolio: best exploit and best exploration first ...
    for role in ("exploit", "explore"):
        for c in eligible:
            if c[1] == role and take(c):
                break
    # ... one recovery if justified (it must not displace stronger refinements) ...
    n_normal = sum(1 for c in eligible if c[1] != "recovery")
    for c in eligible:
        if c[1] == "recovery":
            if n_normal < cap or c[0] >= top - sched["recover_margin"]:
                take(c)
            break
    # ... then every remaining worker by priority.
    for c in eligible:
        if c[1] != "recovery":
            take(c)
    # Avoid a serial singleton when another credible candidate exists.
    if len(batch) == 1 and cap > 1:
        for c in cands:
            if c[3] not in batch and c[0] >= thr - sched["wide_band"] and take(c):
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
            level = _best_success(prefix)
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
