"""Signal helpers named in Listing 2 (lines 43-45), plus the failure taxonomy.

The paper names only ``"ok"`` and ``"compile_other"`` as fail classes; the
rest of the taxonomy is inferred from the failure kinds Listing 2 lists as
"normally repairable" (lines 65-67) versus environment/dependency failures.
"""

from __future__ import annotations

import re

OK = "ok"
REPAIRABLE_FAIL_CLASSES = frozenset(
    {
        "correctness",  # output/correctness mismatch
        "resource",  # shared-memory / resource limits, OOM
        "code",  # variable/code errors
        "shape",  # mask/layout/shape errors
        "compile_other",  # "compile_other alone is not permanently hard"
        "timeout",
        "no_program",  # agent wrote nothing evaluable
    }
)
HARD_FAIL_CLASSES = frozenset({"env"})  # environment/dependency failure
FAIL_CLASSES = frozenset({OK}) | REPAIRABLE_FAIL_CLASSES | HARD_FAIL_CLASSES

# First match wins; order matters (a compile error can mention a missing header).
_RULES = [
    (
        "env",
        r"No module named|ModuleNotFoundError|command not found|CUDA driver|"
        r"no CUDA-capable|cannot open shared object|Permission denied",
    ),
    ("timeout", r"[Tt]imed? ?out|TimeoutExpired|deadline exceeded"),
    (
        "resource",
        r"[Mm]emory|shared mem|out of resources|OOM|too many resources|"
        r"exceeds .*limit|Killed",
    ),
    (
        "compile_other",
        r"[Cc]ompil(e|ation) (failed|error)|error: |undefined reference|"
        r"ld returned",
    ),
    (
        "shape",
        r"[Ss]hape mismatch|size mismatch|[Oo]utput size|layout|mask|"
        r"dimension|broadcast",
    ),
    (
        "correctness",
        r"[Cc]orrectness|mismatch|max_gap|not close|allclose|"
        r"[Ww]rong (answer|result)|[Ii]ncorrect|[Vv]alidation failed",
    ),
    (
        "code",
        r"NameError|AttributeError|TypeError|ValueError|IndexError|KeyError|"
        r"SyntaxError|ZeroDivisionError|Traceback|[Ee]xception",
    ),
]


def classify_failure(error: str | None) -> str:
    """Map an evaluator/agent error string to a fail class (heuristic)."""
    if not error:
        return OK
    for fail_class, pattern in _RULES:
        if re.search(pattern, error):
            return fail_class
    return "code"


def is_success(obs) -> bool:
    """Listing 2 lines 47-49: evaluated, no error, fail_class "ok"; ``valid`` is irrelevant."""
    return bool(obs.evaluated) and obs.error is None and obs.fail_class == OK


def probe_improved_vs_parent(obs) -> bool:
    return is_success(obs) and obs.delta_vs_parent is not None and obs.delta_vs_parent > 0


def probe_improved_vs_baseline(obs) -> bool:
    return is_success(obs) and obs.delta_vs_baseline is not None and obs.delta_vs_baseline > 0


def branch_promising(obs) -> bool:
    """A successful probe that beat its parent or the baseline."""
    return probe_improved_vs_parent(obs) or probe_improved_vs_baseline(obs)


def branch_failed_hard(obs) -> bool:
    """A failed probe with zero valid instances or a hard fail class.

    Listing 2 calls this a signal, not unconditional closure: a zero-valid
    ``compile_other`` failure trips it but is still repairable.
    """
    if is_success(obs):
        return False
    return obs.fail_class in HARD_FAIL_CLASSES or obs.n_valid == 0


def is_repairable_failure(obs) -> bool:
    return not is_success(obs) and obs.fail_class in REPAIRABLE_FAIL_CLASSES
