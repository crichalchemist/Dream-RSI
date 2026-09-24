"""Synthetic discovery trees for tests and demos. Not calibrated to any real task.

Each branch has a latent ceiling and speed: few branches are strong, some
improve early and some late, and scores are noisy with occasional
regressions. Failures are mostly repairable; an environment failure ends a
branch. Some chains are truncated, as a live policy that stops refining a
branch would leave them, which makes the grid irregular. Cells are numbered
in the order a parallel-refine live run would create them.
"""
import math
import random

from see.world import Cell, Trace


def synthetic_trace(seed: int, branches: int = 10, refine: int = 10, baseline: float = 1.0,
                    max_parallelism: int = 10, fail_rate: float = 0.15,
                    env_rate: float = 0.02) -> Trace:
    rng = random.Random(seed)
    ceiling = [baseline * (1.0 + 2.0 * rng.betavariate(1.5, 5.0)) for _ in range(branches)]
    speed = [rng.uniform(0.08, 0.7) for _ in range(branches)]
    length = [refine + 1 if rng.random() < 0.6 else rng.randint(2, refine + 1)
              for _ in range(branches)]
    ended = [False] * branches
    cells, seq = [], 0
    for a in range(refine + 1):
        for b in range(branches):
            if ended[b] or a >= length[b]:
                continue
            r = rng.random()
            if r < env_rate:
                cell = Cell(b, a, seq, 0.0, evaluated=False, valid=False, fail_class="env",
                            error="ModuleNotFoundError: No module named 'triton'")
                ended[b] = True
            elif r < env_rate + fail_rate:
                kind = rng.choice(["correctness", "compile_other", "shape", "resource", "timeout"])
                cell = Cell(b, a, seq, 0.0, evaluated=kind != "compile_other", valid=False,
                            fail_class=kind, error=f"synthetic {kind} failure",
                            n_valid=0, n_total=17)
            else:
                progress = 1.0 - math.exp(-(a + 1) * speed[b])
                score = baseline + (ceiling[b] - baseline) * progress
                score *= 1.0 + rng.gauss(0.0, 0.03)
                if rng.random() < 0.1:
                    score *= 0.9  # a regression
                cell = Cell(b, a, seq, round(score, 6), n_valid=17, n_total=17)
            cells.append(cell)
            seq += 1
    return Trace(cells, baseline, max_parallelism, trace_id=f"synthetic-{seed}",
                 grid=(branches, refine), info={"generator": "see.synthetic", "seed": seed})
