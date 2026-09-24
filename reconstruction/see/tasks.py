"""Paper tasks backed by SimpleTES's public task suite (AGPL-3.0, not vendored).

Dream-RSI's Lasso task reuses SimpleTES's benchmark (paper Sec. 4.1) and its
three math tasks are ones SimpleTES also publishes. Each SimpleTES task dir
holds init_program.py, evaluator.py (``evaluate(path)``, larger
``combined_score`` is better; autocorrelation already reports 1 / C1) and a
statement .txt. The four KernelBench tasks need KernelBench and a GPU and
have no adapter here.
"""

import glob
import os
import shutil

from see.live import TaskSpec
from see.loader import load_module_from_path

SIMPLETES_TASKS = {
    "lasso_path": "datasets/numerical_tasks/lasso_path",
    "sum_difference": "datasets/sums_diffs/sums_diffs",
    "circle_packing_26": "datasets/circle_packing/circle_packing_26",
    "autocorrelation_first": "datasets/autocorrelation/autocorrelation_first",
}


def simpletes_task(simpletes_dir: str, name: str, workdir: str) -> TaskSpec:
    src = os.path.join(simpletes_dir, SIMPLETES_TASKS.get(name, name))
    local = os.path.join(workdir, "task", "src")
    if not os.path.exists(local):
        # SimpleTES's vendored Eigen lacks Eigen/Core; without the copy the Lasso
        # evaluator falls back to the system headers (libeigen3-dev).
        shutil.copytree(src, local, ignore=shutil.ignore_patterns("eigen", "__pycache__"))
    base = os.path.join(workdir, "task", "baseline")
    os.makedirs(base, exist_ok=True)
    shutil.copy(os.path.join(local, "init_program.py"), os.path.join(base, "init_program.py"))
    statement = sorted(glob.glob(os.path.join(local, "*.txt")))[0]
    os.environ.setdefault("EVALUATOR_CONCURRENT_PROCESSES", "1")  # evaluations are serialised
    mod = load_module_from_path(f"simpletes_{name}_evaluator", os.path.join(local, "evaluator.py"))
    return TaskSpec(name, base, "init_program.py", statement, mod.evaluate)
