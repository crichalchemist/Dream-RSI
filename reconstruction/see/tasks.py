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


def link_eigen(task_dir: str, eigen_include: str) -> None:
    """Point SimpleTES's first Eigen lookup, ``<task dir>/eigen``, at a host Eigen root.

    The Lasso evaluator compiles with ``-I<its own dir>/eigen`` when that directory exists and
    with ``-I/usr/include/eigen3``, a Linux path, otherwise. ``eigen_include`` is the directory
    that holds ``Eigen/`` (``/opt/local/include/eigen3`` under MacPorts). A restart finds the
    link it made; a link to anything else is refused.
    """
    target = os.path.abspath(eigen_include)
    if not os.path.isdir(os.path.join(target, "Eigen")):
        raise FileNotFoundError(f"{target} has no Eigen/ directory: pass the root that holds it")
    link = os.path.join(task_dir, "eigen")
    if os.path.lexists(link):
        if os.path.realpath(link) != os.path.realpath(target):
            raise FileExistsError(f"{link} exists and does not point at {target}")
        return
    os.symlink(target, link)


def simpletes_task(
    simpletes_dir: str, name: str, workdir: str, *, eigen_include: str | None = None
) -> TaskSpec:
    src = os.path.join(simpletes_dir, SIMPLETES_TASKS.get(name, name))
    local = os.path.join(workdir, "task", "src")
    if not os.path.exists(local):
        # SimpleTES's vendored Eigen lacks Eigen/Core, so it is not copied: the Lasso
        # evaluator then uses eigen_include when given, else the system headers (libeigen3-dev).
        shutil.copytree(src, local, ignore=shutil.ignore_patterns("eigen", "__pycache__"))
    if eigen_include is not None:
        link_eigen(local, eigen_include)
    base = os.path.join(workdir, "task", "baseline")
    os.makedirs(base, exist_ok=True)
    shutil.copy(os.path.join(local, "init_program.py"), os.path.join(base, "init_program.py"))
    statement = sorted(glob.glob(os.path.join(local, "*.txt")))[0]
    os.environ.setdefault("EVALUATOR_CONCURRENT_PROCESSES", "1")  # evaluations are serialised
    mod = load_module_from_path(f"simpletes_{name}_evaluator", os.path.join(local, "evaluator.py"))
    return TaskSpec(name, base, "init_program.py", statement, mod.evaluate)
