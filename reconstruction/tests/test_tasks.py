"""Pins for the SimpleTES adapter's host hook (GAPS.md §3, "Lasso host toolchain").

A fake SimpleTES task stands in for the AGPL checkout: its evaluator reports the directory the
real one would try first for Eigen (``<its own dir>/eigen``), so no compiler is involved.
"""

import os

import pytest

from see.loader import load_module_from_path
from see.tasks import link_eigen, simpletes_task

VERIFY_LASSO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "verify_lasso.py"
)
FAKE_EVALUATOR = """import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(path):
    eigen = os.path.join(TASK_DIR, "eigen")  # the real evaluator's first -I candidate
    found = os.path.realpath(eigen) if os.path.isdir(eigen) else None
    return {"combined_score": 0.0, "eigen": found}
"""


def _fake_simpletes(root) -> str:
    task = root / "datasets" / "numerical_tasks" / "lasso_path"
    (task / "eigen" / "Eigen").mkdir(parents=True)  # the vendored tree, which lacks Eigen/Core
    (task / "evaluator.py").write_text(FAKE_EVALUATOR)
    (task / "init_program.py").write_text('CPP_CODE = ""\n')
    (task / "lasso_path.txt").write_text("Compute the Lasso path.\n")
    return str(root)


def _host_eigen(root) -> str:
    (root / "Eigen").mkdir(parents=True)
    return str(root)


def test_without_eigen_include_the_evaluator_falls_back_to_the_system_headers(tmp_path):
    """The vendored tree is not copied, so nothing sits where the evaluator looks first."""
    task = simpletes_task(_fake_simpletes(tmp_path / "st"), "lasso_path", str(tmp_path / "w"))
    assert task.evaluate("unused")["eigen"] is None


def test_eigen_include_is_where_the_evaluator_looks_first(tmp_path):
    host = _host_eigen(tmp_path / "host" / "eigen3")
    task = simpletes_task(
        _fake_simpletes(tmp_path / "st"), "lasso_path", str(tmp_path / "w"), eigen_include=host
    )
    assert task.evaluate("unused")["eigen"] == os.path.realpath(host)


def test_a_restart_reuses_its_eigen_link_and_refuses_a_different_one(tmp_path):
    simpletes, workdir = _fake_simpletes(tmp_path / "st"), str(tmp_path / "w")
    first, other = _host_eigen(tmp_path / "a"), _host_eigen(tmp_path / "b")
    simpletes_task(simpletes, "lasso_path", workdir, eigen_include=first)
    again = simpletes_task(simpletes, "lasso_path", workdir, eigen_include=first)  # the restart
    assert again.evaluate("unused")["eigen"] == os.path.realpath(first)
    with pytest.raises(FileExistsError, match="does not point at"):
        simpletes_task(simpletes, "lasso_path", workdir, eigen_include=other)


def test_an_eigen_root_without_eigen_headers_is_refused_before_any_run(tmp_path):
    """/opt/local/include (one level too high) would compile nothing; say so before the run."""
    (tmp_path / "include").mkdir()
    with pytest.raises(FileNotFoundError, match="has no Eigen/ directory"):
        link_eigen(str(tmp_path / "task"), str(tmp_path / "include"))


def test_verify_lasso_points_its_evaluator_copy_at_the_host_eigen(tmp_path):
    verify_lasso = load_module_from_path("verify_lasso_under_test", VERIFY_LASSO)
    host = _host_eigen(tmp_path / "eigen3")
    (tmp_path / "copy").mkdir()
    ev = verify_lasso.load_evaluator(_fake_simpletes(tmp_path / "st"), str(tmp_path / "copy"), host)
    assert ev.evaluate("unused")["eigen"] == os.path.realpath(host)
