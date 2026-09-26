"""The runner's launch record: what a real-agent run's manifests do not carry, kept per launch.

scripts/report_run.py reads the caps and the program file from the last line of
<workdir>/launches.jsonl, and shows the agents' argv and versions and the host toolchain.
"""

import json
import os
import subprocess

from see.live import CommandAgent, TaskSpec
from see.loader import load_module_from_path

RECON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
runner = load_module_from_path(
    "run_dream_rsi_under_test", os.path.join(RECON, "scripts", "run_dream_rsi.py")
)
ARGS = [
    "--simpletes",
    "../SimpleTES",
    "--task",
    "lasso_path",
    "--workdir",
    "unused",
    "--discovery-agent",
    "gemini",
    "--policy-agent",
    "claude",
    "--grid",
    "4",
    "3",
    "--hard-max",
    "6",
    "4",
]


def _record(tmp_path, eigen_include=None):
    args = ARGS + (["--eigen-include", eigen_include] if eigen_include else [])
    a = runner.build_parser().parse_args(args)
    task = TaskSpec("lasso_path", str(tmp_path), "init_program.py", "p.txt", lambda p: {})
    agent = CommandAgent(["python3", "-c", "{prompt}"])
    return runner.launch_record(a, args, task, agent, agent)


def test_each_launch_records_the_caps_and_program_file_the_report_reads(tmp_path):
    r = _record(tmp_path)
    assert r["task"] == {"name": "lasso_path", "eval_program": "init_program.py"}
    assert (r["config"]["fallback_grid"], r["config"]["hard_max_grid"]) == ([4, 3], [6, 4])
    assert r["agents"]["discovery"]["argv"] == ["python3", "-c", "{prompt}"]
    assert r["agents"]["discovery"]["version"].startswith("Python 3")


def test_the_launch_record_carries_eval_repeats(tmp_path):
    """The report reads the repeats in force from the last launch; the default is the paper's
    single evaluation."""
    assert _record(tmp_path)["config"]["eval_repeats"] == 1
    a = runner.build_parser().parse_args([*ARGS, "--eval-repeats", "3"])
    task = TaskSpec("lasso_path", str(tmp_path), "init_program.py", "p.txt", lambda p: {})
    agent = CommandAgent(["python3", "-c", "{prompt}"])
    assert runner.launch_record(a, ARGS, task, agent, agent)["config"]["eval_repeats"] == 3


def test_an_env_wrapped_agent_reports_the_version_of_the_cli_it_wraps():
    """An isolated agent's argv starts with ``env HOME=...``: the version is the CLI's."""
    version = runner.cli_version(["env", "HOME=/nonexistent", "python3", "-c", "{prompt}"])
    assert version is not None and version.startswith("Python 3")


def test_an_env_argv_with_options_records_no_version_rather_than_the_environment():
    """``env -u SOMEVAR prog`` must not run ``env -u --version``, which prints the environment."""
    version = runner.cli_version(["env", "-u", "SOMEVAR", "python3", "-c", "pass"])
    assert version is None


def test_the_eigen_version_is_read_from_the_headers_the_run_compiles_with(tmp_path):
    util = tmp_path / "eigen3" / "Eigen" / "src" / "Core" / "util"
    util.mkdir(parents=True)
    (util / "Macros.h").write_text(
        "#define EIGEN_WORLD_VERSION 3\n"
        "#define EIGEN_MAJOR_VERSION 4\n"
        "#define EIGEN_MINOR_VERSION 1\n"
    )
    assert _record(tmp_path, str(tmp_path / "eigen3"))["host"]["eigen"] == "3.4.1"
    assert runner.eigen_version(str(tmp_path)) is None  # no headers there: say so, do not guess


def test_the_host_facts_name_the_simpletes_checkout_commit(tmp_path):
    """This repository stands in for a SimpleTES clone; a directory outside any clone has none."""
    head = subprocess.run(
        ["git", "-C", RECON, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert runner.host_facts(RECON, None)["simpletes_commit"] == head
    assert runner.host_facts(str(tmp_path), None)["simpletes_commit"] is None


def test_a_restart_appends_its_launch_and_keeps_the_first(tmp_path):
    runner.record_launch(str(tmp_path), {"launch": 1})
    runner.record_launch(str(tmp_path), {"launch": 2})
    lines = (tmp_path / "launches.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == [{"launch": 1}, {"launch": 2}]
