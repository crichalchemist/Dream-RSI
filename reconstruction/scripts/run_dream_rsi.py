"""Run the reconstructed Dream-RSI loop on a paper task with real coding agents.

    python scripts/run_dream_rsi.py --simpletes ../SimpleTES --task lasso_path \
        --workdir runs/lasso --discovery-agent gemini --policy-agent gemini

Agents are CLI presets from see.live.AGENT_PRESETS ("gemini", "claude") or a
JSON argv list containing "{prompt}". Defaults mirror the paper's
Gemini-3.1-Pro setting (10 workers, 10 branches x 11 attempts, 5 rounds);
M, K1, K2, lambda and the beta grid are not given in the paper.

Every launch appends one line to <workdir>/launches.jsonl before the first
iteration: the caps and program file the manifests do not carry, each agent's
argv and CLI version, and the host toolchain (scripts/report_run.py reads it).
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

from see.live import CommandAgent, TaskSpec
from see.loop import DreamRSI, LoopConfig, install_signal_handlers
from see.objective import OBJECTIVES
from see.tasks import SIMPLETES_TASKS, simpletes_task


def agent(spec: str, timeout: float) -> CommandAgent:
    return CommandAgent(json.loads(spec) if spec.startswith("[") else spec, timeout=timeout)


def first_line(cmd: list) -> str | None:
    """The first line a command prints, or None when it cannot run or exits non-zero."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = (p.stdout or p.stderr).strip().splitlines()
    return lines[0] if p.returncode == 0 and lines else None


def cli_version(argv: list) -> str | None:
    """``<program> --version`` for an agent argv, under the ``env K=V ...`` prefix it may carry."""
    i = 0
    if argv[0] == "env":
        i = 1
        while i < len(argv) and "=" in argv[i] and not argv[i].startswith("-"):
            i += 1
        if i < len(argv) and argv[i].startswith("-"):
            return None  # an env option (e.g. -u) before the program, not the program itself
    return first_line([*argv[: i + 1], "--version"])


def eigen_version(include_dir: str) -> str | None:
    """``3.4.1`` from ``<include_dir>/Eigen/src/Core/util/Macros.h``, or None without one."""
    path = os.path.join(include_dir, "Eigen", "src", "Core", "util", "Macros.h")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        found = dict(re.findall(r"#define EIGEN_(WORLD|MAJOR|MINOR)_VERSION (\d+)", f.read()))
    return f"{found['WORLD']}.{found['MAJOR']}.{found['MINOR']}" if len(found) == 3 else None


def host_facts(simpletes: str, eigen_include: str | None) -> dict:
    """The toolchain a Lasso evaluation compiles with, as this process would find it."""
    return {
        "platform": platform.platform(),
        "cpu": first_line(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "compiler": first_line(["g++", "--version"]),
        "eigen": eigen_version(eigen_include or "/usr/include/eigen3"),
        "simpletes_commit": first_line(["git", "-C", simpletes, "rev-parse", "HEAD"]),
    }


def launch_record(a, argv: list, task: TaskSpec, discovery, policy) -> dict:
    return {
        "started": time.time(),
        "argv": argv,
        "task": {"name": a.task, "eval_program": task.eval_program},
        "config": {
            "iterations": a.iterations,
            "versions": a.versions,
            "max_parallelism": a.workers,
            "fallback_grid": list(a.grid),
            "hard_max_grid": list(a.hard_max),
            "objective": a.objective,
            "agent_timeout": a.agent_timeout,
            "eval_repeats": a.eval_repeats,
        },
        "agents": {
            "discovery": {"argv": discovery.argv, "version": cli_version(discovery.argv)},
            "policy": {"argv": policy.argv, "version": cli_version(policy.argv)},
        },
        "host": host_facts(a.simpletes, a.eigen_include),
    }


def record_launch(workdir: str, record: dict) -> None:
    """Append, never overwrite: a restart keeps the first launch's record."""
    with open(os.path.join(workdir, "launches.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("--simpletes", required=True)
    ap.add_argument("--task", required=True, choices=sorted(SIMPLETES_TASKS))
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--discovery-agent", required=True)
    ap.add_argument("--policy-agent", required=True)
    ap.add_argument("--agent-timeout", type=float, default=3600)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--versions", type=int, default=4, help="M")
    ap.add_argument("--workers", type=int, default=10, help="W")
    ap.add_argument("--grid", type=int, nargs=2, default=(10, 10), help="fallback W R")
    ap.add_argument("--hard-max", type=int, nargs=2, default=(32, 19))
    ap.add_argument("--objective", choices=OBJECTIVES, default="pareto")
    ap.add_argument(
        "--eval-repeats",
        type=int,
        default=1,
        help="evaluations per program, odd; the median run is kept (default: 1, as the paper)",
    )
    ap.add_argument(
        "--eigen-include",
        help="directory holding Eigen/ for the Lasso evaluator, e.g. /opt/local/include/eigen3 "
        "(default: the evaluator's /usr/include/eigen3)",
    )
    return ap


def main(argv=None):
    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
    a = build_parser().parse_args(argv)
    os.makedirs(a.workdir, exist_ok=True)
    cfg = LoopConfig(
        workdir=a.workdir,
        iterations=a.iterations,
        versions=a.versions,
        max_parallelism=a.workers,
        fallback_grid=tuple(a.grid),
        hard_max_grid=tuple(a.hard_max),
        objective=a.objective,
        eval_repeats=a.eval_repeats,
    )
    task = simpletes_task(a.simpletes, a.task, a.workdir, eigen_include=a.eigen_include)
    discovery = agent(a.discovery_agent, a.agent_timeout)
    policy = agent(a.policy_agent, a.agent_timeout)
    loop = DreamRSI(cfg, task, discovery, policy)
    loop.check_iteration(loop.state["iteration"] + 1)  # a refused restart records no launch
    record_launch(
        a.workdir,
        launch_record(a, sys.argv if argv is None else list(argv), task, discovery, policy),
    )
    state = loop.run()
    print(json.dumps(state["log"][-1], indent=1, default=str))


if __name__ == "__main__":
    main()
