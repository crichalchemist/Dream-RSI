"""Run the reconstructed Dream-RSI loop on a paper task with real coding agents.

    python scripts/run_dream_rsi.py --simpletes ../SimpleTES --task lasso_path \
        --workdir runs/lasso --discovery-agent gemini --policy-agent gemini

Agents are CLI presets from see.live.AGENT_PRESETS ("gemini", "claude") or a
JSON argv list containing "{prompt}". Defaults mirror the paper's
Gemini-3.1-Pro setting (10 workers, 10 branches x 11 attempts, 5 rounds);
M, K1, K2, lambda and the beta grid are not given in the paper.
"""

import argparse
import json
import os

from see.live import CommandAgent
from see.loop import DreamRSI, LoopConfig, install_signal_handlers
from see.tasks import SIMPLETES_TASKS, simpletes_task


def agent(spec: str, timeout: float) -> CommandAgent:
    return CommandAgent(json.loads(spec) if spec.startswith("[") else spec, timeout=timeout)


def main(argv=None):
    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
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
    ap.add_argument("--objective", choices=("pareto", "eq1"), default="pareto")
    a = ap.parse_args(argv)
    os.makedirs(a.workdir, exist_ok=True)
    cfg = LoopConfig(
        workdir=a.workdir,
        iterations=a.iterations,
        versions=a.versions,
        max_parallelism=a.workers,
        fallback_grid=tuple(a.grid),
        hard_max_grid=tuple(a.hard_max),
        objective=a.objective,
    )
    loop = DreamRSI(
        cfg,
        simpletes_task(a.simpletes, a.task, a.workdir),
        agent(a.discovery_agent, a.agent_timeout),
        agent(a.policy_agent, a.agent_timeout),
    )
    state = loop.run()
    print(json.dumps(state["log"][-1], indent=1, default=str))


if __name__ == "__main__":
    main()
