"""Command line: ``python -m see sweep|demo ...`` (run from reconstruction/)."""

import argparse
import hashlib
import json
import os
import sys

from see.objective import DEFAULT_BETAS, DEFAULT_LAMBDA, beta_sweep, next_live_plan


def cmd_sweep(a):
    from see.loader import load_policy, overrides_plan_grid
    from see.pool import context_factory, load_pool, next_context

    with open(a.method, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()  # before loading runs the code
    cls = load_policy(a.method)
    pool = load_pool(a.pool)
    if not pool:
        sys.exit(f"no traces under {a.pool}")
    report, executions = beta_sweep(
        cls,
        [t for t, _ in pool],
        context_for=context_factory(pool, a.fallback, a.hard_max),
        betas=a.betas,
        lam=a.lam,
        beta1=a.beta1,
        beta2=a.beta2,
        max_rounds=a.max_rounds,
    )
    # after the sweep is scored, so it cannot change a score (GAPS §3, "Next live plan")
    width = a.max_parallelism if a.max_parallelism is not None else pool[-1][0].max_parallelism
    report["next_live_plan"] = next_live_plan(
        cls, next_context(pool, a.fallback, a.hard_max, width), [t.grid for t, _ in pool]
    )
    report["plan_grid_override"] = overrides_plan_grid(cls)
    report["method"] = os.path.abspath(a.method)
    report["sha256"] = digest
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "beta_sweep.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.out, "policy_execution_traces.jsonl"), "w") as f:
        for e in executions:
            f.write(json.dumps(e) + "\n")
    p = report["pareto"]
    print(
        json.dumps(
            {
                "policy": report["policy"],
                "valid": report["valid"],
                "reward": p["reward"],
                "auc": p["auc"],
                "parallel_penalty": p["parallel_penalty"],
                "eq1_V": report["eq1"]["V"],
            }
        )
    )


def cmd_demo(a):
    from see.loop import DreamRSI, LoopConfig, install_signal_handlers
    from see.toy import ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
    cfg = LoopConfig(
        workdir=a.workdir,
        iterations=a.iterations,
        versions=a.versions,
        max_parallelism=4,
        fallback_grid=(4, 5),
        hard_max_grid=(8, 8),
    )
    loop = DreamRSI(cfg, make_task(a.workdir), ScriptedDiscoveryAgent(), ScriptedPolicyAgent())
    state = loop.run()
    for entry in state["log"]:
        live = entry["live"]
        print(
            f"iter {entry['iteration']}: grid {live['effective_grid']} probes {live['probes']} "
            f"rounds {live['decision_rounds']} best {live['best_score']} -> selected "
            f"{entry.get('selected')} from {[round(c['score'], 3) for c in entry['offline']]}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m see")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="replay-evaluate one policy file over a trace pool")
    s.add_argument("--method", required=True)
    s.add_argument("--pool", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    s.add_argument("--lam", type=float, default=DEFAULT_LAMBDA)
    s.add_argument("--beta1", type=float, default=0.0)
    s.add_argument("--beta2", type=float, default=0.0)
    s.add_argument("--max-rounds", type=int, default=None)
    s.add_argument("--fallback", type=int, nargs=2, default=(10, 10))
    s.add_argument("--hard-max", type=int, nargs=2, default=(32, 19))
    s.add_argument("--max-parallelism", type=int, default=None)  # default: the newest tree's
    s.set_defaults(func=cmd_sweep)
    d = sub.add_parser("demo", help="run the whole loop on the toy task with scripted agents")
    d.add_argument("--workdir", required=True)
    d.add_argument("--iterations", type=int, default=3)
    d.add_argument("--versions", type=int, default=3)
    d.set_defaults(func=cmd_demo)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
