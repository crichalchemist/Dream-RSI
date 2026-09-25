"""Replay can stand in for live execution only if, on-policy, it repeats the live episode."""

import itertools
import json
import os

import pytest

from see.loader import load_policy
from see.loop import BASELINE_POLICY, DreamRSI, LoopConfig
from see.objective import attainment, beta_sweep, eq1_value, run_episode
from see.pool import context_factory, load_pool
from see.toy import PORTFOLIO, ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

# With this grid, seeds 0 and 1 leave the portfolio's tree partial (cells the live episode could
# have probed but never did); seed 2 and every parallel-refine episode fill the grid.
SEEDS = (0, 1, 2)
POLICIES = {"parallel_refine": BASELINE_POLICY, "portfolio": PORTFOLIO}


def _transcript(log: list) -> list:
    """Per round: the prefix the policy decided on, its batch, and what each probe revealed.

    The prefix's ``legal`` count is left out: live it counts every unprobed cell of the planned
    grid, in replay only those in the recorded tree (GAPS.md §6), so the two differ whenever
    the live tree is partial. The batches are what must agree.
    """
    return [
        (
            step["round"],
            {k: step["prefix"][k] for k in ("opened", "revealed", "best")},
            step["batch"],
            step["revealed"],
        )
        for step in log
    ]


def _first_differing_round(live: list, replay: list) -> int:
    pairs = itertools.zip_longest(live, replay)  # a missing round differs too
    return next(number for number, (a, b) in enumerate(pairs, start=1) if a != b)


@pytest.mark.usefixtures("stub_prompts")
def test_replay_reproduces_live_episodes_across_seeds_and_policies(tmp_path):
    for seed in SEEDS:
        for name, policy_file in POLICIES.items():
            where = f"seed {seed}, {name}"
            work = str(tmp_path / f"{name}-s{seed}")
            cfg = LoopConfig(
                workdir=work,
                max_parallelism=4,
                fallback_grid=(6, 5),  # more branches than workers: roots open across rounds
                hard_max_grid=(8, 8),
                betas=(0.0, 0.5, 1.0),
                beta1=0.01,  # non-zero, so Eq. (1) also depends on probes and rounds
                beta2=0.05,
                initial_policy=policy_file,
            )
            loop = DreamRSI(
                cfg, make_task(work), ScriptedDiscoveryAgent(seed=seed), ScriptedPolicyAgent()
            )
            manifest = loop.online(1)  # the live episode, frozen into trace_pool/iter0001/
            assert manifest["error"] is None, f"{where}: {manifest['error']}"
            with open(os.path.join(loop.pool, "iter0001", "live_episode.jsonl")) as f:
                live_log = [json.loads(line) for line in f]
            live = _transcript(live_log)
            assert len(live) > 1, f"{where}: the live episode should span several rounds"
            spent = sum(len(step["batch"]) for step in live_log)
            live_eq1 = eq1_value(
                manifest["best_score"],
                manifest["baseline_score"],
                spent,
                manifest["decision_rounds"],
                cfg.beta1,
                cfg.beta2,
            )

            pool = load_pool(loop.pool)  # read back exactly as `python -m see sweep` does
            assert len(pool) == 1, where
            trace = pool[0][0]
            context_for = context_factory(pool, cfg.fallback_grid, cfg.hard_max_grid)
            policy_cls = load_policy(loop.state["deployed"])
            episode = run_episode(
                policy_cls(None),
                trace,
                context_for(trace),
                max_rounds=cfg.max_replay_rounds,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
                record=True,
            )
            replay = _transcript(episode.log)
            assert replay == live, (
                f"{where}: replay departs from live at round {_first_differing_round(live, replay)}"
            )
            # exact equality: the claim is determinism, and JSON round-trips floats exactly
            assert episode.probes == spent == manifest["probes"], where
            assert episode.rounds == manifest["decision_rounds"], where
            assert episode.effective_rounds == manifest["effective_sequential_rounds"], where
            assert episode.best == manifest["best_score"], where
            assert episode.eq1 == live_eq1, where

            # the offline scorer's default-beta leg is the live policy: it must score the live run
            report, executions = beta_sweep(
                policy_cls,
                [trace],
                context_for=context_for,
                betas=cfg.betas,
                lam=cfg.lam,
                beta1=cfg.beta1,
                beta2=cfg.beta2,
                max_rounds=cfg.max_replay_rounds,
            )
            assert report["valid"], f"{where}: {report['errors']}"
            default = _transcript(executions[-1]["log"])
            assert default == live, (
                f"{where}: beta_sweep's default episode departs from live at round "
                f"{_first_differing_round(live, default)}"
            )
            assert report["eq1"]["V"] == live_eq1, where
            assert report["default_episode"] == {
                "attainment": attainment(manifest["best_score"], trace),
                "work": spent / len(trace),
                "penalty": manifest["effective_sequential_rounds"] / spent,
            }, where
