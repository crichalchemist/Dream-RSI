"""The outer Dream-RSI loop (paper Sec. 3, Fig. 1).

For t = 1, 2, ...: deploy pi_t online to grow a new tree T_t, append it to
the history H_t, then build M policy versions offline, pi_t^0 = pi_t first,
each later one written by the policy-development agent from replay feedback,
score each on every tree in H_t, and deploy the best as pi_{t+1}. Because
pi_t^0 is a candidate, the deployed policy never scores below pi_t on H_t.

Layout under ``workdir`` (names quoted in Listing 2 are kept):

    runs/iterNNNN/tree/attempt_*/     the live tree (agent cwd)
    runs/iterNNNN/history/            links to earlier trees ($history_dir)
    trace_pool/iterNNNN/              trace.json + live_cycle_manifest.json
    trace_pool/_current               link to the newest completed cycle
    policy_dev/method.py              {method_file}, edited by the dev agent
    policy_dev/history/baseline/      the parallel-refine floor
    policy_dev/history/rNNNN_*/       method.py + proposal_results/
    state.json                        deployed policy, round counter, log
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable

from see.live import LiveQuestion, TaskSpec, kill_process_group, oriented_score
from see.loader import load_policy
from see.objective import DEFAULT_BETAS, DEFAULT_LAMBDA, score_of, validate_plan
from see.policy.api import GridPlanningContext
from see.prompts import policy_improvement_prompt

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE_POLICY = os.path.join(PKG_ROOT, "see", "policies", "parallel_refine.py")


def archive_name(round_no: int, t: int, m: int) -> str:
    """The policy_dev/history/ entry for version m of iteration t; rounds are numbered globally."""
    return f"r{round_no:04d}_t{t:02d}_m{m}"


def install_signal_handlers() -> None:
    """Make SIGTERM and SIGHUP take the same path as Ctrl-C: raise KeyboardInterrupt in the main
    thread. A SIGHUP that was inherited ignored (a nohup launch) stays ignored.

    Called by the CLI entry points only; a library must not change signal disposition on import.
    """
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        signal.signal(signal.SIGHUP, _raise_keyboard_interrupt)


def _raise_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


@dataclasses.dataclass
class LoopConfig:
    workdir: str
    iterations: int = 5  # outer rounds (5 for Lasso, 10 for math in the paper)
    versions: int = 4  # M, policy versions per offline phase (unstated)
    max_parallelism: int = 10  # W workers (10 for Gemini-3.1-Pro runs, 32 for Flash)
    fallback_grid: tuple = (10, 10)  # runner default: 10 branches x 11 attempts = 110 calls
    hard_max_grid: tuple = (32, 19)  # caps plan_grid may request (unstated)
    max_live_rounds: int | None = None  # K1 (unstated)
    max_replay_rounds: int | None = None  # K2 (unstated)
    betas: tuple = DEFAULT_BETAS
    lam: float = DEFAULT_LAMBDA
    beta1: float = 0.0  # Eq. (1) coefficients (unstated)
    beta2: float = 0.0
    objective: str = "pareto"  # "pareto" (Listing 2) or "eq1" (Sec. 3)
    sweep_timeout: float = 1800.0
    kill_grace: float = 5.0  # seconds between SIGTERM and SIGKILL for the sweep subprocess
    initial_policy: str = BASELINE_POLICY  # pi_1: the paper starts from parallel refine
    serialize_eval: bool = True


class DreamRSI:
    def __init__(
        self,
        config: LoopConfig,
        task: TaskSpec,
        discovery_agent: Callable,
        policy_agent: Callable,
        directions: Callable[[int], str] | None = None,
    ):
        self.c, self.task = config, task
        self.discovery_agent, self.policy_agent = discovery_agent, policy_agent
        self.directions = directions
        self.w = os.path.abspath(config.workdir)
        self.pool = os.path.join(self.w, "trace_pool")
        self.dev = os.path.join(self.w, "policy_dev")
        self.dev_history = os.path.join(self.dev, "history")
        for d in (self.pool, self.dev_history, os.path.join(self.w, "runs")):
            os.makedirs(d, exist_ok=True)
        self.state_path = os.path.join(self.w, "state.json")
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                self.state = json.load(f)
        else:
            first = os.path.join(self.w, "deployed", "iter0001.py")
            os.makedirs(os.path.dirname(first), exist_ok=True)
            shutil.copy(config.initial_policy, first)
            self.state = {"iteration": 0, "round": 0, "deployed": first, "log": []}
            self._save_state()

    # -- bookkeeping ------------------------------------------------------------
    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=1)

    def manifests(self) -> list:
        out = []
        for path in sorted(glob.glob(os.path.join(self.pool, "iter*", "live_cycle_manifest.json"))):
            with open(path) as f:
                out.append(json.load(f))
        return out

    def _context(self, history) -> GridPlanningContext:
        (fb, fr), (hb, hr) = self.c.fallback_grid, self.c.hard_max_grid
        return GridPlanningContext(tuple(history), fb, fr, hb, hr, self.c.max_parallelism)

    def baseline_score(self) -> float:
        path = os.path.join(self.w, "baseline_eval.json")
        if not os.path.exists(path):
            result = dict(
                self.task.evaluate(os.path.join(self.task.baseline_dir, self.task.eval_program))
            )
            with open(path, "w") as f:
                json.dump(result, f, indent=1, default=str)
        with open(path) as f:
            return oriented_score(self.task, json.load(f))

    # -- the loop -----------------------------------------------------------------
    def run(self, iterations: int | None = None):
        for _ in range(iterations or self.c.iterations):
            t = self.state["iteration"] + 1
            self.online(t)
            self.offline(t)
            self.state["iteration"] = t
            self._save_state()
        return self.state

    def online(self, t: int):
        """Stage 1: the deployed policy drives discovery; the tree is frozen into the pool."""
        run_dir = os.path.join(self.w, "runs", f"iter{t:04d}")
        out = os.path.join(self.pool, f"iter{t:04d}")
        # runs/ is created first and trace_pool/ last; the first archive is what a restart's
        # offline() would recreate, because the round counter is persisted only on success
        first_archive = os.path.join(self.dev_history, archive_name(self.state["round"] + 1, t, 0))
        existing = [p for p in (run_dir, out, first_archive) if os.path.exists(p)]
        if existing:
            one = len(existing) == 1
            raise RuntimeError(
                f"{' and '.join(existing)} {'exists' if one else 'exist'}: iteration {t} was "
                f"interrupted or already ran; delete {'it' if one else 'them'}, do not merge into "
                f"{'it' if one else 'them'}"
            )
        policy = load_policy(self.state["deployed"])(None)  # baked-in default beta
        ctx = self._context(self.manifests())
        plan = validate_plan(policy.plan_grid(ctx), ctx)
        grid = (plan.branch_count, plan.refine_count) if plan else tuple(self.c.fallback_grid)
        # before runs/iterNNNN exists: an interrupt here leaves nothing to clean up
        baseline = self.baseline_score()
        tree, history = os.path.join(run_dir, "tree"), os.path.join(run_dir, "history")
        os.makedirs(tree, exist_ok=True)
        os.makedirs(history, exist_ok=True)
        for prev in range(1, t):
            link = os.path.join(history, f"iter{prev:04d}")
            if not os.path.lexists(link):
                os.symlink(os.path.join(self.w, "runs", f"iter{prev:04d}", "tree"), link)
        q = LiveQuestion(
            self.task,
            self.discovery_agent,
            tree,
            history,
            baseline,
            self.c.max_parallelism,
            grid[0],
            grid[1],
            self.c.max_live_rounds,
            self.directions,
            self.c.serialize_eval,
        )
        started, error = time.time(), None
        try:
            policy.solve(q, budget=None)
        except Exception as e:  # keep what was collected
            error = f"{type(e).__name__}: {e}"
        except BaseException as e:  # an interrupt: freeze under runs/, never into the pool
            partial = os.path.join(run_dir, "partial")
            os.makedirs(partial, exist_ok=True)
            error = f"{type(e).__name__}: {e}"
            manifest = self._manifest(t, policy, plan, grid, q, error, started)
            manifest["partial"] = True
            self._freeze(
                q, partial, f"iter{t:04d}-partial", {"iteration": t, "partial": True}, manifest
            )
            raise
        manifest = self._manifest(t, policy, plan, grid, q, error, started)
        os.makedirs(out)
        self._freeze(q, out, f"iter{t:04d}", {"iteration": t}, manifest)
        current = os.path.join(self.pool, "_current")
        if os.path.lexists(current):
            os.remove(current)
        os.symlink(f"iter{t:04d}", current)
        self.state["log"].append({"iteration": t, "live": manifest})
        return manifest

    def _manifest(self, t, policy, plan, grid, q: LiveQuestion, error, started) -> dict:
        return {
            "iteration": t,
            "policy_round": self.state.get("deployed_round", "initial"),
            "beta": getattr(policy, "beta", None),
            "planned_grid": dataclasses.asdict(plan) if plan else None,
            "used_fallback": plan is None,
            "effective_grid": {"branch_count": grid[0], "refine_count": grid[1]},
            **q.manifest_stats(),
            "error": error,
            "started": started,
            "finished": time.time(),
        }

    @staticmethod
    def _freeze(q: LiveQuestion, into: str, trace_id: str, info: dict, manifest: dict) -> None:
        q.frozen(trace_id, info).save(os.path.join(into, "trace.json"))
        with open(os.path.join(into, "live_episode.jsonl"), "w") as f:
            for step in q.episode:
                f.write(json.dumps(step) + "\n")
        with open(os.path.join(into, "live_cycle_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)

    def offline(self, t: int) -> str:
        """Stages 2-3: evaluate M versions by replay over H_t and deploy the argmax."""
        floor = os.path.join(self.dev_history, "baseline")
        os.makedirs(floor, exist_ok=True)
        shutil.copy(BASELINE_POLICY, os.path.join(floor, "method.py"))
        self._sweep(os.path.join(floor, "method.py"), floor)  # re-scored on the current pool
        method_file = os.path.join(self.dev, "method.py")
        candidates = [self._archive(self.state["deployed"], t, 0)]
        for m in range(1, self.c.versions):
            shutil.copy(candidates[-1]["method"], method_file)  # revise the latest version
            prompt = policy_improvement_prompt(
                method_file=method_file, history_dir=self.dev_history, trace_pool=self.pool
            )
            run = self.policy_agent(prompt, cwd=self.dev, target=method_file)
            candidates.append(self._archive(method_file, t, m, agent_run=run))
        best = max(candidates, key=lambda c: (c["score"], -c["m"]))  # ties keep the earlier one
        deployed = self._deploy(t, best)
        self.state["log"][-1]["offline"] = [
            {k: c[k] for k in ("round", "m", "score", "valid", "sha256")} for c in candidates
        ]
        self.state["log"][-1]["selected"] = best["round"]
        self._save_state()
        return deployed

    def _deploy(self, t: int, record: dict) -> str:
        """Copy the scored candidate to deployed/ as pi_{t+1}, refusing any changed bytes."""
        with open(record["method"], "rb") as f:
            code = f.read()
        digest = hashlib.sha256(code).hexdigest()
        if digest != record["sha256"]:
            raise RuntimeError(
                f"{record['method']} changed after it was scored (sha256 {digest}, "
                f"scored {record['sha256']}): refusing to deploy it"
            )
        deployed = os.path.join(self.w, "deployed", f"iter{t + 1:04d}.py")
        with open(deployed, "wb") as f:
            f.write(code)  # the verified bytes, not a second read of the file
        self.state["deployed"], self.state["deployed_round"] = deployed, record["round"]
        self.state["deployed_sha256"] = digest
        self._save_state()
        return deployed

    def _archive(self, method_path: str, t: int, m: int, agent_run=None) -> dict:
        self.state["round"] += 1
        name = archive_name(self.state["round"], t, m)
        rdir = os.path.join(self.dev_history, name)
        os.makedirs(rdir, exist_ok=True)
        archived = os.path.join(rdir, "method.py")
        shutil.copy(method_path, archived)
        with open(archived, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()  # before the sweep runs the code
        if agent_run is not None:
            with open(os.path.join(rdir, "dev_agent.json"), "w") as f:
                json.dump(agent_run, f, indent=1, default=str)
        report = self._sweep(archived, rdir)
        return {
            "round": name,
            "m": m,
            "method": archived,
            "sha256": digest,
            "valid": report.get("valid", False),
            "score": score_of(report, self.c.objective) if report.get("valid") else float("-inf"),
        }

    def _sweep(self, method: str, rdir: str) -> dict:
        """Replay-evaluate one version in a subprocess (it is LLM-written code)."""
        out = os.path.join(rdir, "proposal_results")
        os.makedirs(out, exist_ok=True)
        (fb, fr), (hb, hr) = self.c.fallback_grid, self.c.hard_max_grid
        cmd = [
            sys.executable,
            "-m",
            "see",
            "sweep",
            "--method",
            method,
            "--pool",
            self.pool,
            "--out",
            out,
            "--lam",
            str(self.c.lam),
            "--beta1",
            str(self.c.beta1),
            "--beta2",
            str(self.c.beta2),
            "--fallback",
            str(fb),
            str(fr),
            "--hard-max",
            str(hb),
            str(hr),
            "--betas",
            *map(str, self.c.betas),
        ]
        if self.c.max_replay_rounds is not None:
            cmd += ["--max-rounds", str(self.c.max_replay_rounds)]
        report_path = os.path.join(out, "beta_sweep.json")
        p = subprocess.Popen(
            cmd,
            cwd=PKG_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,  # a timeout or an interrupt kills what the policy code forked
        )
        try:
            try:
                _, err = p.communicate(timeout=self.c.sweep_timeout)
            except subprocess.TimeoutExpired:
                kill_process_group(p, self.c.kill_grace)
                raise RuntimeError(f"sweep timed out after {self.c.sweep_timeout}s") from None
            if p.returncode != 0 or not os.path.exists(report_path):
                raise RuntimeError(err[-2000:] or f"exit {p.returncode}")
            with open(report_path) as f:
                return json.load(f)
        except Exception as e:
            report = {
                "valid": False,
                "errors": [f"{type(e).__name__}: {e}"],
                "pareto": {"reward": float("-inf")},
                "eq1": {"V": float("-inf")},
            }
            with open(report_path, "w") as f:
                json.dump(report, f, indent=1)
            return report
        finally:
            if p.poll() is None:  # an interrupt escaped communicate(): take the child with us
                kill_process_group(p, self.c.kill_grace)
