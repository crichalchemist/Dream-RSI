"""A toy task and scripted agents, so the loop runs end to end without an LLM.

The "program" is a JSON file holding one number. The scripted discovery agent
nudges its parent's number by a step whose drift depends on the branch (some
directions are good, most are not) and sometimes writes a broken file; the
evaluator returns the number. The scripted policy agent writes the example
portfolio policy with a different default beta each revision. Nothing here
models a real coding agent; it only exercises the plumbing.
"""
import hashlib
import json
import os
import random
import re

from see.live import TaskSpec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORTFOLIO = os.path.join(ROOT, "see", "policies", "portfolio.py")
PROGRAM = "solution.json"


def _rng(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:16], 16))


def evaluate(path: str) -> dict:
    try:
        with open(path) as f:
            x = float(json.load(f)["x"])
    except Exception as e:
        return {"combined_score": 0.0, "validity": 0.0, "n_valid": 0, "n_total": 1,
                "error": f"ValueError: unreadable solution ({type(e).__name__})"}
    return {"combined_score": x, "validity": 1.0, "n_valid": 1, "n_total": 1}


def make_task(workdir: str) -> TaskSpec:
    base = os.path.join(workdir, "task", "baseline")
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, PROGRAM), "w") as f:
        json.dump({"x": 1.0}, f)
    problem = os.path.join(workdir, "task", "problem.md")
    with open(problem, "w") as f:
        f.write("Maximise x in solution.json.\n")
    return TaskSpec("toy", base, PROGRAM, problem, evaluate)


class ScriptedDiscoveryAgent:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        node = os.path.basename(target)
        branch, attempt = map(int, re.match(r"attempt_b(\d+)_a(\d+)", node).groups())
        iteration = os.path.basename(os.path.dirname(os.path.dirname(target)))
        drift = _rng(self.seed, "branch", branch).gauss(0.0, 0.05)  # a direction's quality
        r = _rng(self.seed, iteration, branch, attempt)
        path = os.path.join(target, PROGRAM)
        x = 1.0  # repair a broken parent from the nearest readable ancestor
        for a in range(attempt, -1, -1):
            src = path if a == attempt else os.path.join(cwd, f"attempt_b{branch:03d}_a{a:03d}", PROGRAM)
            try:
                with open(src) as f:
                    x = float(json.load(f)["x"])
                break
            except (OSError, ValueError, KeyError):
                continue
        with open(os.path.join(target, "proposal.md"), "w") as f:
            f.write(f"scripted step on branch {branch}\n")
        if r.random() < 0.1:
            with open(path, "w") as f:
                f.write("{broken")
            return {"returncode": 0}
        with open(path, "w") as f:
            json.dump({"x": round(max(0.0, x + drift + r.gauss(0.0, 0.03)), 6)}, f)
        return {"returncode": 0}


class ScriptedPolicyAgent:
    """Writes the portfolio example with the next default beta from ``betas``."""

    def __init__(self, betas=(0.4, 0.8, 0.6), broken_every: int = 0):
        self.betas, self.broken_every, self.calls = betas, broken_every, 0

    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        self.calls += 1
        with open(PORTFOLIO) as f:
            src = f.read()
        if self.broken_every and self.calls % self.broken_every == 0:
            src = src.replace("def solve(", "def solve(self_broken, ", 1)
        beta = self.betas[(self.calls - 1) % len(self.betas)]
        src = src.replace('self.config.get("beta", 0.6)', f'self.config.get("beta", {beta})')
        with open(target, "w") as f:
            f.write(src)
        return {"returncode": 0, "default_beta": beta}
