"""Online rollout: the replay API backed by a real coding agent and evaluator.

Paper Sec. 3: each selected node is assigned to a worker; the discovery agent
resumes the parent's saved workspace, produces one new attempt, and the
evaluator scores it. Directory names follow Listing 1 (sibling
``attempt_*/`` dirs holding ``proposal.md``, the program, ``eval/score.json``
and ``error.txt``). Everything the paper leaves open about the orchestration
layer (process isolation, how workspaces are snapshotted, what the direction
provider says) is a reconstruction choice noted in GAPS.md.
"""
from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

from see.policy.api import CellMeta
from see.policy.observation_signal import classify_failure
from see.prompts import exploration_prompt
from see.world import Cell, Question, Trace, cell_id, observation_for

# Non-interactive invocations of common coding-agent CLIs. Flags drift between
# releases: check `<cli> --help` for the installed version.
AGENT_PRESETS = {
    "gemini": ["gemini", "--yolo", "--prompt", "{prompt}"],
    "claude": ["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits"],
}


@dataclasses.dataclass
class TaskSpec:
    """One discovery task. ``evaluate`` follows SimpleTES: path -> dict with
    ``combined_score`` and optionally ``error``, ``validity``, ``n_valid``, ``n_total``."""

    name: str
    baseline_dir: str                  # seed workspace: the program and anything it needs
    eval_program: str                  # file the agent edits, e.g. "initial_program.py"
    problem_file: str                  # task statement shown to the agent
    evaluate: Callable[[str], dict]
    higher_is_better: bool = True      # Sec. 3 wants larger = better; flip e.g. autocorrelation


class CommandAgent:
    """Run a coding-agent CLI; ``{prompt}`` in the argv template is replaced."""

    def __init__(self, argv, timeout: float = 3600.0, env: Optional[dict] = None):
        self.argv = list(AGENT_PRESETS.get(argv, argv) if isinstance(argv, str) else argv)
        self.timeout = timeout
        self.env = env

    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        argv = [prompt if a == "{prompt}" else a for a in self.argv]
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                               timeout=self.timeout, env={**os.environ, **(self.env or {})})
            return {"returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}
        except subprocess.TimeoutExpired:
            return {"returncode": None, "stdout": "", "stderr": f"agent timed out after {self.timeout}s"}


def oriented_score(task: TaskSpec, result: dict) -> float:
    score = float(result.get("combined_score", 0.0) or 0.0)
    return score if task.higher_is_better else -score


def node_dirname(branch: int, attempt: int) -> str:
    return f"attempt_b{branch:03d}_a{attempt:03d}"


class LiveQuestion(Question):
    """Every cell inside the planned grid exists; probing one runs agent + evaluator.

    ``directions(branch) -> str`` is the direction provider: its text fills
    ``$direction_guidance`` for every attempt on that branch.
    """

    def __init__(self, task: TaskSpec, agent: Callable, tree_dir: str, history_dir: str,
                 baseline_score: float, max_parallelism: int, branch_count: int,
                 refine_count: int, max_rounds: Optional[int] = None,
                 directions: Optional[Callable[[int], str]] = None,
                 serialize_eval: bool = True):
        self.task, self.agent = task, agent
        self.tree_dir, self.history_dir = tree_dir, history_dir
        self._direction_of = directions or (lambda branch: "")
        self._directions = {}
        self._eval_lock = threading.Lock() if serialize_eval else None
        self.cells = {}
        self._next_seq = 0
        super().__init__(baseline_score, max_parallelism, branch_count, refine_count, max_rounds,
                         record_episode=True)

    def reset(self) -> None:
        if getattr(self, "cells", None):
            raise RuntimeError("a live episode cannot be reset once it has spent budget")
        super().reset()

    def _direction(self, branch: int) -> str:
        if branch not in self._directions:  # ask the provider once per branch
            self._directions[branch] = self._direction_of(branch)
        return self._directions[branch]

    def _cell(self, branch: int, attempt: int) -> Optional[CellMeta]:
        if not self._in_grid(branch, attempt):
            return None
        done = self.cells.get(cell_id(branch, attempt))
        seq = done.seq if done else self._next_seq + branch  # provisional until executed
        return CellMeta(cell_id(branch, attempt), branch, attempt,
                        None if attempt == 0 else cell_id(branch, attempt - 1), seq,
                        {"direction": self._direction(branch)} if attempt == 0 else {})

    def _execute(self, metas: list) -> list:
        jobs = []
        for m in metas:
            jobs.append((m, self._next_seq, self._direction(m.branch)))
            self._next_seq += 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallelism) as pool:
            cells = list(pool.map(lambda job: self._run_attempt(*job), jobs))
        out = []
        for cell in cells:
            self.cells[cell.id] = cell
            parent = self.cells.get(cell.parent_id) if cell.parent_id else None
            out.append(observation_for(cell, parent, self.baseline_score))
        return out

    def _resume_from(self, branch: int, attempt: int) -> str:
        """The parent's saved program; past an attempt that left none, the nearest ancestor's."""
        for a in range(attempt - 1, -1, -1):
            path = os.path.join(self.tree_dir, node_dirname(branch, a), self.task.eval_program)
            if os.path.exists(path):
                return path
        return os.path.join(self.task.baseline_dir, self.task.eval_program)

    def _run_attempt(self, meta: CellMeta, seq: int, direction: str) -> Cell:
        t = self.task
        node = os.path.join(self.tree_dir, node_dirname(meta.branch, meta.attempt))
        os.makedirs(node, exist_ok=True)
        shutil.copy(self._resume_from(meta.branch, meta.attempt), os.path.join(node, t.eval_program))
        prompt = exploration_prompt(node_dir=node, history_dir=self.history_dir,
                                    baseline_dir=t.baseline_dir, eval_program=t.eval_program,
                                    problem_file=t.problem_file, direction_guidance=direction)
        started = time.time()
        program = os.path.join(node, t.eval_program)
        try:
            run = self.agent(prompt, cwd=self.tree_dir, target=node)
        except Exception as e:  # a crashed agent is a failed attempt, not a failed episode
            run = {"returncode": None, "stderr": f"{type(e).__name__}: {e}"}
            if os.path.exists(program):
                os.remove(program)
        if not os.path.exists(program):
            result = {"combined_score": 0.0, "no_program": True,
                      "error": f"agent left no program ({(run or {}).get('stderr', '')[:200]})"}
        else:
            result = self._evaluate(program)
        error = result.get("error")
        fail_class = "no_program" if result.get("no_program") else classify_failure(error)
        score = oriented_score(t, result)
        os.makedirs(os.path.join(node, "eval"), exist_ok=True)
        with open(os.path.join(node, "eval", "score.json"), "w") as f:
            json.dump({**result, "fail_class": fail_class, "seconds": time.time() - started,
                       "agent_returncode": run.get("returncode") if run else None}, f,
                      indent=1, default=str)
        if error:
            with open(os.path.join(node, "error.txt"), "w") as f:
                f.write(str(error))
        return Cell(meta.branch, meta.attempt, seq, score,
                    evaluated=not result.get("no_program") and not result.get("evaluator_crashed"),
                    valid=bool(result.get("validity", error is None)), fail_class=fail_class,
                    error=error, n_valid=result.get("n_valid"), n_total=result.get("n_total"),
                    tags=dict(meta.tags))

    def _evaluate(self, program: str) -> dict:
        lock = self._eval_lock or _NullLock()
        with lock:  # timing tasks (Lasso, kernels) are distorted by concurrent evaluation
            try:
                return dict(self.task.evaluate(program))
            except Exception as e:
                return {"combined_score": 0.0, "error": f"{type(e).__name__}: {e}",
                        "evaluator_crashed": True}

    def frozen(self, trace_id: str, info: dict) -> Trace:
        return Trace(self.cells.values(), self.baseline_score, self.max_parallelism,
                     trace_id=trace_id, grid=(self.branch_count, self.refine_count), info=info)

    def manifest_stats(self) -> dict:
        ok = [c for c in self.cells.values() if c.success and c.score is not None]
        best = max(ok, key=lambda c: c.score, default=None)
        depth = collections.Counter(c.branch for c in self.cells.values())
        return {
            "probes": len(self.cells), "decision_rounds": self.decision_rounds,
            "effective_sequential_rounds": self.effective_sequential_rounds,
            "opened_width": len(depth), "max_depth": max(depth.values(), default=0),
            "best_score": best.score if best else None, "best_cell": best.id if best else None,
            "best_attempt": best.attempt if best else None,
            "baseline_score": self.baseline_score,
            "fail_classes": dict(collections.Counter(c.fail_class for c in self.cells.values())),
        }


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
