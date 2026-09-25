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
import numbers
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Mapping

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
    baseline_dir: str  # seed workspace: the program and anything it needs
    eval_program: str  # file the agent edits, e.g. "initial_program.py"
    problem_file: str  # task statement shown to the agent
    evaluate: Callable[[str], dict]
    higher_is_better: bool = True  # Sec. 3 wants larger = better; flip e.g. autocorrelation


def kill_process_group(p: subprocess.Popen, grace: float) -> None:
    """End ``p``'s whole process group: SIGTERM, then SIGKILL after ``grace`` seconds.

    ``p`` must have been started with ``start_new_session=True``, so its pid is the group id.
    Members that outlive ``p`` (a CLI that exits and leaves a background process holding its
    pipes) are still in the group, so both signals go to the group whether or not ``p`` is
    still running; a group that is already gone is not an error.
    """
    kill_process_groups([p], grace)


def kill_process_groups(ps: list, grace: float) -> None:
    """End every group in ``ps`` at once: SIGTERM to all, one shared ``grace``, SIGKILL to all."""
    for p in ps:
        _signal_group(p.pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    for p in ps:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for p in ps:
        _signal_group(p.pid, signal.SIGKILL)  # whatever ignored SIGTERM, including survivors of p
    for p in ps:
        p.wait()


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:  # nothing left in the group
        pass
    except PermissionError:  # Darwin: only zombies left in the group, which nothing can signal
        pass


class CommandAgent:
    """Run a coding-agent CLI; ``{prompt}`` in the argv template is replaced.

    Every call runs in its own session, so a timeout kills the CLI together with
    everything it forked, not just the CLI.
    """

    def __init__(
        self, argv, timeout: float = 3600.0, env: dict | None = None, kill_grace: float = 5.0
    ):
        self.argv = list(AGENT_PRESETS.get(argv, argv) if isinstance(argv, str) else argv)
        self.timeout = timeout
        self.env = env
        self.kill_grace = kill_grace  # seconds between SIGTERM and SIGKILL
        self._live: set[subprocess.Popen] = set()
        self._lock = threading.Lock()
        self._closed = False  # set by terminate(); a closed agent never spawns again

    def __call__(self, prompt: str, *, cwd: str, target: str) -> dict:
        argv = [prompt if a == "{prompt}" else a for a in self.argv]
        with self._lock:  # spawning under the lock closes the race with terminate()
            if self._closed:
                return self._terminated()
            p = subprocess.Popen(
                argv,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env={**os.environ, **(self.env or {})},
                start_new_session=True,
            )
            self._live.add(p)
        try:
            deadline = time.monotonic() + self.timeout
            swept = False
            while True:  # wait in slices, so a claimed call notices terminate() within a second
                remaining = max(0.0, deadline - time.monotonic())
                try:
                    out, err = p.communicate(timeout=min(1.0, remaining))
                    break
                except subprocess.TimeoutExpired:
                    with self._lock:
                        claimed = p not in self._live
                    if claimed:
                        # terminate() killed the group; a descendant outside it may hold the pipes
                        self._close_pipes(p)
                        return self._terminated()
                    if p.poll() is not None:  # the CLI has exited; something still holds its pipes
                        if not swept:  # a member of its group: kill the rest; the next slice is EOF
                            _signal_group(p.pid, signal.SIGKILL)
                            swept = True
                            continue
                        # still held after the sweep, so by a process outside the group, which is
                        # not killed by design: keep the CLI's exit status and drop what it printed
                        self._close_pipes(p)
                        return {
                            "returncode": p.returncode,
                            "stdout": "",
                            "stderr": "agent exited, but a process outside its group holds its "
                            "output; output dropped",
                        }
                    if time.monotonic() >= deadline:  # the CLI is alive, so its group id is valid
                        kill_process_group(p, self.kill_grace)
                        self._close_pipes(p)
                        return {
                            "returncode": None,
                            "stdout": "",
                            "stderr": f"agent timed out after {self.timeout}s",
                            "timed_out": True,
                        }
            # members that outlived the CLI hold nothing this call waits on any more; none of them
            # outlives the call (the group id is the reaped leader's pid: its reuse by an unrelated
            # session within these microseconds is not defended against)
            _signal_group(p.pid, signal.SIGKILL)
            with self._lock:  # terminate() claims a running call by removing it from _live
                if p not in self._live:
                    return self._terminated()
                self._live.discard(p)
            return {"returncode": p.returncode, "stdout": out[-4000:], "stderr": err[-4000:]}
        finally:
            if p.poll() is None:  # an exception escaped communicate(): take the group with us
                kill_process_group(p, self.kill_grace)
                self._close_pipes(p)
            with self._lock:
                self._live.discard(p)

    @staticmethod
    def _terminated() -> dict:
        return {"returncode": None, "stdout": "", "stderr": "agent terminated"}

    def terminate(self) -> None:
        """Kill every call still running (whole process groups) and refuse every later call."""
        with self._lock:
            self._closed = True
        self.kill_running()

    def kill_running(self) -> None:
        """Kill every call still running (whole process groups); later calls are still accepted.

        A running call is claimed by removing it from ``_live`` under the lock, so a call whose
        process had already finished keeps its real result. The claimed groups share one
        ``kill_grace``, so the cost of an interrupt does not grow with the batch width.
        """
        with self._lock:
            claimed = [p for p in self._live if p.poll() is None]
            self._live.difference_update(claimed)
        kill_process_groups(claimed, self.kill_grace)

    @staticmethod
    def _close_pipes(p: subprocess.Popen) -> None:
        """Drop what a killed group left in its pipes; reading them could block on a survivor."""
        for pipe in (p.stdout, p.stderr):
            if pipe is not None:
                pipe.close()


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

    def __init__(
        self,
        task: TaskSpec,
        agent: Callable,
        tree_dir: str,
        history_dir: str,
        baseline_score: float,
        max_parallelism: int,
        branch_count: int,
        refine_count: int,
        max_rounds: int | None = None,
        directions: Callable[[int], str] | None = None,
        serialize_eval: bool = True,
    ):
        self.task, self.agent = task, agent
        self.tree_dir, self.history_dir = tree_dir, history_dir
        self._direction_of = directions or (lambda branch: "")
        self._directions = {}
        self._eval_lock = threading.Lock() if serialize_eval else None
        self.cells = {}
        self._next_seq = 0
        self._cancelled = threading.Event()  # set once the batch in flight has been abandoned
        self.fault: str | None = None  # why the batch in flight was abandoned, once one was
        super().__init__(
            baseline_score,
            max_parallelism,
            branch_count,
            refine_count,
            max_rounds,
            record_episode=True,
        )

    def reset(self) -> None:
        if getattr(self, "cells", None):
            raise RuntimeError("a live episode cannot be reset once it has spent budget")
        super().reset()

    def _direction(self, branch: int) -> str:
        if branch not in self._directions:  # ask the provider once per branch
            self._directions[branch] = self._direction_of(branch)
        return self._directions[branch]

    def _cell(self, branch: int, attempt: int) -> CellMeta | None:
        if not self._in_grid(branch, attempt):
            return None
        done = self.cells.get(cell_id(branch, attempt))
        seq = done.seq if done else self._next_seq + branch  # provisional until executed
        return CellMeta(
            cell_id(branch, attempt),
            branch,
            attempt,
            None if attempt == 0 else cell_id(branch, attempt - 1),
            seq,
            {"direction": self._direction(branch)} if attempt == 0 else {},
        )

    def _execute(self, metas: list) -> list:
        jobs = []
        for m in metas:
            jobs.append((m, self._next_seq, self._direction(m.branch)))
            self._next_seq += 1
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallelism)
        futures = []
        try:
            for job in jobs:  # inside the try: an interrupt mid-submission still cancels the rest
                futures.append(pool.submit(self._run_attempt, *job))
            for f in concurrent.futures.as_completed(futures):
                f.result()  # surfaces a worker's exception as soon as it happens
        except BaseException as e:
            # a worker fault ends the episode and an interrupt ends the run; either way the rest
            # of the batch is abandoned: queued attempts never start, running agents are killed
            self._cancel(pool, close=not isinstance(e, Exception), fault=e)
            raise
        finally:
            try:
                pool.shutdown(wait=True)  # brief once the agents are dead
            except BaseException as e:  # an interrupt while the workers were being joined
                self._cancel(pool, close=True, fault=e)
                raise
        cells = [f.result() for f in futures]
        observations = [  # parents come from earlier batches: no parent/child pair shares one
            observation_for(
                c, self.cells.get(c.parent_id) if c.parent_id else None, self.baseline_score
            )
            for c in cells
        ]
        self.cells.update({c.id: c for c in cells})  # one statement, last: never half a batch
        return observations

    def _cancel(
        self, pool: concurrent.futures.ThreadPoolExecutor, close: bool, fault: BaseException
    ) -> None:
        """Abandon the batch: queued attempts never start and running agents are killed.

        ``close`` (an interrupt) also refuses every later call. A worker fault leaves the agent
        usable, because the loop goes on to offline() and to the next iteration with it. The
        fault is kept so online() can record it even when the policy swallows the exception.
        """
        self.fault = f"{type(fault).__name__}: {fault}"
        self._cancelled.set()
        pool.shutdown(wait=False, cancel_futures=True)
        stop = getattr(self.agent, "terminate" if close else "kill_running", None)
        if callable(stop):
            stop()

    def _resume_from(self, branch: int, attempt: int) -> str:
        """The parent's saved program; past an attempt that left none, the nearest ancestor's."""
        for a in range(attempt - 1, -1, -1):
            path = os.path.join(self.tree_dir, node_dirname(branch, a), self.task.eval_program)
            if os.path.exists(path):
                return path
        return os.path.join(self.task.baseline_dir, self.task.eval_program)

    def _run_attempt(self, meta: CellMeta, seq: int, direction: str) -> Cell:
        if self._cancelled.is_set():  # the batch was abandoned before this attempt started
            raise RuntimeError(f"attempt b{meta.branch}a{meta.attempt} cancelled with its batch")
        t = self.task
        node = os.path.join(self.tree_dir, node_dirname(meta.branch, meta.attempt))
        os.makedirs(node, exist_ok=True)
        shutil.copy(
            self._resume_from(meta.branch, meta.attempt), os.path.join(node, t.eval_program)
        )
        prompt = exploration_prompt(
            node_dir=node,
            history_dir=self.history_dir,
            baseline_dir=t.baseline_dir,
            eval_program=t.eval_program,
            problem_file=t.problem_file,
            direction_guidance=direction,
        )
        started = time.time()
        program = os.path.join(node, t.eval_program)
        if self._cancelled.is_set():  # abandoned while this attempt was being set up
            raise RuntimeError(f"attempt b{meta.branch}a{meta.attempt} cancelled with its batch")
        try:
            run = self.agent(prompt, cwd=self.tree_dir, target=node)
        except Exception as e:  # a crashed agent is a failed attempt, not a failed episode
            run = {"returncode": None, "stderr": f"{type(e).__name__}: {e}"}
            if os.path.exists(program):
                os.remove(program)
        if self._cancelled.is_set():  # _cancel killed the agent: do not evaluate what it left
            raise RuntimeError(f"attempt b{meta.branch}a{meta.attempt} killed with its batch")
        if not os.path.exists(program):
            result = {
                "combined_score": 0.0,
                "no_program": True,
                "error": f"agent left no program ({(run or {}).get('stderr', '')[:200]})",
            }
        else:
            result = self._evaluate(program)
        error = result.get("error")
        if run and run.get("timed_out") and not error:
            # the agent was killed at its timeout: what it left is scored, but the attempt is
            # the failure the taxonomy already names (classify_failure maps this text to timeout)
            error = run["stderr"]
        fail_class = "no_program" if result.get("no_program") else classify_failure(error)
        score = oriented_score(t, result)
        os.makedirs(os.path.join(node, "eval"), exist_ok=True)
        with open(os.path.join(node, "eval", "score.json"), "w") as f:
            json.dump(
                {
                    **result,
                    "fail_class": fail_class,
                    "seconds": time.time() - started,
                    "agent_returncode": run.get("returncode") if run else None,
                    "agent_timed_out": bool(run.get("timed_out")) if run else False,
                },
                f,
                indent=1,
                default=str,
            )
        if error:
            with open(os.path.join(node, "error.txt"), "w") as f:
                f.write(str(error))
        return Cell(
            meta.branch,
            meta.attempt,
            seq,
            score,
            evaluated=not result.get("no_program") and not result.get("evaluator_crashed"),
            valid=bool(result.get("validity", error is None)),
            fail_class=fail_class,
            error=error,
            n_valid=result.get("n_valid"),
            n_total=result.get("n_total"),
            tags=dict(meta.tags),
        )

    def _evaluate(self, program: str) -> dict:
        lock = self._eval_lock or _NullLock()
        with lock:  # timing tasks (Lasso, kernels) are distorted by concurrent evaluation
            try:
                result = self.task.evaluate(program)
                # checked inside the guard: a bad result fails its own cell, not the whole batch
                if not (
                    isinstance(result, Mapping)
                    and isinstance(result.get("combined_score"), numbers.Real)  # numpy scalars pass
                    # absent, None, or a string; any other falsy value is still malformed
                    and ((err := result.get("error")) is None or isinstance(err, str))
                ):
                    raise ValueError(f"malformed evaluator result: {repr(result)[:200]}")
                return dict(result)
            except Exception as e:
                return {
                    "combined_score": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                    "evaluator_crashed": True,
                }

    def frozen(self, trace_id: str, info: dict) -> Trace:
        return Trace(
            self.cells.values(),
            self.baseline_score,
            self.max_parallelism,
            trace_id=trace_id,
            grid=(self.branch_count, self.refine_count),
            info=info,
        )

    def manifest_stats(self) -> dict:
        ok = [c for c in self.cells.values() if c.success and c.score is not None]
        best = max(ok, key=lambda c: c.score, default=None)
        depth = collections.Counter(c.branch for c in self.cells.values())
        return {
            "probes": len(self.cells),
            "decision_rounds": self.decision_rounds,
            "effective_sequential_rounds": self.effective_sequential_rounds,
            "opened_width": len(depth),
            "max_depth": max(depth.values(), default=0),
            "best_score": best.score if best else None,
            "best_cell": best.id if best else None,
            "best_attempt": best.attempt if best else None,
            "baseline_score": self.baseline_score,
            "fail_classes": dict(collections.Counter(c.fail_class for c in self.cells.values())),
        }


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
