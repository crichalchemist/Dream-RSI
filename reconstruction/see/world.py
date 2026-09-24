"""Frozen discovery trees (replay worlds) and the replay-side ``question``.

Paper Sec. 3: every non-root node has one parent and only the root and the
leaves are selectable, so a discovery tree is a root plus disjoint chains
(Listing 2: "a frozen, irregular branch x attempt grid"). Cell (b, a) is
attempt ``a`` of branch ``b``; attempt 0 starts from the root workspace.

Departure from the paper's formalism, following Listing 2 instead: Sec. 3
puts the single root r in the action set, so a batch could open at most one
branch per round and "selecting r" reveals the earliest-created unrevealed
branch. That cannot express the paper's own baseline, which opens W
workspaces in its first round. Listing 2's API exposes each unopened branch
root as its own legal cell, which is what is implemented here.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Iterable

from see.policy.api import CellMeta, Observation
from see.policy.observation_signal import OK

TRACE_FORMAT = "dream-rsi-trace/1"


def cell_id(branch: int, attempt: int) -> str:
    return f"b{branch}a{attempt}"


class IllegalBatch(ValueError):
    pass


@dataclasses.dataclass
class Cell:
    branch: int
    attempt: int
    seq: int
    score: float | None
    evaluated: bool = True
    valid: bool = True
    fail_class: str = OK
    error: str | None = None
    n_valid: int | None = None
    n_total: int | None = None
    tags: dict = dataclasses.field(default_factory=dict)

    @property
    def id(self) -> str:
        return cell_id(self.branch, self.attempt)

    @property
    def parent_id(self) -> str | None:
        return None if self.attempt == 0 else cell_id(self.branch, self.attempt - 1)

    @property
    def success(self) -> bool:
        return self.evaluated and self.error is None and self.fail_class == OK


class Trace:
    """One completed discovery tree T_i, frozen for replay."""

    def __init__(
        self,
        cells: Iterable[Cell],
        baseline_score: float,
        max_parallelism: int,
        trace_id: str = "",
        grid: tuple | None = None,
        info: dict | None = None,
    ):
        self.cells = {c.id: c for c in cells}
        self.baseline_score = float(baseline_score)
        self.max_parallelism = int(max_parallelism)
        self.trace_id = trace_id
        self.info = dict(info or {})
        depth = {}
        for c in self.cells.values():
            depth.setdefault(c.branch, set()).add(c.attempt)
        for b, attempts in depth.items():
            if attempts != set(range(len(attempts))):
                raise ValueError(f"branch {b} is not a chain from attempt 0: {sorted(attempts)}")
        if len({c.seq for c in self.cells.values()}) != len(self.cells):
            raise ValueError("cell seq numbers must be unique")
        derived = (max(depth, default=-1) + 1, max((len(a) for a in depth.values()), default=0) - 1)
        # grid = (branch_count, refine_count) of the live plan; recorded cells never exceed it
        self.grid = tuple(grid) if grid is not None else derived
        if derived[0] > self.grid[0] or derived[1] > self.grid[1]:
            raise ValueError(f"cells exceed the recorded grid {self.grid}")

    def __len__(self) -> int:
        return len(self.cells)

    def cell(self, branch: int, attempt: int) -> Cell | None:
        return self.cells.get(cell_id(branch, attempt))

    def ceiling(self) -> float:
        """Best successful score recorded (the trace's known ceiling), or the baseline."""
        scores = [c.score for c in self.cells.values() if c.success and c.score is not None]
        return max(scores, default=self.baseline_score)

    def to_dict(self) -> dict:
        return {
            "format": TRACE_FORMAT,
            "trace_id": self.trace_id,
            "baseline_score": self.baseline_score,
            "max_parallelism": self.max_parallelism,
            "grid": {"branch_count": self.grid[0], "refine_count": self.grid[1]},
            "info": self.info,
            "cells": [
                dataclasses.asdict(c) for c in sorted(self.cells.values(), key=lambda c: c.seq)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Trace:
        if d.get("format") != TRACE_FORMAT:
            raise ValueError(f"not a {TRACE_FORMAT} document")
        g = d["grid"]
        return cls(
            (Cell(**c) for c in d["cells"]),
            d["baseline_score"],
            d["max_parallelism"],
            d.get("trace_id", ""),
            (g["branch_count"], g["refine_count"]),
            d.get("info"),
        )

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    @classmethod
    def load(cls, path: str) -> Trace:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def observation_for(cell: Cell, parent: Cell | None, baseline: float) -> Observation:
    delta_base = delta_parent = None
    if cell.success and cell.score is not None:
        delta_base = cell.score - baseline
        if parent is None:
            delta_parent = delta_base  # attempt 0's parent is the root (baseline) workspace
        elif parent.success and parent.score is not None:
            delta_parent = cell.score - parent.score
    return Observation(
        cell.id,
        cell.branch,
        cell.attempt,
        cell.score,
        cell.evaluated,
        cell.valid,
        cell.fail_class,
        cell.error,
        delta_base,
        delta_parent,
        cell.n_valid,
        cell.n_total,
    )


class Question:
    """The decision interface shared by replay and live execution (paper Sec. 3).

    Subclasses supply ``_cell(branch, attempt)`` for cells that exist (replay:
    recorded; live: inside the planned grid) and ``_execute(cells)``, which
    returns their Observations. Legality, batching and bookkeeping live here.
    """

    def __init__(
        self,
        baseline_score: float,
        max_parallelism: int,
        branch_count: int,
        refine_count: int,
        max_rounds: int | None = None,
        record_episode: bool = False,
    ):
        if max_parallelism < 1:
            raise ValueError("max_parallelism must be >= 1")
        self.baseline_score = float(baseline_score)
        self.max_parallelism = int(max_parallelism)
        self.branch_count = int(branch_count)
        self.refine_count = int(refine_count)
        self.max_rounds = max_rounds  # K1 online, K2 in replay
        self.record_episode = record_episode
        self.reset()

    # -- subclass hooks -------------------------------------------------------
    def _cell(self, branch: int, attempt: int) -> CellMeta | None:
        raise NotImplementedError

    def _execute(self, metas: list) -> list:
        raise NotImplementedError

    # -- policy-facing API (Listing 2, lines 30-38) ---------------------------
    def reset(self) -> None:
        self._revealed = {}
        self._depth = {}
        self._seen = set()
        self.budget_spent = 0
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0
        self.best_so_far = None
        self.episode = []

    def observed(self) -> dict:
        return dict(self._revealed)

    def _rounds_left(self) -> bool:
        return self.max_rounds is None or self.decision_rounds < self.max_rounds

    def _in_grid(self, branch: int, attempt: int) -> bool:
        return 0 <= branch < self.branch_count and 0 <= attempt <= self.refine_count

    def _root_metas(self) -> list:
        metas = []
        for b in range(self.branch_count):
            if b not in self._depth:
                m = self._cell(b, 0)
                if m is not None:
                    metas.append(m)
        return sorted(metas, key=lambda m: (m.seq, m.branch))

    def legal_roots(self) -> list:
        return [m.cell_id for m in self._root_metas()] if self._rounds_left() else []

    def legal_actions(self) -> list:
        if not self._rounds_left():
            return []
        out = self.legal_roots()
        for b, depth in sorted(self._depth.items()):
            if self._in_grid(b, depth):
                m = self._cell(b, depth)
                if m is not None:
                    out.append(m.cell_id)
        return out

    def opened_branches(self) -> list:
        return sorted(self._depth)

    def meta(self, cid: str) -> CellMeta:
        if cid in self._seen or cid in self.legal_actions():
            return self._meta_by_id(cid)
        raise KeyError(f"{cid} is neither revealed nor legal")

    def probe_batch(self, cells, on_reveal: Callable | None = None) -> list:
        cells = list(cells)
        if not cells:
            raise IllegalBatch("empty batch: stop by not probing")
        if len(set(cells)) != len(cells):
            raise IllegalBatch(f"duplicate cells in {cells}")
        if len(cells) > self.max_parallelism:
            raise IllegalBatch(f"{len(cells)} cells > max_parallelism {self.max_parallelism}")
        legal = set(self.legal_actions())
        bad = [c for c in cells if c not in legal]
        if bad:
            raise IllegalBatch(f"not legal before the call: {bad}")
        # all cells legal beforehand => no parent/child pair can share a batch
        prefix = self._prefix_summary() if self.record_episode else None
        self.decision_rounds += 1
        self.effective_sequential_rounds += math.ceil(len(cells) / self.max_parallelism)
        metas = [self._meta_by_id(c) for c in cells]
        for m in metas:
            self._depth[m.branch] = m.attempt + 1
        observations = self._execute(metas)
        for obs in observations:
            self._revealed[obs.cell_id] = obs
            self._seen.add(obs.cell_id)
            self.budget_spent += 1
            if (
                obs.evaluated
                and obs.error is None
                and obs.fail_class == OK
                and obs.score is not None
                and (self.best_so_far is None or obs.score > self.best_so_far)
            ):
                self.best_so_far = obs.score
            if on_reveal is not None:
                on_reveal(obs)
        if self.record_episode:
            self.episode.append(
                {
                    "round": self.decision_rounds,
                    "prefix": prefix,
                    "batch": cells,
                    "revealed": [
                        {
                            "cell": o.cell_id,
                            "score": o.score,
                            "fail_class": o.fail_class,
                            "delta_vs_parent": o.delta_vs_parent,
                        }
                        for o in observations
                    ],
                }
            )
        return observations

    # -- helpers --------------------------------------------------------------
    def _meta_by_id(self, cid: str) -> CellMeta:
        b, a = (int(x) for x in cid[1:].split("a"))
        m = self._cell(b, a) if self._in_grid(b, a) else None
        if m is None:
            raise KeyError(cid)
        return m

    def _prefix_summary(self) -> dict:
        return {
            "opened": len(self._depth),
            "revealed": len(self._revealed),
            "best": self.best_so_far,
            "legal": len(self.legal_actions()),
        }


class ReplayQuestion(Question):
    """Replay over a frozen trace: probing reveals recorded outcomes, nothing runs.

    ``branch_count``/``refine_count`` restrict the replay to a planned grid
    (clipped to the trace's support); ``max_rounds`` is the paper's K2.
    """

    def __init__(
        self,
        trace: Trace,
        max_parallelism: int | None = None,
        branch_count: int | None = None,
        refine_count: int | None = None,
        max_rounds: int | None = None,
        record_episode: bool = False,
    ):
        self.trace = trace
        tb, tr = trace.grid
        self.out_of_support = (branch_count is not None and branch_count > tb) or (
            refine_count is not None and refine_count > tr
        )
        super().__init__(
            trace.baseline_score,
            max_parallelism or trace.max_parallelism,
            tb if branch_count is None else min(branch_count, tb),
            tr if refine_count is None else min(refine_count, tr),
            max_rounds,
            record_episode,
        )

    def _cell(self, branch: int, attempt: int) -> CellMeta | None:
        c = self.trace.cell(branch, attempt)
        if c is None:
            return None
        return CellMeta(c.id, c.branch, c.attempt, c.parent_id, c.seq, dict(c.tags))

    def _execute(self, metas: list) -> list:
        out = []
        for m in metas:
            c = self.trace.cells[m.cell_id]
            parent = self.trace.cells.get(c.parent_id) if c.parent_id else None
            out.append(observation_for(c, parent, self.trace.baseline_score))
        return out
