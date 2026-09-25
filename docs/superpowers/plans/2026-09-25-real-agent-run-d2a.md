# Real-Agent Lasso Run (Track D2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the reconstructed Dream-RSI loop once on SimpleTES's Lasso task with real coding agents on this Mac. Turn the run into committed, license-safe evidence that answers track D2's four questions with counts.

**Architecture:** There are three code tasks, then three tasks the controller runs itself.

- Code tasks:
  - an Eigen include hook in the SimpleTES adapter and both Lasso scripts;
  - a launch record that the runner appends to the workdir;
  - `scripts/report_run.py`, which reads a workdir read-only and writes `report.md` and `report.json`.
- Controller-run tasks:
  - the host smoke test;
  - the isolated-agent pre-flight and the run itself;
  - the report and the evidence commit.

No loop, scoring or policy-API code changes.

**Tech Stack:** Python 3.10+ standard library (the `see` package), pytest, ruff and pyright. SimpleTES's Lasso evaluator (AGPL; cloned, not vendored), MacPorts gcc 13.4.0 with OpenMP, and Eigen 3.4.1. Gemini CLI for discovery and Claude CLI 2.1.282 for policy development. The launch record captures the exact versions at run time.

**Spec:** `docs/superpowers/specs/2026-09-25-real-agent-run-d2a-design.md`. It is amended in this plan's commit (new section 11), and executors read both.

## Global Constraints

- The gate runs from `reconstruction/` inside its venv. Every step must pass:
  - `ruff format --check .`;
  - `ruff check .`;
  - `pyright` (basic mode, 0 errors);
  - `python tools/extract_listings.py --check`;
  - `python -m pytest -q --junitxml=report.xml` with zero skips;
  - `python tools/check_junit.py report.xml --expect N`, where N is the count pinned in `.github/workflows/ci.yml`. N moves 96 → 101 → 106 → 118 across Tasks 1–3 and stays at 118 after.
- No `xfail`, `skip`, `# noqa` or `# type: ignore`.
- Nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes. Nor do `see/loop.py`, `see/live.py`, `see/world.py` or `see/objective.py`: they are read, never edited. No workdir directory is renamed, and the agent callable's signature is unchanged.
- Every paper-silent choice gets a GAPS §3 row with its "How to change" cell and a pin.
- Commits:
  - one commit per task (Task 5 changes nothing in the repository and makes none);
  - a plain imperative subject;
  - the message is written to a file and committed with `git commit -F <file>`;
  - check `git diff --stat` before every commit;
  - **never add a `Co-Authored-By:` line, or any other line naming an agent, to a commit message.**
- SimpleTES is AGPL and stays unvendored. No attempt program (a modified SimpleTES seed) ever enters the repository, and neither does anything under `generated/`.
- Symbol work goes through Serena first (`.claude/CLAUDE.md`, "Code navigation: Serena first"). Load the `mcp__serena__*` tools via ToolSearch; their paths are relative to the repo root.
- Tasks 4, 5 and 6 are run by the controller (the session) itself, never dispatched to a subagent:
  - Task 4 spends host CPU and Task 5 spends API budget; each starts only after the owner's explicit go.
  - Task 6 has an owner decision in the middle (the evidence scan).
  - Tasks 4 and 6 each end in a commit that gets its task review.
  - Task 5 changes nothing in the repository: it closes with a ledger entry and no task review.
- Test names state the claim or outcome they protect. Tests that run the loop request the `stub_prompts` fixture.

## Rulings made while planning

Each ruling is recorded in the spec's new section 11, in this plan's commit.

1. **The workdir lives outside the repository and outside the home directory**, at `/Users/Shared/dream-rsi-runs/d2a-lasso`. The run directory is `chmod 700`, because it holds a copy of the Gemini login.
   - Why outside the repository: the discovery agent has a shell (`--yolo`). Under `reconstruction/runs/` it could reach `generated/lasso_path_dream_rsi.py` (the paper's final answer), tracked files, and this repo's `.gemini/GEMINI.md`.
   - Why outside the home directory: both CLIs read context files from the cwd's ancestors, and `/Users/controlroom/GEMINI.md` exists. `/Users/Shared`, `/Users` and `/` hold no such file, and the pre-flight re-checks that.
   - Cost if wrong: longer paths.
2. **Both agents run isolated from the owner's global CLI configuration.** This is the owner's decision (2026-09-25).
   - Why: the owner's Gemini settings disable yolo mode and load the superpowers and context-mode extensions and hooks. The owner's Claude setup loads a global CLAUDE.md, plugins, and hooks that write into the owner's notes. A headless call cannot pass an approval gate, so an attempt could end with its program untouched, which is exactly the count D2a measures.
   - Gemini runs under a dedicated `HOME` that holds only its login, the owner's auth type and the owner's `billing` block. Its model is pinned to the owner's current `auto-gemini-2.5`.
   - Claude runs with `--setting-sources project,local --strict-mcp-config`. If the pre-flight shows user instructions still load, it runs under a dedicated `CLAUDE_CONFIG_DIR` instead.
   - Both are JSON argv, which the runner already accepts, so no code changes.
3. **The runner appends a launch record** (`launches.jsonl`, Task 2).
   - Why: the manifests carry neither the caps nor the program's file name, and nothing records the agents' argv and versions or the host toolchain at run time. The report reads the caps and the program name from the record's last line.
   - Cost if wrong: one more file in the workdir.
4. **`report_run.py` imports `see.live.node_dirname`** as well as `see.world`, so the attempt directory name has one definition. The script stays read-only over loop code.
5. **The evidence subset gains `launches.jsonl` and each version's `policy_execution_traces.jsonl`.** The latter is the per-episode source of the out-of-support and empty-batch counts; `beta_sweep.json` keeps only three errors and one `any()` flag.
   - `report_run.py --copy-evidence` copies the subset to `evidence/d2a-lasso/workdir/` and withholds any file that quotes `CPP_CODE` (a pasted program).
   - `reconstruction/evidence/` is excluded from ruff and from the whitespace hooks, so evidence stays byte for byte as written.
6. **A restart moves the guard-named directories aside** to `/Users/Shared/dream-rsi-runs/d2a-lasso-restart1/`, keeping their relative paths, instead of deleting them. The partial iteration is evidence too. The loop never sees the moved directories again, so nothing is merged.
7. **The report's test run uses a 2 x 1 fallback grid**: two branches of two attempts, reading the spec's "2 x 2" as cells. Its hard caps are 3 x 2, so the wider version asks for 3 x 1.

## File structure

- `reconstruction/see/tasks.py`: gains `link_eigen`; `simpletes_task` gains a keyword-only `eigen_include`.
- `reconstruction/scripts/verify_lasso.py`: gains `--eigen-include`; `load_evaluator` becomes `load_evaluator(..., eigen_include=None)`.
- `reconstruction/scripts/run_dream_rsi.py`:
  - `--eigen-include`;
  - `build_parser`, `first_line`, `cli_version`, `eigen_version`, `host_facts`, `launch_record` and `record_launch`;
  - an appended line in `<workdir>/launches.jsonl` per launch.
- `reconstruction/scripts/report_run.py`: new. It holds the evidence report and the evidence copy.
- New pins: `reconstruction/tests/test_tasks.py` (5), `tests/test_run_dream_rsi.py` (5) and `tests/test_report_run.py` (12).
- `reconstruction/evidence/d2a-lasso/`: new (Tasks 4 and 6). It holds `host.json`, `report.md`, `report.json` and `workdir/…`.
- Docs: `reconstruction/GAPS.md`, `reconstruction/README.md` and `.claude/CLAUDE.md`. Config: `.github/workflows/ci.yml`, `reconstruction/pyproject.toml` and `.pre-commit-config.yaml`.

Every code block in Tasks 1–3 was run in a throwaway worktree before this plan was written:

- the full suite: 118 passed, no skips;
- ruff, format and pyright: clean;
- `check_junit`: `junit: 118 tests, no skips`.

Each edit's "replace" text occurs exactly once in the file at the start of its task.

---

### Task 1: Eigen include hook for the SimpleTES Lasso evaluator

**Files:**
- Modify: `reconstruction/see/tasks.py` (new `link_eigen` before `simpletes_task`; `simpletes_task` signature and body)
- Modify: `reconstruction/scripts/verify_lasso.py` (docstring, import, `load_evaluator`, `--eigen-include`)
- Modify: `reconstruction/scripts/run_dream_rsi.py` (`--eigen-include`, the `simpletes_task` call)
- Create: `reconstruction/tests/test_tasks.py`
- Modify: `reconstruction/GAPS.md` (§2 SimpleTES bullet; §3 new last row "Lasso host toolchain")
- Modify: `reconstruction/README.md` (Lasso check paragraph; run paragraph)
- Modify: `.claude/CLAUDE.md` (the "Paper-task scripts need" bullet)
- Modify: `.github/workflows/ci.yml` (`--expect 96` → `--expect 101`)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `see.tasks.link_eigen(task_dir: str, eigen_include: str) -> None`. It raises `FileNotFoundError` when `eigen_include` holds no `Eigen/`, and `FileExistsError` when `<task_dir>/eigen` exists and resolves elsewhere.
  - `see.tasks.simpletes_task(simpletes_dir, name, workdir, *, eigen_include: str | None = None) -> TaskSpec`.
  - The runner flag `--eigen-include` (parsed as `a.eigen_include`).
  - `verify_lasso.load_evaluator(simpletes_dir, workdir, eigen_include=None)` and its flag `--eigen-include`.

Why: SimpleTES's Lasso evaluator compiles with `g++ ... -I<its own dir>/eigen` when that directory exists, else with `-I/usr/include/eigen3`, a Linux path that does not exist on macOS. The adapter deliberately does not copy SimpleTES's vendored `eigen/` (it lacks `Eigen/Core`). A symlink at exactly `<task dir>/eigen` therefore puts a host Eigen root where the evaluator looks first, with no change to SimpleTES.

- [ ] **Step 1: Write the failing tests.** Create `reconstruction/tests/test_tasks.py`:

````python
"""Pins for the SimpleTES adapter's host hook (GAPS.md §3, "Lasso host toolchain").

A fake SimpleTES task stands in for the AGPL checkout: its evaluator reports the directory the
real one would try first for Eigen (``<its own dir>/eigen``), so no compiler is involved.
"""

import os

import pytest

from see.loader import load_module_from_path
from see.tasks import link_eigen, simpletes_task

VERIFY_LASSO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "verify_lasso.py"
)
FAKE_EVALUATOR = """import os

TASK_DIR = os.path.dirname(os.path.abspath(__file__))


def evaluate(path):
    eigen = os.path.join(TASK_DIR, "eigen")  # the real evaluator's first -I candidate
    found = os.path.realpath(eigen) if os.path.isdir(eigen) else None
    return {"combined_score": 0.0, "eigen": found}
"""


def _fake_simpletes(root) -> str:
    task = root / "datasets" / "numerical_tasks" / "lasso_path"
    (task / "eigen" / "Eigen").mkdir(parents=True)  # the vendored tree, which lacks Eigen/Core
    (task / "evaluator.py").write_text(FAKE_EVALUATOR)
    (task / "init_program.py").write_text('CPP_CODE = ""\n')
    (task / "lasso_path.txt").write_text("Compute the Lasso path.\n")
    return str(root)


def _host_eigen(root) -> str:
    (root / "Eigen").mkdir(parents=True)
    return str(root)


def test_without_eigen_include_the_evaluator_falls_back_to_the_system_headers(tmp_path):
    """The vendored tree is not copied, so nothing sits where the evaluator looks first."""
    task = simpletes_task(_fake_simpletes(tmp_path / "st"), "lasso_path", str(tmp_path / "w"))
    assert task.evaluate("unused")["eigen"] is None


def test_eigen_include_is_where_the_evaluator_looks_first(tmp_path):
    host = _host_eigen(tmp_path / "host" / "eigen3")
    task = simpletes_task(
        _fake_simpletes(tmp_path / "st"), "lasso_path", str(tmp_path / "w"), eigen_include=host
    )
    assert task.evaluate("unused")["eigen"] == os.path.realpath(host)


def test_a_restart_reuses_its_eigen_link_and_refuses_a_different_one(tmp_path):
    simpletes, workdir = _fake_simpletes(tmp_path / "st"), str(tmp_path / "w")
    first, other = _host_eigen(tmp_path / "a"), _host_eigen(tmp_path / "b")
    simpletes_task(simpletes, "lasso_path", workdir, eigen_include=first)
    again = simpletes_task(simpletes, "lasso_path", workdir, eigen_include=first)  # the restart
    assert again.evaluate("unused")["eigen"] == os.path.realpath(first)
    with pytest.raises(FileExistsError, match="does not point at"):
        simpletes_task(simpletes, "lasso_path", workdir, eigen_include=other)


def test_an_eigen_root_without_eigen_headers_is_refused_before_any_run(tmp_path):
    """/opt/local/include (one level too high) would compile nothing; say so before the run."""
    (tmp_path / "include").mkdir()
    with pytest.raises(FileNotFoundError, match="has no Eigen/ directory"):
        link_eigen(str(tmp_path / "task"), str(tmp_path / "include"))


def test_verify_lasso_points_its_evaluator_copy_at_the_host_eigen(tmp_path):
    verify_lasso = load_module_from_path("verify_lasso_under_test", VERIFY_LASSO)
    host = _host_eigen(tmp_path / "eigen3")
    (tmp_path / "copy").mkdir()
    ev = verify_lasso.load_evaluator(_fake_simpletes(tmp_path / "st"), str(tmp_path / "copy"), host)
    assert ev.evaluate("unused")["eigen"] == os.path.realpath(host)
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_tasks.py`
Expected: a collection error, `ImportError: cannot import name 'link_eigen' from 'see.tasks'`.

- [ ] **Step 3: Add the hook to the adapter**

In `reconstruction/see/tasks.py`, replace:

````python
def simpletes_task(simpletes_dir: str, name: str, workdir: str) -> TaskSpec:
````

with:

````python
def link_eigen(task_dir: str, eigen_include: str) -> None:
    """Point SimpleTES's first Eigen lookup, ``<task dir>/eigen``, at a host Eigen root.

    The Lasso evaluator compiles with ``-I<its own dir>/eigen`` when that directory exists and
    with ``-I/usr/include/eigen3``, a Linux path, otherwise. ``eigen_include`` is the directory
    that holds ``Eigen/`` (``/opt/local/include/eigen3`` under MacPorts). A restart finds the
    link it made; a link to anything else is refused.
    """
    target = os.path.abspath(eigen_include)
    if not os.path.isdir(os.path.join(target, "Eigen")):
        raise FileNotFoundError(f"{target} has no Eigen/ directory: pass the root that holds it")
    link = os.path.join(task_dir, "eigen")
    if os.path.lexists(link):
        if os.path.realpath(link) != os.path.realpath(target):
            raise FileExistsError(f"{link} exists and does not point at {target}")
        return
    os.symlink(target, link)


def simpletes_task(
    simpletes_dir: str, name: str, workdir: str, *, eigen_include: str | None = None
) -> TaskSpec:
````

In `reconstruction/see/tasks.py`, replace:

````python
        # SimpleTES's vendored Eigen lacks Eigen/Core; without the copy the Lasso
        # evaluator falls back to the system headers (libeigen3-dev).
        shutil.copytree(src, local, ignore=shutil.ignore_patterns("eigen", "__pycache__"))
````

with:

````python
        # SimpleTES's vendored Eigen lacks Eigen/Core, so it is not copied: the Lasso
        # evaluator then uses eigen_include when given, else the system headers (libeigen3-dev).
        shutil.copytree(src, local, ignore=shutil.ignore_patterns("eigen", "__pycache__"))
    if eigen_include is not None:
        link_eigen(local, eigen_include)
````

- [ ] **Step 4: Add `--eigen-include` to `verify_lasso.py`**

In `reconstruction/scripts/verify_lasso.py`, replace:

````python
The evaluator is copied into a temp dir before import because the Eigen tree
vendored in SimpleTES lacks Eigen/Core (its .gitignore drops it); from the
temp dir the evaluator falls back to -I/usr/include/eigen3 (libeigen3-dev).
````

with:

````python
The evaluator is copied into a temp dir before import because the Eigen tree
vendored in SimpleTES lacks Eigen/Core (its .gitignore drops it); from the
temp dir the evaluator falls back to -I/usr/include/eigen3 (libeigen3-dev),
or uses --eigen-include, the directory holding Eigen/ (MacPorts:
/opt/local/include/eigen3). --eigen-include covers the search score only.
````

In `reconstruction/scripts/verify_lasso.py`, replace:

````python
from see.loader import load_module_from_path
````

with:

````python
from see.loader import load_module_from_path
from see.tasks import link_eigen
````

In `reconstruction/scripts/verify_lasso.py`, replace:

````python
def load_evaluator(simpletes_dir, workdir):
    src = os.path.join(simpletes_dir, "datasets", "numerical_tasks", "lasso_path", "evaluator.py")
    dst = os.path.join(workdir, "lasso_evaluator.py")
    shutil.copy(src, dst)
````

with:

````python
def load_evaluator(simpletes_dir, workdir, eigen_include=None):
    src = os.path.join(simpletes_dir, "datasets", "numerical_tasks", "lasso_path", "evaluator.py")
    dst = os.path.join(workdir, "lasso_evaluator.py")
    shutil.copy(src, dst)
    if eigen_include is not None:
        link_eigen(workdir, eigen_include)  # the evaluator looks in its own directory first
````

In `reconstruction/scripts/verify_lasso.py`, replace:

````python
    ap.add_argument("--json", help="write the per-problem results here")
````

with:

````python
    ap.add_argument("--json", help="write the per-problem results here")
    ap.add_argument(
        "--eigen-include",
        help="directory holding Eigen/, e.g. /opt/local/include/eigen3 "
        "(default: the evaluator's /usr/include/eigen3)",
    )
````

In `reconstruction/scripts/verify_lasso.py`, replace:

````python
        ev = load_evaluator(args.simpletes, workdir)
````

with:

````python
        ev = load_evaluator(args.simpletes, workdir, args.eigen_include)
````

- [ ] **Step 5: Add `--eigen-include` to the runner**

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
    ap.add_argument("--objective", choices=OBJECTIVES, default="pareto")
````

with:

````python
    ap.add_argument("--objective", choices=OBJECTIVES, default="pareto")
    ap.add_argument(
        "--eigen-include",
        help="directory holding Eigen/ for the Lasso evaluator, e.g. /opt/local/include/eigen3 "
        "(default: the evaluator's /usr/include/eigen3)",
    )
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
        simpletes_task(a.simpletes, a.task, a.workdir),
````

with:

````python
        simpletes_task(a.simpletes, a.task, a.workdir, eigen_include=a.eigen_include),
````

- [ ] **Step 6: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_tasks.py`
Expected: `5 passed`.

- [ ] **Step 7: Ledger, docs and the CI count**

In `reconstruction/GAPS.md`, replace:

````markdown
`.gitignore` drops it), so compilation needs system Eigen.
````

with:

````markdown
`.gitignore` drops it), so compilation needs system Eigen or a host Eigen root passed as
  `--eigen-include` (§3, "Lasso host toolchain").
````

In `reconstruction/GAPS.md`, replace:

````markdown
the floor is fixed in `see/loop.py` (`offline`) |

## 4. Places where the paper contradicts itself
````

with:

````markdown
the floor is fixed in `see/loop.py` (`offline`) |
| Lasso host toolchain | not discussed; SimpleTES's evaluator compiles with a hardcoded `g++ -O3 -march=native -std=c++17`, with `-I<task dir>/eigen` when that directory exists, else `-I/usr/include/eigen3` | the adapter does not copy SimpleTES's vendored `eigen/` (it lacks `Eigen/Core`); `eigen_include` makes `task/src/eigen` a symlink to a host Eigen root, the directory the evaluator tries first, refusing a root without `Eigen/` and an existing link that points elsewhere (a restart reuses its own link); the compiler is whatever `g++` resolves to on `PATH`: on macOS Apple's `g++` is clang without OpenMP, so MacPorts gcc is selected with `port select` (pinned in `test_tasks.py::test_eigen_include_is_where_the_evaluator_looks_first` and `::test_a_restart_reuses_its_eigen_link_and_refuses_a_different_one`) | `simpletes_task(eigen_include=)`; `run_dream_rsi.py --eigen-include`; `verify_lasso.py --eigen-include` (search score only); the compiler: `PATH` |

## 4. Places where the paper contradicts itself
````

In `reconstruction/README.md`, replace:

````markdown
Check the paper's Lasso solver against SimpleTES's own evaluator (needs g++,
OpenMP and `libeigen3-dev`):

```bash
git clone --depth 1 https://github.com/wq-will/SimpleTES ../SimpleTES
python scripts/verify_lasso.py --simpletes ../SimpleTES --repeats 2
````

with:

````markdown
Check the paper's Lasso solver against SimpleTES's own evaluator. It needs a `g++` with
OpenMP first on `PATH` and Eigen 3: on Linux, `g++` and `libeigen3-dev`; on macOS, where
Apple's `g++` is clang without OpenMP, MacPorts `gcc13` (`sudo port select --set gcc mp-gcc13`)
and `eigen3`, passed as `--eigen-include /opt/local/include/eigen3` (the directory holding
`Eigen/`; it covers the search score, not `--downstream`):

```bash
git clone --depth 1 https://github.com/wq-will/SimpleTES ../SimpleTES
python scripts/verify_lasso.py --simpletes ../SimpleTES --repeats 2  # macOS: --eigen-include DIR
````

In `reconstruction/README.md`, replace:

````markdown
budget: about 110 discovery calls per round at the paper's 3.1-Pro setting.
````

with:

````markdown
budget: about 110 discovery calls per round at the paper's 3.1-Pro setting. On macOS add
`--eigen-include` as above.
````

In `.claude/CLAUDE.md`, replace:

````markdown
  plus g++, OpenMP and system Eigen (`libeigen3-dev`). `scripts/verify_lasso.py` scores Listing 3
````

with:

````markdown
  plus a `g++` with OpenMP first on `PATH` and Eigen 3 (Linux: `libeigen3-dev`; macOS: MacPorts
  `gcc13` selected with `sudo port select --set gcc mp-gcc13`, and `eigen3` passed as
  `--eigen-include /opt/local/include/eigen3`). `scripts/verify_lasso.py` scores Listing 3
````

In `.github/workflows/ci.yml`, replace:

````yaml
--expect 96
````

with:

````yaml
--expect 101
````

- [ ] **Step 8: Run the gate**

Run the Global Constraints gate. Expected: `101 passed`, `junit: 101 tests, no skips`, ruff and pyright clean. Then run `python scripts/run_dream_rsi.py --help` and `python scripts/verify_lasso.py --help`: each must list `--eigen-include`. The runner's wiring has no unit test; the smoke test (Task 4) and the run (Task 5) exercise it.

- [ ] **Step 9: Commit**

Run `git diff --stat`. It should show exactly the 8 files above. Then:

```bash
cat > /tmp/d2a-t1-msg.txt <<'EOF'
Add an Eigen include hook for the SimpleTES Lasso evaluator

SimpleTES's evaluator tries <task dir>/eigen first and falls back to
/usr/include/eigen3, a Linux path. simpletes_task(eigen_include=) and
the --eigen-include flag of both Lasso scripts link a host Eigen root
there, refusing a root without Eigen/ and a link that points elsewhere.
GAPS §3 records the host toolchain; five pins cover the hook without a
compiler.
EOF
git add reconstruction/see/tasks.py reconstruction/scripts/verify_lasso.py \
  reconstruction/scripts/run_dream_rsi.py reconstruction/tests/test_tasks.py \
  reconstruction/GAPS.md reconstruction/README.md .claude/CLAUDE.md .github/workflows/ci.yml
git commit -F /tmp/d2a-t1-msg.txt
```

---

### Task 2: A launch record in the runner

**Files:**
- Modify: `reconstruction/scripts/run_dream_rsi.py` (docstring, imports, new functions, `build_parser`, `main`)
- Create: `reconstruction/tests/test_run_dream_rsi.py`
- Modify: `reconstruction/GAPS.md` (§2 Gemini CLI bullet)
- Modify: `.claude/CLAUDE.md` (the same bullet as Task 1)
- Modify: `.github/workflows/ci.yml` (`--expect 101` → `--expect 106`)

**Interfaces:**
- Consumes: from Task 1, `--eigen-include` (`a.eigen_include`) and `simpletes_task(..., eigen_include=)`.
- Produces: `<workdir>/launches.jsonl`, one JSON object per launch with these keys:
  - `started`: a float;
  - `argv`: a list;
  - `task`: `{"name", "eval_program"}`;
  - `config`: `{"iterations", "versions", "max_parallelism", "fallback_grid": [B, R], "hard_max_grid": [B, R], "objective", "agent_timeout"}`;
  - `agents`: `{"discovery" | "policy": {"argv", "version"}}`;
  - `host`: `{"platform", "cpu", "cpus", "python", "compiler", "eigen", "simpletes_commit"}`.

  It also produces these runner functions: `build_parser()`, `first_line(cmd)`, `cli_version(argv)`, `eigen_version(include_dir)`, `host_facts(simpletes, eigen_include)`, `launch_record(a, argv, task, discovery, policy)` and `record_launch(workdir, record)`. Task 3 reads the record's last line.

Why: a manifest records the grid it ran but not the caps in force or the program's file name. Nothing records which agent argv, CLI versions, compiler, Eigen or SimpleTES commit a run used. The record is written before the first iteration, and a restart appends a second line rather than replacing the first.

- [ ] **Step 1: Write the failing tests.** Create `reconstruction/tests/test_run_dream_rsi.py`:

````python
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


def test_an_env_wrapped_agent_reports_the_version_of_the_cli_it_wraps():
    """An isolated agent's argv starts with ``env HOME=...``: the version is the CLI's."""
    version = runner.cli_version(["env", "HOME=/nonexistent", "python3", "-c", "{prompt}"])
    assert version is not None and version.startswith("Python 3")


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
````

`test_the_host_facts_name_the_simpletes_checkout_commit` uses this repository's own checkout as a stand-in for a SimpleTES clone. CI's checkout is a git repository too.

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_run_dream_rsi.py`
Expected: `5 failed`, each with an `AttributeError` such as `AttributeError: module 'run_dream_rsi_under_test' has no attribute 'build_parser'`.

- [ ] **Step 3: Implement the record**

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
Agents are CLI presets from see.live.AGENT_PRESETS ("gemini", "claude") or a
JSON argv list containing "{prompt}". Defaults mirror the paper's
Gemini-3.1-Pro setting (10 workers, 10 branches x 11 attempts, 5 rounds);
M, K1, K2, lambda and the beta grid are not given in the paper.
"""

import argparse
import json
import os

from see.live import CommandAgent
````

with:

````python
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
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
def main(argv=None):
    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
````

with:

````python
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
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
        "(default: the evaluator's /usr/include/eigen3)",
    )
    a = ap.parse_args(argv)
    os.makedirs(a.workdir, exist_ok=True)
````

with:

````python
        "(default: the evaluator's /usr/include/eigen3)",
    )
    return ap


def main(argv=None):
    install_signal_handlers()  # SIGTERM freezes and refuses like Ctrl-C
    a = build_parser().parse_args(argv)
    os.makedirs(a.workdir, exist_ok=True)
````

In `reconstruction/scripts/run_dream_rsi.py`, replace:

````python
    loop = DreamRSI(
        cfg,
        simpletes_task(a.simpletes, a.task, a.workdir, eigen_include=a.eigen_include),
        agent(a.discovery_agent, a.agent_timeout),
        agent(a.policy_agent, a.agent_timeout),
    )
````

with:

````python
    task = simpletes_task(a.simpletes, a.task, a.workdir, eigen_include=a.eigen_include)
    discovery = agent(a.discovery_agent, a.agent_timeout)
    policy = agent(a.policy_agent, a.agent_timeout)
    record_launch(
        a.workdir,
        launch_record(a, sys.argv if argv is None else list(argv), task, discovery, policy),
    )
    loop = DreamRSI(cfg, task, discovery, policy)
````

- [ ] **Step 4: Run the tests to see them pass**

Run: `python -m pytest -q tests/test_run_dream_rsi.py`
Expected: `5 passed`.

- [ ] **Step 5: Ledger, docs and the CI count**

In `reconstruction/GAPS.md`, replace:

````markdown
- **Gemini CLI** is the discovery agent. Its invocation flags, tools, timeouts and sampling
  settings are not given.
````

with:

````markdown
- **Gemini CLI** is the discovery agent. Its invocation flags, tools, timeouts and sampling
  settings are not given. `scripts/run_dream_rsi.py` records each launch's agent argv and CLI
  version in `<workdir>/launches.jsonl`.
````

In `.claude/CLAUDE.md`, replace:

````markdown
  spends real API budget (~110 discovery calls per round at the paper's default grid).
````

with:

````markdown
  spends real API budget (~110 discovery calls per round at the paper's default grid); each
  launch appends its caps, agent argv and CLI versions, and host toolchain to
  `<workdir>/launches.jsonl`.
````

In `.github/workflows/ci.yml`, replace:

````yaml
--expect 101
````

with:

````yaml
--expect 106
````

- [ ] **Step 6: Run the gate**

Expected: `106 passed`, `junit: 106 tests, no skips`, ruff and pyright clean.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 5 files above.

```bash
cat > /tmp/d2a-t2-msg.txt <<'EOF'
Record each real-agent launch in the workdir

The manifests carry neither the caps nor the program's file name, and
nothing kept the agents' argv and versions or the host toolchain. Every
run_dream_rsi.py launch now appends them to <workdir>/launches.jsonl
before the first iteration; a restart appends and keeps the first.
EOF
git add reconstruction/scripts/run_dream_rsi.py reconstruction/tests/test_run_dream_rsi.py \
  reconstruction/GAPS.md .claude/CLAUDE.md .github/workflows/ci.yml
git commit -F /tmp/d2a-t2-msg.txt
```

---

### Task 3: The evidence report

**Files:**
- Create: `reconstruction/scripts/report_run.py`
- Create: `reconstruction/tests/test_report_run.py`
- Modify: `reconstruction/pyproject.toml` (`[tool.ruff] extend-exclude`)
- Modify: `.pre-commit-config.yaml` (exclude `reconstruction/evidence/` from four hooks)
- Modify: `reconstruction/README.md` (report paragraph; layout)
- Modify: `.claude/CLAUDE.md` (commands block; conventions)
- Modify: `.github/workflows/ci.yml` (`--expect 106` → `--expect 118`)

**Interfaces:**
- Consumes:
  - From Task 2, the last line of `<workdir>/launches.jsonl`: `task.eval_program`, `config.fallback_grid`, `config.hard_max_grid`, `agents` and `host`.
  - The loop's workdir layout (the `see/loop.py` module docstring):
    - `state.json`, for `log[*].selected`;
    - `runs/iterNNNN/tree/attempt_bBBB_aAAA/{<program>, eval/score.json, error.txt, proposal.md}`;
    - `runs/iterNNNN/partial/`;
    - `trace_pool/iterNNNN/{trace.json, live_cycle_manifest.json, live_episode.jsonl}`;
    - `policy_dev/history/rNNNN_tNN_mM/{method.py, proposal_results/beta_sweep.json, proposal_results/policy_execution_traces.jsonl}`.
  - `see.world.Trace` and `see.live.node_dirname`.
- Produces:
  - The command `python scripts/report_run.py --workdir W --out O [--host-json H] [--archive A] [--copy-evidence]`.
  - `O/report.md` and `O/report.json`. The JSON keys are `workdir` (shown with `~` for the home directory), `launches`, `task`, `agents`, `host`, `caps`, `archive`, `iterations`, `unfrozen_iterations`, `versions`, `out_of_support`, `untouched`, `empty_batches` and `noise`, plus `evidence` when `--copy-evidence` is given.
  - With `--copy-evidence`, the subset under `O/workdir/`.
  - The functions the tests use:
    - `main(argv) -> dict`;
    - `build_report(workdir, host_json=None, archive=None) -> dict`;
    - `resume_source(tree_dir, baseline_dir, program, branch, attempt) -> (path, source attempt or None)`;
    - `noise(host_json) -> dict`;
    - `copy_evidence(workdir, dest) -> {"copied", "withheld"}`.

What the report counts, and from where:

1. **Per-round call budget.** Each iteration's manifest gives the planned grid, whether the fallback was used, the effective grid, the probes spent and the decision rounds. The launch record gives the caps in force; the paper's figures are 110 and 640. The manifest records only `used_fallback`, not why: the policy may have returned no plan, or a plan outside the caps.
2. **Out-of-support replay.** For each policy version, its `policy_execution_traces.jsonl` gives episodes, clipped episodes, the grids they asked for (a missing plan means the fallback), and the recorded grids of the trees they replayed. `state.json` says whether the version was deployed.
3. **Untouched programs.** For each recorded attempt, the resume source is recomputed by the loop's own rule (`see/live.py:LiveQuestion._resume_from`). The attempt's program is then compared with it byte for byte, on disk, after the run. Identical cases are listed with fail class, timed-out flag, evaluated flag, score and the source's score.
4. **Empty batches.** Episode errors, or the sweep report's errors when no episode file exists, classify each failed version: `empty_batch`, `illegal_batch` or `other`. A manifest's `error` does the same for live batches.

A health table per iteration and per version follows the four sections, then the host facts, then the smoke test's noise.

The test run is one scripted run on the toy task, built in the test. Every expected number in the tests is counted by hand from its `SCRIPT` and the policy alternation. That run was executed and printed before this plan was written.

- [ ] **Step 1: Write the failing tests.** Create `reconstruction/tests/test_report_run.py`:

````python
"""Pins for scripts/report_run.py, the D2a evidence report (spec section 5).

One scripted run on the toy task stands in for the real-agent run: a 2 x 1 fallback grid (two
branches of two attempts), two iterations, three policy versions per iteration, two workers. The
discovery agent adds 0.1 to the program's x except where SCRIPT says otherwise; the policy agent
alternates a version that plans one branch wider than any recorded tree (WIDE) and one that ends
on an empty batch (EMPTY). Every expected number below is counted by hand from that script.
"""

import json
import os

import pytest

from see.live import LiveQuestion
from see.loader import load_module_from_path
from see.loop import DreamRSI, LoopConfig
from see.toy import PROGRAM, ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

RECON = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report_run = load_module_from_path(
    "report_run_under_test", os.path.join(RECON, "scripts", "report_run.py")
)
WIDE = """from see.policies.parallel_refine import ParallelRefine
from see.policy.api import GridPlan

NAME = "WiderRefine"


class WiderRefine(ParallelRefine):
    NAME = NAME

    def plan_grid(self, context):
        return GridPlan(
            context.fallback_branch_count + 1, context.fallback_refine_count, reason="one wider"
        )
"""
EMPTY = """from see.policy.api import LLMDesignedMethod

NAME = "StopsWithAnEmptyBatch"


class StopsWithAnEmptyBatch(LLMDesignedMethod):
    NAME = NAME

    def solve(self, question, budget=None):
        question.reset()
        question.probe_batch([])
"""
SCRIPT = {
    ("iter0001", "attempt_b000_a000"): "untouched",  # identical to the baseline
    ("iter0001", "attempt_b001_a001"): "timeout",  # identical to its parent, agent timed out
    ("iter0002", "attempt_b001_a000"): "delete",  # no program; its child resumes the baseline
    ("iter0002", "attempt_b000_a001"): "paste",  # its proposal quotes a program: withheld
}


class ScriptedDiscovery:
    def __call__(self, prompt, *, cwd, target):
        iteration = os.path.basename(os.path.dirname(os.path.dirname(target)))
        what = SCRIPT.get((iteration, os.path.basename(target)))
        with open(os.path.join(target, "proposal.md"), "w") as f:
            f.write("pasted CPP_CODE = ...\n" if what == "paste" else "one step up\n")
        path = os.path.join(target, PROGRAM)
        if what == "untouched":
            return {"returncode": 0}
        if what == "timeout":
            return {"returncode": None, "timed_out": True, "stderr": "agent timed out after 900s"}
        if what == "delete":
            os.remove(path)
            return {"returncode": 0}
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:
            json.dump({"x": round(x + 0.1, 6)}, f)
        return {"returncode": 0}


class AlternatingPolicy:
    def __init__(self):
        self.calls = 0

    def __call__(self, prompt, *, cwd, target):
        self.calls += 1
        with open(target, "w") as f:
            f.write(WIDE if self.calls % 2 else EMPTY)
        return {"returncode": 0}


def _launch(workdir) -> None:
    """The line scripts/run_dream_rsi.py appends to launches.jsonl, cut to what the report reads."""
    line = {
        "task": {"name": "toy", "eval_program": PROGRAM},
        "config": {"fallback_grid": [2, 1], "hard_max_grid": [3, 2]},
    }
    (workdir / "launches.jsonl").write_text(json.dumps(line) + "\n")


def _config(workdir, **kw) -> LoopConfig:
    return LoopConfig(
        workdir=str(workdir), max_parallelism=2, fallback_grid=(2, 1), hard_max_grid=(3, 2), **kw
    )


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory, stub_prompts):
    root = tmp_path_factory.mktemp("d2a")
    workdir = root / "w"
    workdir.mkdir()
    _launch(workdir)
    cfg = _config(workdir, iterations=2, versions=3)
    DreamRSI(cfg, make_task(str(workdir)), ScriptedDiscovery(), AlternatingPolicy()).run()
    (root / "archive.bin").write_bytes(b"archive")
    out = root / "out"
    args = ["--workdir", str(workdir), "--out", str(out), "--archive", str(root / "archive.bin")]
    return report_run.main([*args, "--copy-evidence"]), out


def test_each_rounds_plan_and_spend_come_from_its_manifest(toy_run):
    report, _ = toy_run
    budget = [
        {k: i[k] for k in ("iteration", "policy_round", "planned_grid", "used_fallback")}
        | {k: i[k] for k in ("effective_grid", "grid_calls", "probes", "decision_rounds")}
        for i in report["iterations"]
    ]
    assert budget == [  # parallel refine plans the fallback, and m0 is deployed both times
        {"iteration": 1, "policy_round": "initial", "planned_grid": [2, 1], "used_fallback": False}
        | {"effective_grid": [2, 1], "grid_calls": 4, "probes": 4, "decision_rounds": 2},
        {"iteration": 2, "policy_round": "r0001_t01_m0", "planned_grid": [2, 1]}
        | {"used_fallback": False, "effective_grid": [2, 1], "grid_calls": 4, "probes": 4}
        | {"decision_rounds": 2},
    ]
    assert (
        report["launches"],
        report["caps"]["fallback_calls"],
        report["caps"]["hard_max_calls"],
    ) == (
        1,
        4,
        9,
    )


def test_versions_that_plan_wider_than_the_tree_are_flagged_with_their_clipped_episodes(toy_run):
    """WIDE asks for 3 x 1 on 2 x 1 trees: every episode is clipped, 11 betas plus the default
    episode per trace, so 12 in iteration 1 and 24 over the two trees of iteration 2."""
    report, _ = toy_run
    rows = [
        (v["version"], v["episodes"], v["clipped"], v["asked"], v["recorded"], v["deployed"])
        for v in report["versions"]
    ]
    assert rows == [
        ("r0001_t01_m0", 12, 0, [], [], True),
        ("r0002_t01_m1", 12, 12, [[3, 1]], [[2, 1]], False),
        ("r0003_t01_m2", 12, 0, [], [], False),
        ("r0004_t02_m0", 24, 0, [], [], True),
        ("r0005_t02_m1", 24, 24, [[3, 1]], [[2, 1]], False),
        ("r0006_t02_m2", 24, 0, [], [], False),
    ]
    assert report["out_of_support"] == {
        "flagged": ["r0002_t01_m1", "r0005_t02_m1"],
        "flagged_deployed": [],  # clipped to the tree, WIDE ties m0, and ties keep the earlier
    }


def test_an_untouched_attempt_is_reported_with_its_source_and_both_scores(toy_run):
    report, _ = toy_run
    assert report["untouched"] == {
        "attempts": 8,
        "cases": [
            {"iteration": 1, "cell": "b0a0", "source": "baseline", "fail_class": "ok"}
            | {"agent_timed_out": False, "evaluated": True, "score": 1.0, "source_score": 1.0},
            {"iteration": 1, "cell": "b1a1", "source": "parent", "fail_class": "timeout"}
            | {"agent_timed_out": True, "evaluated": True, "score": 1.1, "source_score": 1.1},
        ],
    }


def test_versions_that_end_on_an_empty_batch_are_named_with_their_cause(toy_run):
    report, _ = toy_run
    empty = [
        (v["version"], v["valid"], v["cause"], v["episode_errors"], v["first_error"])
        for v in report["versions"]
        if v["cause"]
    ]
    line = "see.world.IllegalBatch: empty batch: stop by not probing"
    assert empty == [
        ("r0003_t01_m2", False, "empty_batch", 12, line),
        ("r0006_t02_m2", False, "empty_batch", 24, line),
    ]
    assert report["empty_batches"] == {"versions": ["r0003_t01_m2", "r0006_t02_m2"], "live": []}


def test_a_live_batch_left_empty_is_reported_from_its_manifest(tmp_path, stub_prompts):
    policy = tmp_path / "empty.py"
    policy.write_text(EMPTY)
    _launch(tmp_path)
    cfg = _config(tmp_path, iterations=1, versions=1, initial_policy=str(policy))
    DreamRSI(cfg, make_task(str(tmp_path)), ScriptedDiscoveryAgent(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(tmp_path))
    (row,) = report["iterations"]
    assert (row["probes"], row["error"], row["error_cause"]) == (
        0,
        "IllegalBatch: empty batch: stop by not probing",
        "empty_batch",
    )
    assert report["empty_batches"] == {"versions": ["r0001_t01_m0"], "live": [1]}


class _Interrupting:
    """Adds 0.1, and raises KeyboardInterrupt on attempt b1a1, as Ctrl-C would mid-batch."""

    def __call__(self, prompt, *, cwd, target):
        if os.path.basename(target) == "attempt_b001_a001":
            raise KeyboardInterrupt("Ctrl-C")
        path = os.path.join(target, PROGRAM)
        with open(path) as f:
            x = json.load(f)["x"]
        with open(path, "w") as f:
            json.dump({"x": round(x + 0.1, 6)}, f)
        return {"returncode": 0}


def test_an_interrupted_iteration_is_reported_as_partial(tmp_path, stub_prompts):
    """The second batch is abandoned whole, so the partial tree holds the first batch only."""
    _launch(tmp_path)
    cfg = _config(tmp_path, iterations=1, versions=1)
    with pytest.raises(KeyboardInterrupt):
        DreamRSI(cfg, make_task(str(tmp_path)), _Interrupting(), ScriptedPolicyAgent()).run()
    report = report_run.build_report(str(tmp_path))
    (row,) = report["iterations"]
    assert (row["partial"], row["probes"], row["attempts"], row["error"]) == (
        True,
        2,
        2,
        "KeyboardInterrupt: Ctrl-C",
    )
    assert report["versions"] == []  # offline never ran


def test_the_health_table_counts_each_iterations_outcomes(toy_run):
    report, _ = toy_run
    keys = ("attempts", "successes", "fail_classes", "agent_timeouts", "no_program")
    keys += ("evaluator_crashed", "baseline_score", "best_score", "error")
    assert [{k: i[k] for k in keys} for i in report["iterations"]] == [
        {"attempts": 4, "successes": 3, "fail_classes": {"ok": 3, "timeout": 1}}
        | {"agent_timeouts": 1, "no_program": 0, "evaluator_crashed": 0}
        | {"baseline_score": 1.0, "best_score": 1.1, "error": None},
        {"attempts": 4, "successes": 3, "fail_classes": {"no_program": 1, "ok": 3}}
        | {"agent_timeouts": 0, "no_program": 1, "evaluator_crashed": 0}
        | {"baseline_score": 1.0, "best_score": 1.2, "error": None},
    ]


def test_the_evidence_subset_carries_no_program_and_withholds_a_quoted_one(toy_run):
    """43 files: state.json and launches.jsonl; three per frozen iteration (6); eight score.json;
    two error.txt (the timeout, the deleted program); seven of eight proposals; method.py and two
    sweep files for each of six versions (18). SimpleTES programs are AGPL and stay out."""
    report, out = toy_run
    copied = [f for _, _, files in os.walk(out / "workdir") for f in files]
    assert PROGRAM not in copied
    assert len(copied) == 43
    assert report["evidence"] == {
        "copied": 43,
        "withheld": ["runs/iter0002/tree/attempt_b000_a001/proposal.md"],
    }


def test_the_archive_digest_is_recorded(toy_run):
    report, _ = toy_run
    assert report["archive"] == {
        "file": "archive.bin",
        "sha256": "0eb3e36bfb24dcd9bb1d1bece1531216b59539a8fde17ee80224af0653c92aa3",
    }


def test_the_reports_resume_source_rule_is_the_loops(tmp_path):
    """Programs exist at b0a0, b0a2 and b2a1 only; report and loop must agree on every cell."""
    task = make_task(str(tmp_path))
    tree = tmp_path / "tree"
    for node in ("attempt_b000_a000", "attempt_b000_a002", "attempt_b002_a001"):
        (tree / node).mkdir(parents=True)
        (tree / node / PROGRAM).write_text("{}")
    q = LiveQuestion(task, ScriptedDiscoveryAgent(), str(tree), str(tmp_path / "h"), 1.0, 2, 3, 3)
    for b in range(3):
        for a in range(4):
            path, _ = report_run.resume_source(str(tree), task.baseline_dir, PROGRAM, b, a)
            assert path == q._resume_from(b, a), (b, a)
    source = {
        (b, a): report_run.resume_source(str(tree), task.baseline_dir, PROGRAM, b, a)[1]
        for b, a in ((0, 0), (0, 2), (0, 3), (2, 1), (2, 3))
    }
    assert source == {(0, 0): None, (0, 2): 0, (0, 3): 2, (2, 1): None, (2, 3): 1}


def test_the_smoke_tests_noise_is_the_relative_spread_of_its_repeats(tmp_path):
    runs = [
        {"combined_score": 0.5, "geo_mean_sol_ms": 2.0},
        {"combined_score": 0.625, "geo_mean_sol_ms": 1.6},
    ]
    (tmp_path / "host.json").write_text(json.dumps({"labels": [], "results": {"seed": runs}}))
    assert report_run.noise(str(tmp_path / "host.json")) == {
        "seed": {"scores": [0.5, 0.625], "geo_mean_ms": [2.0, 1.6], "spread": 0.25}
    }


def test_the_markdown_report_answers_the_four_questions(toy_run):
    _, out = toy_run
    md = (out / "report.md").read_text()
    for answer in (
        "## 1. Per-round call budget",
        "fallback 2 x 1 = 4 calls, hard max 3 x 2 = 9 calls",
        "2 version(s) replayed on clipped episodes; 0 of them deployed.",
        "2 of 8 attempts left their resume source byte for byte.",
        "Versions scored minus infinity for an empty batch: r0003_t01_m2, r0006_t02_m2.",
    ):
        assert answer in md
````

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest -q tests/test_report_run.py`
Expected: a collection error, `FileNotFoundError: [Errno 2] No such file or directory: '…/reconstruction/scripts/report_run.py'`.

- [ ] **Step 3: Write the report.** Create `reconstruction/scripts/report_run.py`:

````python
"""Turn a Dream-RSI workdir into the D2a evidence report: report.md and report.json.

    python scripts/report_run.py --workdir ~/dream-rsi-runs/d2a-lasso --out evidence/d2a-lasso \\
        --host-json evidence/d2a-lasso/host.json --archive ~/dream-rsi-runs/d2a-lasso.tar.gz \\
        --copy-evidence

Read-only over the workdir. It answers, with counts, the four questions track D1 deferred to D2
(docs/superpowers/specs/2026-09-25-real-agent-run-d2a-design.md, section 5): what each round
spent against its caps, which policy versions were scored on replay episodes clipped to a
recorded tree, which attempts left their resume source byte for byte, and which versions or live
batches ended on an empty or illegal batch. The caps and the program's file name come from the
last line of the runner's launches.jsonl. Programs are compared as they are on disk after the
run; an agent runs with the whole tree as its cwd, so a later agent could have edited an earlier
attempt's program.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import shutil

from see.live import node_dirname
from see.world import Trace

PAPER_CALLS = {"pro": 110, "flash": 640}  # Sec. 4: 10 x 11 and 32 x 20 calls per round
# The license-safe subset of a workdir (spec section 6): no attempt program is ever listed.
EVIDENCE = (
    "state.json",
    "launches.jsonl",
    "trace_pool/iter*/trace.json",
    "trace_pool/iter*/live_cycle_manifest.json",
    "trace_pool/iter*/live_episode.jsonl",
    "runs/iter*/partial/trace.json",
    "runs/iter*/partial/live_cycle_manifest.json",
    "runs/iter*/partial/live_episode.jsonl",
    "runs/iter*/tree/attempt_*/eval/score.json",
    "runs/iter*/tree/attempt_*/error.txt",
    "runs/iter*/tree/attempt_*/proposal.md",
    "policy_dev/history/r*/method.py",
    "policy_dev/history/r*/proposal_results/beta_sweep.json",
    "policy_dev/history/r*/proposal_results/policy_execution_traces.jsonl",
)
PROGRAM_MARKER = (
    "CPP_CODE"  # opens every SimpleTES Lasso program (AGPL); a file quoting it stays out
)


def _json(path: str):
    with open(path) as f:
        return json.load(f)


def _jsonl(path: str) -> list:
    if not os.path.exists(path):  # a sweep that crashed or timed out wrote only beta_sweep.json
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def calls(grid) -> int:
    """Attempts a (branch_count, refine_count) grid admits; R counts refinements after the root."""
    return grid[0] * (grid[1] + 1)


def resume_source(tree_dir: str, baseline_dir: str, program: str, branch: int, attempt: int):
    """see.live.LiveQuestion._resume_from, read back after the run: (path, source attempt).

    The parent's saved program; past an attempt that left none, the nearest ancestor's; else the
    baseline's, with source attempt None.
    """
    for a in range(attempt - 1, -1, -1):
        path = os.path.join(tree_dir, node_dirname(branch, a), program)
        if os.path.exists(path):
            return path, a
    return os.path.join(baseline_dir, program), None


def cause(errors: list) -> str | None:
    """Why a version or a live batch failed: the empty batch D2 asks about, another illegal one,
    or anything else."""
    text = "\n".join(errors)
    if "empty batch" in text:
        return "empty_batch"
    if "IllegalBatch" in text:
        return "illegal_batch"
    return "other" if errors else None


def _last_line(error: str) -> str:
    lines = error.strip().splitlines()
    return lines[-1] if lines else ""


def analyse_iteration(workdir: str, run_dir: str, frozen: str, program: str) -> tuple:
    """(iteration row, untouched attempts, trace) for one iteration's frozen or partial trace."""
    trace = Trace.load(os.path.join(frozen, "trace.json"))
    manifest = _json(os.path.join(frozen, "live_cycle_manifest.json"))
    tree = os.path.join(run_dir, "tree")
    baseline_dir = os.path.join(workdir, "task", "baseline")
    t = manifest["iteration"]
    timeouts = crashed = 0
    untouched = []
    for c in sorted(trace.cells.values(), key=lambda c: c.seq):
        node = os.path.join(tree, node_dirname(c.branch, c.attempt))
        score_path = os.path.join(node, "eval", "score.json")
        score = _json(score_path) if os.path.exists(score_path) else {}
        timeouts += bool(score.get("agent_timed_out"))
        crashed += bool(score.get("evaluator_crashed"))
        mine = os.path.join(node, program)
        source, source_attempt = resume_source(tree, baseline_dir, program, c.branch, c.attempt)
        if not (os.path.exists(mine) and _bytes(mine) == _bytes(source)):
            continue
        if source_attempt is None:
            kind, source_score = "baseline", trace.baseline_score
        else:
            parent = trace.cell(c.branch, source_attempt)
            kind = "parent" if source_attempt == c.attempt - 1 else "ancestor"
            source_score = parent.score if parent else None
        untouched.append(
            {
                "iteration": t,
                "cell": c.id,
                "source": kind,
                "fail_class": c.fail_class,
                "agent_timed_out": bool(score.get("agent_timed_out")),
                "evaluated": c.evaluated,
                "score": c.score,
                "source_score": source_score,
            }
        )
    planned = manifest.get("planned_grid")
    grid = manifest["effective_grid"]
    fail_classes = collections.Counter(c.fail_class for c in trace.cells.values())
    row = {
        "iteration": t,
        "partial": bool(manifest.get("partial")),
        "policy_round": manifest.get("policy_round"),
        "planned_grid": [planned["branch_count"], planned["refine_count"]] if planned else None,
        "used_fallback": manifest.get("used_fallback"),
        "effective_grid": [grid["branch_count"], grid["refine_count"]],
        "grid_calls": calls((grid["branch_count"], grid["refine_count"])),
        "probes": manifest["probes"],
        "decision_rounds": manifest["decision_rounds"],
        "attempts": len(trace),
        "successes": sum(c.success for c in trace.cells.values()),
        "fail_classes": dict(sorted(fail_classes.items())),
        "agent_timeouts": timeouts,
        "no_program": fail_classes.get("no_program", 0),
        "evaluator_crashed": crashed,
        "baseline_score": trace.baseline_score,
        "best_score": manifest.get("best_score"),
        "error": manifest.get("error"),
        "error_cause": cause([manifest["error"]]) if manifest.get("error") else None,
    }
    return row, untouched, trace


def analyse_version(rdir: str, grids: dict, fallback, selected: set) -> dict:
    """One policy version: its sweep's validity and scores, and its clipped replay episodes."""
    name = os.path.basename(rdir)
    _, t, m = name.split("_")
    results = os.path.join(rdir, "proposal_results")
    report_path = os.path.join(results, "beta_sweep.json")
    report = _json(report_path) if os.path.exists(report_path) else {}
    episodes = _jsonl(os.path.join(results, "policy_execution_traces.jsonl"))
    clipped = [e for e in episodes if e["out_of_support"]]
    asked = {
        (e["plan"]["branch_count"], e["plan"]["refine_count"]) if e["plan"] else tuple(fallback)
        for e in clipped
    }
    recorded = {tuple(grids[e["trace_id"]]) for e in clipped if e["trace_id"] in grids}
    # every episode's error, not beta_sweep.json's first three; the report's own when none ran
    errors = [e["error"] for e in episodes if e["error"]] or list(report.get("errors", []))
    return {
        "version": name,
        "iteration": int(t[1:]),
        "m": int(m[1:]),
        "valid": bool(report.get("valid", False)),
        "reward": report.get("pareto", {}).get("reward"),
        "eq1": report.get("eq1", {}).get("V"),
        "episodes": len(episodes),
        "clipped": len(clipped),
        "asked": [list(a) for a in sorted(asked)],
        "recorded": [list(g) for g in sorted(recorded)],
        "episode_errors": sum(1 for e in episodes if e["error"]),
        "cause": cause(errors),
        "first_error": _last_line(errors[0]) if errors else None,
        "deployed": name in selected,
    }


def noise(host_json: str) -> dict:
    """The smoke test's repeats per program; spread is (max - min) / min of combined_score."""
    out = {}
    for name, runs in _json(host_json)["results"].items():
        scores = [r["combined_score"] for r in runs]
        out[name] = {
            "scores": scores,
            "geo_mean_ms": [r["geo_mean_sol_ms"] for r in runs],
            "spread": (max(scores) - min(scores)) / min(scores) if min(scores) > 0 else None,
        }
    return out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_evidence(workdir: str, dest: str) -> dict:
    """Copy the EVIDENCE subset under ``dest``, withholding any file that quotes a program."""
    copied, withheld = 0, []
    for pattern in EVIDENCE:
        for path in sorted(glob.glob(os.path.join(glob.escape(workdir), pattern))):
            rel = os.path.relpath(path, workdir)
            with open(path, errors="replace") as f:
                if PROGRAM_MARKER in f.read():
                    withheld.append(rel)
                    continue
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
    return {"copied": copied, "withheld": withheld}


def _home_relative(path: str) -> str:
    """The report is published: show a path under the home directory as ~/..."""
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if path == home or path.startswith(home + os.sep) else path


def build_report(workdir: str, host_json: str | None = None, archive: str | None = None) -> dict:
    workdir = os.path.abspath(workdir)
    launches = _jsonl(os.path.join(workdir, "launches.jsonl"))
    if not launches:
        raise FileNotFoundError(f"{workdir}/launches.jsonl: not a workdir the runner launched")
    launch = launches[-1]
    program = launch["task"]["eval_program"]
    fallback, hard_max = launch["config"]["fallback_grid"], launch["config"]["hard_max_grid"]
    state_path = os.path.join(workdir, "state.json")
    state = _json(state_path) if os.path.exists(state_path) else {"log": []}
    selected = {entry["selected"] for entry in state["log"] if "selected" in entry}
    rows, untouched, grids, unfrozen = [], [], {}, []
    for run_dir in sorted(glob.glob(os.path.join(glob.escape(workdir), "runs", "iter*"))):
        name = os.path.basename(run_dir)
        frozen = os.path.join(workdir, "trace_pool", name)
        if not os.path.exists(os.path.join(frozen, "trace.json")):
            frozen = os.path.join(run_dir, "partial")
        if not os.path.exists(os.path.join(frozen, "trace.json")):
            unfrozen.append(name)
            continue
        row, cases, trace = analyse_iteration(workdir, run_dir, frozen, program)
        rows.append(row)
        untouched += cases
        grids[trace.trace_id] = list(trace.grid)
    history = os.path.join(glob.escape(workdir), "policy_dev", "history", "r*")
    versions = [analyse_version(r, grids, fallback, selected) for r in sorted(glob.glob(history))]
    flagged = [v for v in versions if v["clipped"]]
    return {
        "workdir": _home_relative(workdir),
        "launches": len(launches),
        "task": launch["task"],
        "agents": launch.get("agents"),
        "host": launch.get("host"),
        "caps": {
            "fallback_grid": fallback,
            "fallback_calls": calls(fallback),
            "hard_max_grid": hard_max,
            "hard_max_calls": calls(hard_max),
            "paper_calls": PAPER_CALLS,
        },
        "archive": {"file": os.path.basename(archive), "sha256": sha256_of(archive)}
        if archive
        else None,
        "iterations": rows,
        "unfrozen_iterations": unfrozen,
        "versions": versions,
        "out_of_support": {
            "flagged": [v["version"] for v in flagged],
            "flagged_deployed": [v["version"] for v in flagged if v["deployed"]],
        },
        "untouched": {"attempts": sum(r["attempts"] for r in rows), "cases": untouched},
        "empty_batches": {
            "versions": [v["version"] for v in versions if v["cause"] == "empty_batch"],
            "live": [r["iteration"] for r in rows if r["error_cause"] == "empty_batch"],
        },
        "noise": noise(host_json) if host_json else None,
    }


def _grid(g) -> str:
    return f"{g[0]} x {g[1]}" if g else "none"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, float) else str(x)


def markdown(r: dict) -> str:
    caps = r["caps"]
    out = [
        "# D2a run report",
        "",
        f"Workdir `{r['workdir']}`, {r['launches']} launch(es), task `{r['task'].get('name')}`.",
    ]
    if r["archive"]:
        out.append(f"Archive `{r['archive']['file']}`, sha256 `{r['archive']['sha256']}`.")
    out += [
        "",
        "## 1. Per-round call budget",
        "",
        f"Caps in force: fallback {_grid(caps['fallback_grid'])} = {caps['fallback_calls']} calls, "
        f"hard max {_grid(caps['hard_max_grid'])} = {caps['hard_max_calls']} calls. The paper: "
        f"{caps['paper_calls']['pro']} calls per round (Pro), {caps['paper_calls']['flash']} "
        "(Flash).",
        "",
        "| Iteration | Policy | Planned | Used fallback | Effective | Grid calls | Probes "
        "| Rounds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in r["iterations"]:
        out.append(
            f"| {i['iteration']}{' (partial)' if i['partial'] else ''} | {i['policy_round']} | "
            f"{_grid(i['planned_grid'])} | {i['used_fallback']} | {_grid(i['effective_grid'])} | "
            f"{i['grid_calls']} | {i['probes']} | {i['decision_rounds']} |"
        )
    oos = r["out_of_support"]
    out += [
        "",
        "## 2. Out-of-support replay",
        "",
        f"{len(oos['flagged'])} version(s) replayed on clipped episodes; "
        f"{len(oos['flagged_deployed'])} of them deployed.",
        "",
        "| Version | Episodes | Clipped | Asked | Recorded | Deployed |",
        "|---|---|---|---|---|---|",
    ]
    for v in r["versions"]:
        asked = ", ".join(_grid(a) for a in v["asked"]) or "-"
        recorded = ", ".join(_grid(g) for g in v["recorded"]) or "-"
        out.append(
            f"| {v['version']} | {v['episodes']} | {v['clipped']} | {asked} | {recorded} | "
            f"{v['deployed']} |"
        )
    u = r["untouched"]
    out += [
        "",
        "## 3. Untouched programs",
        "",
        f"{len(u['cases'])} of {u['attempts']} attempts left their resume source byte for byte.",
        "",
        "| Iteration | Cell | Source | Fail class | Agent timed out | Evaluated | Score | "
        "Source score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in u["cases"]:
        out.append(
            f"| {c['iteration']} | {c['cell']} | {c['source']} | {c['fail_class']} | "
            f"{c['agent_timed_out']} | {c['evaluated']} | {_num(c['score'])} | "
            f"{_num(c['source_score'])} |"
        )
    e = r["empty_batches"]
    out += [
        "",
        "## 4. Empty batches",
        "",
        f"Versions scored minus infinity for an empty batch: {', '.join(e['versions']) or 'none'}. "
        f"Live iterations whose policy left a batch empty: "
        f"{', '.join(map(str, e['live'])) or 'none'}.",
        "",
        "| Version | Valid | Cause | Episode errors | Last error line |",
        "|---|---|---|---|---|",
    ]
    for v in r["versions"]:
        if v["cause"]:
            out.append(
                f"| {v['version']} | {v['valid']} | {v['cause']} | {v['episode_errors']} | "
                f"`{v['first_error']}` |"
            )
    out += [
        "",
        "## Health",
        "",
        "| Iteration | Attempts | Successes | Fail classes | Agent timeouts | No program | "
        "Evaluator crashed | Baseline | Best | Error |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i in r["iterations"]:
        classes = ", ".join(f"{k} {n}" for k, n in i["fail_classes"].items())
        out.append(
            f"| {i['iteration']} | {i['attempts']} | {i['successes']} | {classes} | "
            f"{i['agent_timeouts']} | {i['no_program']} | {i['evaluator_crashed']} | "
            f"{_num(i['baseline_score'])} | {_num(i['best_score'])} | {i['error'] or ''} |"
        )
    out += ["", "| Version | Valid | Reward | Eq. (1) V | Deployed |", "|---|---|---|---|---|"]
    for v in r["versions"]:
        out.append(
            f"| {v['version']} | {v['valid']} | {_num(v['reward'])} | {_num(v['eq1'])} | "
            f"{v['deployed']} |"
        )
    if r["unfrozen_iterations"]:
        out += ["", f"Iterations with no trace: {', '.join(r['unfrozen_iterations'])}."]
    out += ["", "## Host", ""]
    for k, v in sorted((r["host"] or {}).items()):
        out.append(f"- {k}: {v}")
    for role, a in sorted((r["agents"] or {}).items()):
        out.append(f"- {role} agent: `{json.dumps(a.get('argv'))}`, version {a.get('version')}")
    if r["noise"]:
        out += [
            "",
            "## Evaluation noise (smoke test)",
            "",
            "| Program | Scores | Spread |",
            "|---|---|---|",
        ]
        for name, n in r["noise"].items():
            scores = ", ".join(_num(s) for s in n["scores"])
            out.append(f"| {name} | {scores} | {_num(n['spread'])} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host-json", help="the smoke test's verify_lasso.py --json output")
    ap.add_argument("--archive", help="the workdir archive whose sha256 the report records")
    ap.add_argument(
        "--copy-evidence",
        action="store_true",
        help="also copy the license-safe subset of the workdir to <out>/workdir/",
    )
    a = ap.parse_args(argv)
    report = build_report(a.workdir, a.host_json, a.archive)
    os.makedirs(a.out, exist_ok=True)
    if a.copy_evidence:
        report["evidence"] = copy_evidence(
            os.path.abspath(a.workdir), os.path.join(a.out, "workdir")
        )
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write(markdown(report))
    print(os.path.join(a.out, "report.md"))
    return report


if __name__ == "__main__":
    main()
````

- [ ] **Step 4: Run the tests to see them pass, with and without `generated/`**

Run: `python -m pytest -q tests/test_report_run.py`
Expected: `12 passed` (about 1 s).

Then check that nothing leans on the real prompts:

```bash
mv generated ../generated.off
python -m pytest -q tests/test_report_run.py
mv ../generated.off generated
```

Expected: `12 passed` again.

- [ ] **Step 5: Keep evidence out of the formatters, then update docs and the CI count**

In `reconstruction/pyproject.toml`, replace:

````toml
[tool.ruff]
line-length = 100
target-version = "py310"
````

with:

````toml
[tool.ruff]
line-length = 100
target-version = "py310"
extend-exclude = ["evidence"]  # run evidence: LLM-written policies and tool output, as written
````

In `.pre-commit-config.yaml`, replace:

````yaml
      - id: trailing-whitespace
        exclude: ^(assets/|papers/)
      - id: end-of-file-fixer
        exclude: ^(assets/|papers/)
````

with:

````yaml
      - id: trailing-whitespace
        exclude: ^(assets/|papers/|reconstruction/evidence/)
      - id: end-of-file-fixer
        exclude: ^(assets/|papers/|reconstruction/evidence/)
````

In `.pre-commit-config.yaml`, replace:

````yaml
      - id: ruff-check
        args: [--fix]
        files: ^reconstruction/
      - id: ruff-format
        files: ^reconstruction/
````

with:

````yaml
      - id: ruff-check
        args: [--fix]
        files: ^reconstruction/
        exclude: ^reconstruction/evidence/
      - id: ruff-format
        files: ^reconstruction/
        exclude: ^reconstruction/evidence/
````

In `reconstruction/README.md`, replace:

````markdown
```bash
python -m see sweep --method my_policy.py --pool runs/lasso/trace_pool --out /tmp/sweep
```
````

with:

````markdown
```bash
python -m see sweep --method my_policy.py --pool runs/lasso/trace_pool --out /tmp/sweep
```

Turn a run's workdir into counts for the four questions track D2 asks (per-round calls,
out-of-support replay, untouched programs, empty batches), as `report.md` and `report.json`:

```bash
python scripts/report_run.py --workdir runs/lasso --out /tmp/report
```
````

In `reconstruction/README.md`, replace:

````markdown
scripts/                    verify_lasso.py, run_dream_rsi.py
````

with:

````markdown
scripts/                    verify_lasso.py, run_dream_rsi.py, report_run.py
evidence/                   committed run evidence: reports, traces, scores (no programs)
````

In `.claude/CLAUDE.md`, replace:

````markdown
python -m see sweep --method my_policy.py --pool runs/lasso/trace_pool --out /tmp/sweep
````

with:

````markdown
python -m see sweep --method my_policy.py --pool runs/lasso/trace_pool --out /tmp/sweep
python scripts/report_run.py --workdir runs/lasso --out /tmp/report   # D2's four questions
````

In `.claude/CLAUDE.md`, replace:

````markdown
- `*.local.md` files are private maintainer notes and are gitignored.
````

with:

````markdown
- `*.local.md` files are private maintainer notes and are gitignored.
- `evidence/` holds committed run evidence written by tools, never an attempt's program
  (SimpleTES is AGPL): `report_run.py --copy-evidence` copies an allowlist and withholds any
  file quoting `CPP_CODE`. It is excluded from ruff and the whitespace hooks, kept as written.
````

In `.github/workflows/ci.yml`, replace:

````yaml
--expect 106
````

with:

````yaml
--expect 118
````

- [ ] **Step 6: Run the gate, and prove the exclusion**

```bash
mkdir -p evidence/probe && printf 'import os,sys\nx=1;y  =  2\n' > evidence/probe/method.py
ruff check . && ruff format --check .      # both pass: evidence/ is excluded
rm -rf evidence/probe
```

Then run the full gate. Expected: `118 passed`, `junit: 118 tests, no skips`, ruff and pyright clean.

- [ ] **Step 7: Commit**

Run `git diff --stat`. It should show exactly the 7 files above.

```bash
cat > /tmp/d2a-t3-msg.txt <<'EOF'
Add the D2a evidence report over a run workdir

scripts/report_run.py reads a finished or partial workdir and writes
report.md and report.json: each round's plan and spend against its caps,
the policy versions replayed on clipped episodes, the attempts that left
their resume source byte for byte, and the versions or live batches that
ended on an empty batch, with a health table. --copy-evidence copies an
allowlist that never names a program. evidence/ is excluded from ruff and
the whitespace hooks so it stays as written.
EOF
git add reconstruction/scripts/report_run.py reconstruction/tests/test_report_run.py \
  reconstruction/pyproject.toml .pre-commit-config.yaml reconstruction/README.md \
  .claude/CLAUDE.md .github/workflows/ci.yml
git commit -F /tmp/d2a-t3-msg.txt
```

---

### Task 4: SimpleTES clone and the host smoke test (controller only; owner's go)

**Files:**
- Create: `reconstruction/evidence/d2a-lasso/host.json`
- Outside the repo: `../SimpleTES` (gitignored as `SimpleTES/` at the repo root) and `/Users/Shared/dream-rsi-runs/smoke.log`

**Interfaces:**
- Consumes: Task 1's `verify_lasso.py --eigen-include`.
- Produces: `evidence/d2a-lasso/host.json`, the `verify_lasso.py --json` output `{"labels": [17 names], "results": {"dream_rsi" | "glmnet_port" | "simpletes_best": [run0, run1]}}`. Each run has `n_valid`, `n_total`, `geo_mean_sol_ms` and `combined_score`. Task 6 reads it through `report_run.py --host-json`.

- [ ] **Step 1: Ask the owner for the go and stop until it comes.** Say that Task 4 clones SimpleTES (AGPL) into the gitignored `../SimpleTES` and runs the Lasso smoke test on this Mac. It is CPU only, roughly 10–20 minutes, and spends no API budget.

- [ ] **Step 2: Write the run's environment file, then check the compiler and the interpreter**

Each Bash tool call starts a fresh shell, so every later command in Tasks 4–6 begins with `. /Users/Shared/dream-rsi-runs/env.sh`. MacPorts must come first on `PATH` and the venv is activated after it, so `python` stays the venv's while `g++` resolves to MacPorts gcc.

```bash
mkdir -p /Users/Shared/dream-rsi-runs && chmod 700 /Users/Shared/dream-rsi-runs
cat > /Users/Shared/dream-rsi-runs/env.sh <<'EOF'
export PATH=/opt/local/bin:$PATH
cd /Users/controlroom/Dream-RSI/reconstruction && . .venv/bin/activate
R=/Users/Shared/dream-rsi-runs
EOF
. /Users/Shared/dream-rsi-runs/env.sh
command -v python           # expected: /Users/controlroom/Dream-RSI/reconstruction/.venv/bin/python
command -v g++              # expected: /opt/local/bin/g++
g++ --version | head -1     # expected: g++ (MacPorts gcc13 13.4.0_1+stdlib_flag) 13.4.0
```

`/usr/bin/g++` is Apple clang without OpenMP, and the seed program compiles without `-fopenmp`. A wrong `PATH` would therefore surface only later, as agent "compile failures". Stop if any of the three lines differs.

- [ ] **Step 3: Clone SimpleTES and record its commit**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
git clone --depth 1 https://github.com/wq-will/SimpleTES ../SimpleTES
git -C ../SimpleTES rev-parse HEAD      # write it into the SDD ledger
git -C /Users/controlroom/Dream-RSI status --short   # expected: nothing (SimpleTES/ is ignored)
```

- [ ] **Step 4: Clear the evaluator's binary cache.** The evaluator caches binaries by the md5 of `CPP_CODE` alone. The compiler, flags and Eigen path are not part of the key, so a stale binary built elsewhere would be reused.

```bash
. /Users/Shared/dream-rsi-runs/env.sh
rm -rf "$(python -c 'import tempfile; print(tempfile.gettempdir())')/lasso_path_cpp_cache"
```

- [ ] **Step 5: Run the smoke test.** It takes longer than the Bash tool's 600 s limit, so run it as a Bash call with `run_in_background: true`. The session is re-invoked when it exits.

```bash
. /Users/Shared/dream-rsi-runs/env.sh
python tools/extract_listings.py --check
mkdir -p evidence/d2a-lasso
python scripts/verify_lasso.py --simpletes ../SimpleTES --repeats 2 \
  --eigen-include /opt/local/include/eigen3 --json evidence/d2a-lasso/host.json \
  2>&1 | tee /Users/Shared/dream-rsi-runs/smoke.log
```

Expected:
- A table with six rows: `dream_rsi`, `glmnet_port` and `simpletes_best`, each with runs 0 and 1.
- Every row's valid column reads `17/17`, and no row says `ERROR`.
- The speed-up lines follow the table.

Stop condition: any row below `17/17`, or any `ERROR`. A compile failure names the toolchain. Report the first error lines to the owner and stop. No budget has been spent.

- [ ] **Step 6: Record the noise in the ledger**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
python -c "
from see.loader import load_module_from_path
r = load_module_from_path('rr', 'scripts/report_run.py')
for name, n in r.noise('evidence/d2a-lasso/host.json').items():
    print(name, n['geo_mean_ms'], n['scores'], round(n['spread'], 4))
"
```

- [ ] **Step 7: Check the file for home paths, then commit**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
grep -c '/Users/' evidence/d2a-lasso/host.json   # expected: 0; if not, show the owner the hits first
git diff --stat; git status --short              # only evidence/d2a-lasso/host.json is new
cat > /tmp/d2a-t4-msg.txt <<'EOF'
Record the D2a host smoke test

SimpleTES's Lasso evaluator, run twice on each of Listing 3, the glmnet
port and SimpleTES's best program on this Mac (MacPorts gcc 13, Eigen
3.4.1 through --eigen-include), proves the toolchain before the run and
measures this host's evaluation noise.
EOF
git add reconstruction/evidence/d2a-lasso/host.json
git commit -F /tmp/d2a-t4-msg.txt
```

---

### Task 5: The run (controller only; owner's go; spends API budget)

**Files:** none in the repository. Outside it, under `/Users/Shared/dream-rsi-runs/` (called `R` below): `gemini-home/`, `discovery.json`, `policy.json`, `preflight/`, `d2a-lasso/` (the workdir), `d2a-lasso.log`, `d2a-lasso.pid` and `d2a-lasso.tar.gz`. It may also hold `d2a-lasso-restart1/`.

**Interfaces:**
- Consumes: Tasks 1–4 (`--eigen-include`, `launches.jsonl`, `report_run.py`, the clone). Task 4's `PATH` and venv must be in place.
- Produces: a finished or partial workdir `R/d2a-lasso` and its archive `R/d2a-lasso.tar.gz`, which Task 6 reports on.

- [ ] **Step 1: Ask the owner for the go and stop until it comes.** Show the owner:
  - the launch command (Step 5);
  - both argv files (Step 3) and the pre-flight (Step 4);
  - the expected spend: about 32–46 Gemini discovery calls and 4 Claude policy calls, over one to two hours.

  The go covers the pre-flight's two small calls and the run.

- [ ] **Step 2: Build the isolated Gemini home.** It holds only the login files, the owner's auth type and billing choice, and trust for `R`.

```bash
. /Users/Shared/dream-rsi-runs/env.sh
mkdir -p $R/gemini-home/.gemini
cp ~/.gemini/oauth_creds.json ~/.gemini/google_accounts.json $R/gemini-home/.gemini/
chmod 600 $R/gemini-home/.gemini/oauth_creds.json
python3 - <<'EOF'
import json
R = "/Users/Shared/dream-rsi-runs"
with open("/Users/controlroom/.gemini/settings.json") as f:
    owner = json.load(f)
settings = {"security": {"auth": {"selectedType": owner["security"]["auth"]["selectedType"]}}}
if "billing" in owner:
    settings["billing"] = owner["billing"]  # keep the owner's overage choice
with open(f"{R}/gemini-home/.gemini/settings.json", "w") as f:
    json.dump(settings, f, indent=1)
with open(f"{R}/gemini-home/.gemini/trustedFolders.json", "w") as f:
    json.dump({R: "TRUST_FOLDER"}, f, indent=1)
EOF
```

- [ ] **Step 3: Write the two agent argv files.**
  - Discovery: `--include-directories` opens the whole workdir. Listing 1 sends the agent to `history/` and `task/baseline/`, both outside its cwd.
  - Policy: `--add-dir` opens `trace_pool/`, which Listing 2 sends it to read from `policy_dev/`.

```bash
. /Users/Shared/dream-rsi-runs/env.sh
cat > $R/discovery.json <<EOF
["env", "HOME=$R/gemini-home", "gemini", "--model", "auto-gemini-2.5", "--yolo", "--include-directories", "$R/d2a-lasso", "--prompt", "{prompt}"]
EOF
cat > $R/policy.json <<EOF
["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits", "--setting-sources", "project,local", "--strict-mcp-config", "--add-dir", "$R/d2a-lasso/trace_pool"]
EOF
python3 -c "import json; [json.load(open('$R/' + f)) for f in ('discovery.json', 'policy.json')]; print('argv ok')"
```

- [ ] **Step 4: Pre-flight: one call per agent, with the run's env and argv, in a scratch layout**

The session's shell is zsh, which does not word-split variables, so the pre-flight and the launch are bash scripts. Both unset this session's `CLAUDE*` variables (among them `CLAUDECODE`) with an `env -u` array, so the nested `claude -p` starts as a top-level session.

```bash
. /Users/Shared/dream-rsi-runs/env.sh
cat > $R/strip.sh <<'EOF'
# bash: STRIP=(-u NAME ...) for every CLAUDE* variable this shell inherited
STRIP=()
for v in $(env | sed -n 's/^\(CLAUDE[A-Za-z0-9_]*\)=.*/\1/p'); do STRIP+=(-u "$v"); done
EOF
cat > $R/preflight.sh <<'EOF'
#!/bin/bash
. /Users/Shared/dream-rsi-runs/env.sh && . $R/strip.sh
P=$R/preflight; rm -rf $P
mkdir -p $P/tree/attempt_b000_a000 $P/task/baseline $P/policy_dev $P/trace_pool
echo seed-line-7 > $P/task/baseline/seed.txt; echo pool-line-3 > $P/trace_pool/note.txt
(cd $P/tree && env "${STRIP[@]}" HOME=$R/gemini-home gemini --model auto-gemini-2.5 --yolo \
  --include-directories $P --prompt "Read $P/task/baseline/seed.txt and write its first line, \
and nothing else, to $P/tree/attempt_b000_a000/out.txt. Then reply DONE.")
echo "discovery out: $(cat $P/tree/attempt_b000_a000/out.txt 2>&1)"
echo "extensions:"; (cd $P/tree && env HOME=$R/gemini-home gemini -l)
echo "ancestor context files:"
d=$R/d2a-lasso
while [ "$d" != "/" ]; do d=$(dirname "$d")
  for f in CLAUDE.md CLAUDE.local.md GEMINI.md AGENTS.md .claude/CLAUDE.md .gemini/GEMINI.md; do
    [ -e "$d/$f" ] && echo "  $d/$f"; done; done
(cd $P/policy_dev && env "${STRIP[@]}" claude -p "Read $P/trace_pool/note.txt and write its first \
line, and nothing else, to ./out.txt. Then answer with one word: does your context contain the \
phrase 'Rule 14'? Answer RULE14 or NONE." --permission-mode acceptEdits \
  --setting-sources project,local --strict-mcp-config --add-dir $P/trace_pool)
echo "policy out: $(cat $P/policy_dev/out.txt 2>&1)"
echo "remember dirs: $(ls -d ~/.remember/*preflight* 2>/dev/null)"
EOF
bash $R/preflight.sh
```

Expected: `discovery out: seed-line-7`; no extension listed under `extensions:`; nothing listed under `ancestor context files:`; Claude's reply `NONE`; `policy out: pool-line-3`; and `remember dirs:` followed by nothing (no user hook ran in that cwd).

The pre-flight passes when all of these hold:
- both `out.txt` files hold their expected line;
- `gemini -l` lists no extension;
- Claude answers `NONE`;
- the `ls` prints nothing.

If Claude answers `RULE14`, or a `~/.remember/*preflight*` directory appears, the user configuration still loads. Then:
1. Ask the owner to log in once, with `! CLAUDE_CONFIG_DIR=/Users/Shared/dream-rsi-runs/claude-home claude`, then `/login` and exit.
2. Rewrite `$R/policy.json` as `["env", "CLAUDE_CONFIG_DIR=$R/claude-home", "claude", "-p", "{prompt}", "--permission-mode", "acceptEdits", "--add-dir", "$R/d2a-lasso/trace_pool"]`.
3. Repeat the Claude half of the pre-flight with that argv.

Any other failure (auth, a missing file, a refused write) means stop and report; do not launch. Record both CLI versions and the pre-flight result in the ledger.

- [ ] **Step 5: Launch detached, so the run survives this session**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
cat > $R/launch.sh <<'EOF'
#!/bin/bash
. /Users/Shared/dream-rsi-runs/env.sh && . $R/strip.sh
rm -rf "$(python -c 'import tempfile; print(tempfile.gettempdir())')/lasso_path_cpp_cache"
nohup python -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' \
  env "${STRIP[@]}" python scripts/run_dream_rsi.py \
  --simpletes /Users/controlroom/Dream-RSI/SimpleTES --task lasso_path --workdir $R/d2a-lasso \
  --discovery-agent "$(cat $R/discovery.json)" --policy-agent "$(cat $R/policy.json)" \
  --iterations 2 --versions 3 --workers 4 --grid 4 3 --hard-max 6 4 --agent-timeout 900 \
  --eigen-include /opt/local/include/eigen3 > $R/d2a-lasso.log 2>&1 &
echo $! > $R/d2a-lasso.pid
EOF
bash $R/launch.sh; cat $R/d2a-lasso.pid
```

`nohup` leaves SIGHUP ignored, and the runner keeps it ignored (`install_signal_handlers`). The `setsid` wrapper `exec`s into the runner in a session of its own. The pid file therefore names the runner itself, and nothing that ends this session's process group reaches the run. A spike confirmed both: the process leads its own group, and it survives the Bash call that started it. SIGTERM to the pid freezes a partial iteration.

- [ ] **Step 6: Check the start, then record the task tree's hashes**

`launches.jsonl` appears within seconds. `baseline_eval.json` follows after one full Lasso evaluation, about as long as one smoke-test run. `runs/iter0001/tree/` appears only after that.

```bash
. /Users/Shared/dream-rsi-runs/env.sh
python3 -c "
import json
l = json.loads(open('$R/d2a-lasso/launches.jsonl').readlines()[-1])
print(l['config']['fallback_grid'], l['config']['hard_max_grid'], l['task']['eval_program'])
print(l['agents']['discovery']['version'], '|', l['agents']['policy']['version'])
print(l['host']['compiler'], '|', l['host']['eigen'], '|', l['host']['simpletes_commit'])
"
cd $R/d2a-lasso && find task -type f -print0 | sort -z | xargs -0 shasum -a 256 > $R/task.sha256
wc -l < $R/task.sha256    # the adapter's copy of the task (task/src/eigen is a symlink, not hashed)
```

Once `$R/d2a-lasso/baseline_eval.json` exists, `ls $R/d2a-lasso/runs/iter0001/tree | head` should list `attempt_b000_a000` and its neighbours. The discovery agent's file tools can write anywhere in the workdir. `task.sha256` therefore lets Task 6 prove that the baseline and the task files every "source: baseline" comparison depends on were not edited during the run.

Expected:
- `[4, 3] [6, 4] init_program.py`;
- two non-empty CLI versions;
- `g++ (MacPorts gcc13 13.4.0_1+stdlib_flag) 13.4.0 | 3.4.1 | <the Task 4 commit>`;
- a hash count of at least 4: `evaluator.py`, `init_program.py` and the statement file under `task/src/`, plus `task/baseline/init_program.py`.

A `None` compiler or Eigen means the environment is wrong. Stop the run with `kill -TERM $(cat $R/d2a-lasso.pid)` and fix the environment. The iteration is then partial; handle it as in Step 8.

- [ ] **Step 7: Wait for the exit.** Use a background wait, which re-invokes the session when the run ends: Bash with `run_in_background: true` running

```bash
while kill -0 $(cat /Users/Shared/dream-rsi-runs/d2a-lasso.pid) 2>/dev/null; do sleep 60; done
tail -40 /Users/Shared/dream-rsi-runs/d2a-lasso.log
```

While waiting, look at progress only when the owner asks. Progress means counting `eval/score.json` files, and reading any `error.txt`. The exception is quota: if consecutive attempts' `error.txt` mention a quota (`429`, `RESOURCE_EXHAUSTED`, `quota`), stop the run with `kill -TERM` and tell the owner. The run is not retried into a broken quota.

- [ ] **Step 8: On exit, decide: done, restart once, or stop**
  - **Done:** the log ends with the last iteration's log entry, and `state.json` has `"iteration": 2`.
  - **Restart once:** use this path only for the first failure, and only if it is not a quota failure.
    1. Check that no sweep survived: `pgrep -fl "see sweep"` should print nothing; kill any that did.
    2. Move aside exactly the directories `online()`'s guard names (`see/loop.py`, lines 157–163), keeping their relative paths. Copy `task/`, `launches.jsonl` and `state.json` beside them, so Task 6 can report on the aside directory as a workdir of its own:

       ```bash
       . /Users/Shared/dream-rsi-runs/env.sh
       python3 - <<'EOF'
       import glob, json, os, shutil
       R = "/Users/Shared/dream-rsi-runs"
       w, aside = f"{R}/d2a-lasso", f"{R}/d2a-lasso-restart1"
       with open(f"{w}/state.json") as f:
           t = json.load(f)["iteration"] + 1
       named = [f"{w}/runs/iter{t:04d}", f"{w}/trace_pool/iter{t:04d}",
                *sorted(glob.glob(f"{w}/policy_dev/history/r*_t{t:02d}_m*"))]
       for p in [p for p in named if os.path.lexists(p)]:
           dest = os.path.join(aside, os.path.relpath(p, w))
           os.makedirs(os.path.dirname(dest), exist_ok=True)
           shutil.move(p, dest)
           print("moved", os.path.relpath(p, w))
       shutil.copytree(f"{w}/task", f"{aside}/task", symlinks=True)
       for name in ("launches.jsonl", "state.json"):
           shutil.copy(f"{w}/{name}", f"{aside}/{name}")
       EOF
       ```

    3. Relaunch with `bash $R/launch.sh`, the same script as Step 5. The runner appends a second launch line.
  - **Stop:** on a second failure, or on a quota failure, the experiment ends with what exists. A failing run is evidence too.
  - Either way, write the exit status, the start and end times, the number of launches and every move into the ledger.

- [ ] **Step 9: Archive the workdir**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
tar -czf $R/d2a-lasso.tar.gz -C $R d2a-lasso
[ -d $R/d2a-lasso-restart1 ] && tar -czf $R/d2a-lasso-restart1.tar.gz -C $R d2a-lasso-restart1
shasum -a 256 $R/d2a-lasso*.tar.gz      # into the ledger; the report recomputes the first
```

No commit: the repository did not change.

---

### Task 6: Report, evidence, ledger and docs (controller only; owner decision at Step 3)

**Files:**
- Create: `reconstruction/evidence/d2a-lasso/report.md`, `report.json`, `workdir/…` (copied by the script)
- Modify: `reconstruction/GAPS.md` (§5: new "D2a run" subsection)
- Modify: `reconstruction/README.md` (status table row "Runs with real LLM agents")

**Interfaces:**
- Consumes: Task 5's workdir and archive, Task 4's `host.json`, and Task 3's `report_run.py`.
- Produces: the committed evidence and the ledger's headline numbers. D2b opens on `evidence/d2a-lasso/report.json`.

- [ ] **Step 1: Check the task tree, then write the report and copy the evidence**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
(cd $R/d2a-lasso && shasum -a 256 -c --quiet $R/task.sha256) && echo "task tree unchanged"
python scripts/report_run.py --workdir $R/d2a-lasso --out evidence/d2a-lasso \
  --host-json evidence/d2a-lasso/host.json --archive $R/d2a-lasso.tar.gz --copy-evidence
```

Expected: `task tree unchanged`, then `evidence/d2a-lasso/report.md`. If any task file changed, an agent edited the task mid-run. Stop and show the owner the changed files before reporting: every comparison against the baseline depends on them. If Task 5 restarted, report on the aside directory too. It is a self-contained workdir (Task 5, Step 8):

```bash
python scripts/report_run.py --workdir $R/d2a-lasso-restart1 --out evidence/d2a-lasso/restart1 \
  --archive $R/d2a-lasso-restart1.tar.gz --copy-evidence
```

- [ ] **Step 2: Scan the evidence before anything is staged**

```bash
. /Users/Shared/dream-rsi-runs/env.sh
python3 - <<'EOF'
import os, subprocess
root = "evidence/d2a-lasso"
email = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True).stdout.strip()
checks = {
    "CPP_CODE (a pasted program)": "CPP_CODE",
    "#include (quoted source)": "#include",
    "home path": "/Users/",
    "owner email": email or "\0",
}
hits = {k: [] for k in checks}
big, n = [], 0
for d, _, files in os.walk(root):
    for f in files:
        p = os.path.join(d, f)
        n += 1
        if os.path.getsize(p) > 500_000:
            big.append(p)
        text = open(p, errors="replace").read()
        for k, needle in checks.items():
            if needle in text:
                hits[k].append(p)
print("files", n, "over 500 KB", big)
for k, ps in hits.items():
    print(f"{k}: {len(ps)}", ps[:5])
EOF
find evidence -name init_program.py        # expected: nothing
```

- [ ] **Step 3: Owner decision on the scan (stop point).** Show the owner the counts. The repository is public, and the pre-commit hook `check-added-large-files` refuses files over 500 KB. The options:
  - (a) commit as is;
  - (b) rewrite the home prefix `/Users/controlroom` to `~` in the copied evidence (the archive's sha256 still covers the originals);
  - (c) drop the files that quote source or exceed 500 KB, listing them in the GAPS subsection.

  Recommend (b) when home paths are the only hits. Apply the owner's choice with a script, then re-run Step 2.

- [ ] **Step 4: Read the report and write the ledger subsection.** Read `evidence/d2a-lasso/report.md` with the Read tool; it contains tables. Then insert the subsection at the end of GAPS §5.

In `reconstruction/GAPS.md`, replace:

````markdown
that noise without any estimate of it.

## 6. Reading the reported results
````

with the text below. Fill every angle-bracketed slot from `report.json` or `host.json`, at `report.md`'s precision. Write "none" for an empty list.

````markdown
that noise without any estimate of it.

**D2a run (2026-09-25).** One real-agent run on the Lasso task on a second machine, <host.cpu> (<host.cpus> threads), <host.compiler>, Eigen <host.eigen>, SimpleTES <first 12 characters of host.simpletes_commit>. Gemini CLI <agents.discovery.version> (model `auto-gemini-2.5`) drove discovery and Claude CLI <agents.policy.version> policy development, both isolated from the owner's global CLI configuration. It took <launches> launch(es), with a 4 x 3 fallback grid (16 calls), 6 x 4 hard caps (30 calls) and three policy versions per iteration. The full report and the evidence subset are in `evidence/d2a-lasso/`. The workdir archive stays off the repository (sha256 `<archive.sha256>`).

- Per-round calls: <per iteration: planned grid, used fallback, probes> against 16 and 30 here, and the paper's 110 and 640.
- Out-of-support replay: <len(out_of_support.flagged)> of <len(versions)> versions replayed on clipped episodes (<version: clipped/episodes, asked vs recorded>); <len(out_of_support.flagged_deployed)> of them deployed.
- Untouched programs: <len(untouched.cases)> of <untouched.attempts> attempts left their resume source byte for byte (<per case: iteration, cell, source, fail class, agent timed out, score vs source score>).
- Empty batches: versions <empty_batches.versions>; live iterations <empty_batches.live>.
- Health: <per iteration: attempts, successes, fail classes, agent timeouts, no program, evaluator crashes, baseline and best score>.
- Smoke test on this host (`host.json`, two runs each): <per program: valid counts, geo-mean ms, spread>, against the Xeon's 2% above.

## 6. Reading the reported results
````

- [ ] **Step 5: Update the README status row**

In `reconstruction/README.md`, replace:

````markdown
| Runs with real LLM agents | Gemini CLI or other | wired (`scripts/run_dream_rsi.py`), not run here |
````

with:

````markdown
| Runs with real LLM agents | Gemini CLI or other | wired (`scripts/run_dream_rsi.py`); run once on Lasso (track D2a: Gemini CLI discovery, Claude CLI policy development, <iterations completed> of 2 iterations; `evidence/d2a-lasso/report.md`) |
````

- [ ] **Step 6: Run the gate, then commit**

The gate count stays 118; Task 6 adds no test. `git status --short` should show only `reconstruction/evidence/d2a-lasso/` (new), `GAPS.md` and `README.md`. Check `git diff --stat` first. Write the commit message with the headline counts from `report.json` filled into this frame:

```text
Commit the D2a run's evidence and report

One real-agent Lasso run on this Mac (Gemini CLI discovery, Claude CLI
policy development, both isolated): <n> iteration(s), <probes> discovery
calls. Of D2's four questions: <clipped versions> version(s) replayed on
clipped episodes, <untouched> of <attempts> attempts left their resume
source untouched, <empty> version(s) ended on an empty batch, and every
round stayed inside its caps. GAPS §5 records the numbers; the evidence
holds no program.
```

```bash
git add reconstruction/evidence/d2a-lasso reconstruction/GAPS.md reconstruction/README.md
git commit -F /tmp/d2a-t6-msg.txt
```

Write the last sentence of the message only if `report.json` shows no round with more probes than its grid allows. Otherwise state what happened.

---

## After the tasks

Run the whole-branch review (SDD's final reviewer, on the most capable model) against the spec and this plan, then use the finishing menu. The PR targets `main`. D2b's brainstorm opens on `evidence/d2a-lasso/report.json`.
