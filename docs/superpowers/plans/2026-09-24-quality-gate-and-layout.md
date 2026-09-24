# Quality Gate and Layout (Track A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `reconstruction/` an installable package whose 38 tests are a hard floor enforced by lint, type, extraction-integrity and no-skip checks on every push, without moving anything.

**Architecture:** Plugin root = repo root (nothing moves). `reconstruction/pyproject.toml` packages `see` as distribution `dream-rsi`; ruff/pyright/pre-commit are configured there; a GitHub Actions matrix installs the editable package, regenerates `generated/` from the PDF, checks its digests against a committed manifest, runs pytest and fails on any skip. Three duplicated `spec_from_file_location` sites collapse into one guarded helper.

**Tech Stack:** Python ≥ 3.10, hatchling, pytest, ruff 0.16.8, pyright 1.1.414, pre-commit, GitHub Actions (`actions/checkout@v7`, `actions/setup-python@v7`).

**Spec:** `docs/superpowers/specs/2026-09-24-quality-gate-and-layout-design.md` (evidence base: `docs/superpowers/research/2026-09-24-maturity-audit.md`).

## Global Constraints

- All commands run from `reconstruction/` inside its venv (`.venv/`, created in Task 1) unless a step says otherwise.
- `requires-python = ">=3.10"`; distribution name `dream-rsi`; import package `see` — the import paths `see.policy.api` and `see.policy.observation_signal` are frozen (the paper's Listing 2 prompt imports them) and must not be renamed.
- `generated/` is never committed and never packaged (the paper's text, © Google). Editable install only; no wheel or sdist is built.
- Test count is a hard floor: 38 today → 47 at the end of this plan (+2 Task 4, +4 Task 5 junit, +3 Task 5 manifest). Every task ends with the current expected count green and zero skips (with `generated/` present).
- ruff: `line-length = 100`, `select = ["E","F","I","UP","B","RUF"]`, `tools/extract_listings.py` ignores `RUF001`. pyright `basic`. mypy is not used.
- Owner's machine: the *system* interpreter has a broken global pytest plugin; inside `.venv/` no workaround is needed. Outside it, prefix `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Commit messages: plain imperative sentences like the existing history (no `feat:` prefixes), **no `Co-Authored-By` or other agent-attribution trailers** (owner's rule).
- Work on branch `spec/quality-gate-and-layout` (or a worktree created from it). Never push without the owner's explicit go-ahead; Task 7 asks for it.
- Spec deviations recorded in this plan: (a) `scripts/verify_lasso.py` has no `sys.path` insert to remove (spec §3.2 lists it; only `tests/conftest.py` and `scripts/run_dream_rsi.py:16` have one); (b) `--check` compares digests in memory instead of writing to a temp dir (same guarantee, less code); (c) hand lint fixes and pyright fixes are two commits, not one; (d) the three `spec_from_file_location` sites share one helper with one reachable-path test, because the `tasks.py` and `verify_lasso.py` branches are unreachable with their hard-coded `.py` names; (e) pyright runs as a pre-commit `local`/`system` hook rather than the `pyright-python` mirror, whose isolated environment cannot resolve `numpy` or `pymupdf`; (f) CI expects 47 tests, not the spec's 38, because the guard and tool tests this plan adds are counted; (g) the drift self-test corrupts the manifest rather than a file in `generated/` — `--check` re-extracts from the PDF and compares with the manifest, it never reads `generated/`; `shasum -a 256 -c` from inside `generated/` is the on-disk check; (h) the gate installs the `lasso` extra (`pip install -e ".[extract,lasso,dev]"` locally and in CI) because pyright must resolve `numpy` and `scikit-learn` to check `scripts/verify_lasso.py` — the earlier "0 errors" baselines were resolving them through the system interpreter on PATH, which CI does not have; (i) the pre-commit pyright hook's entry is `reconstruction/.venv/bin/pyright --project reconstruction` rather than a bare `pyright`, so commits from a shell without the venv activated (every agent commit) still run it; (j) the generic pre-commit hooks (whitespace, end-of-file, YAML, large files) run repo-wide with `assets/` and `papers/` excluded rather than scoped to `reconstruction/`; (k) `.claude/settings.json` is narrower than spec §7's literal list — the arbitrary-`python` and `pre-commit *` patterns are replaced by pytest-only and install/run-only patterns to honour §7's stated intent ("exactly the gate commands"); (l) `pymupdf` is pinned exactly because the digest manifest depends on its extraction, and the pyright hook passes `--pythonpath` so it never resolves imports through a system interpreter.

---

### Task 1: Packaging — `pyproject.toml`, venv, no path hacks, install docs

**Files:**
- Create: `reconstruction/pyproject.toml`
- Delete: `reconstruction/requirements.txt`, `reconstruction/tests/conftest.py`
- Modify: `reconstruction/scripts/run_dream_rsi.py:11-20`, `reconstruction/README.md:23-33`, `.claude/CLAUDE.md:16-34`, `.gitignore` (root)

**Interfaces:**
- Consumes: nothing.
- Produces: an editable install where `import see` works from any cwd; `python -m see` and the `dream-rsi` console script; `[tool.ruff]`, `[tool.pyright]`, `[tool.pytest.ini_options]` that Tasks 2–7 rely on.

- [ ] **Step 1: Record the baseline (this is the "failing test" for the whole track)**

Run (from `reconstruction/`, system interpreter):
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests | tail -1
cd /tmp && python3 -c "import see" ; cd -
```
Expected: `38 passed` (if `5 skipped`, run `python3 tools/extract_listings.py` first — it needs pymupdf, which the system interpreter has). The second command must fail with `ModuleNotFoundError: No module named 'see'` — that is the defect this task removes.

- [ ] **Step 2: Write `reconstruction/pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "dream-rsi"
version = "0.1.0"
description = "Independent reconstruction of the Dream-RSI exploration layer from the paper"
readme = "README.md"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
extract = ["pymupdf>=1.24"]
lasso = ["numpy", "scikit-learn", "psutil"]
dev = ["pytest", "ruff==0.16.8", "pyright==1.1.414", "pre-commit"]

[project.scripts]
dream-rsi = "see.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["see"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "RUF"]

[tool.ruff.lint.per-file-ignores]
"tools/extract_listings.py" = ["RUF001"]  # the glyph map must contain those characters

[tool.pyright]
include = ["see", "scripts", "tools", "tests"]
pythonVersion = "3.10"
typeCheckingMode = "basic"
```

The `license` field is deliberately absent (owner decision pending, spec §2). `ruff` and `pyright` are pinned to the exact versions the pre-commit hooks in Task 6 use, so local, pre-commit and CI results agree.

- [ ] **Step 3: Create the venv and install**

```bash
cd reconstruction
python3 -m venv .venv && . .venv/bin/activate
python -m pip install -e ".[extract,dev]"
python -c "import see, sys; print(see.__file__, sys.version.split()[0])"
```
Expected: the path printed ends in `reconstruction/see/__init__.py` (editable, not site-packages) and the version is 3.13.x. `.venv/` is already gitignored at the root.

- [ ] **Step 4: Delete the path hacks**

```bash
git rm -q requirements.txt tests/conftest.py
```
In `scripts/run_dream_rsi.py` replace lines 11–20:
```python
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from see.live import CommandAgent  # noqa: E402
from see.loop import DreamRSI, LoopConfig  # noqa: E402
from see.tasks import SIMPLETES_TASKS, simpletes_task  # noqa: E402
```
with:
```python
import argparse
import json

from see.live import CommandAgent
from see.loop import DreamRSI, LoopConfig
from see.tasks import SIMPLETES_TASKS, simpletes_task
```
(`os` and `sys` are used only by the removed line; check with `grep -n 'os\.\|sys\.' scripts/run_dream_rsi.py` — if either is still used, keep that import.)

- [ ] **Step 5: Verify the package works from anywhere and the venv needs no pytest workaround**

```bash
python -m pytest -q | tail -1                 # from reconstruction/, no env var
(cd /tmp && python -m pytest -q "$OLDPWD/tests" | tail -1)
(cd / && python -m see --help | head -1)
dream-rsi --help | head -1
```
Expected: `38 passed` twice; both `--help` calls print `usage: python -m see [-h] {sweep,demo} ...`. If the first pytest crashes with a `langsmith`/pydantic traceback, the venv was not activated — activate it; do not add the env var to any file.

- [ ] **Step 6: Update the install docs**

`reconstruction/README.md`: replace the "Use" code block (the one starting `cd reconstruction`) and the sentence at line 23 ("The core package is standard-library Python.") so the section reads:

````markdown
The core package is standard-library Python; `pyproject.toml` extras cover the tools
(`extract`), the SimpleTES scripts (`lasso`) and the quality gate (`dev`). The package is
installed in editable mode only: `see/prompts.py` finds `generated/` next to the source tree, so
a wheel install cannot locate the prompts.

## Use

```bash
cd reconstruction
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[extract,dev]"           # add ,lasso for scripts/verify_lasso.py
python tools/extract_listings.py          # prerequisite: writes generated/ (2 prompts + the Lasso solver)
python -m pytest -q                       # 38 tests; 5 of them skip while generated/ is missing
python -m see demo --workdir /tmp/drsi    # whole loop on a toy task, scripted agents
```
````

`.claude/CLAUDE.md`: replace the "Commands" intro and code block (lines 16–28) with:

````markdown
## Commands

All from `reconstruction/`, inside its venv. The core package (`see/`) is standard-library Python;
the `extract`, `lasso` and `dev` extras in `pyproject.toml` cover the tools and scripts.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[extract,dev]"       # add ,lasso for scripts/verify_lasso.py
python tools/extract_listings.py      # PREREQUISITE: writes generated/ (2 prompts + Lasso solver)
python -m pytest -q                   # 38 tests, ~13s
python -m pytest -q tests/test_loop.py::test_on_policy_replay_reproduces_the_live_episode
python -m see demo --workdir /tmp/drsi                                          # whole loop, toy task, ~8s
python -m see sweep --method my_policy.py --pool runs/lasso/trace_pool --out /tmp/sweep
```
````

and replace the bullet that begins "`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed on this machine" with:

```markdown
- Use the venv. The system interpreter on this machine has a broken global pytest plugin
  (langsmith) that crashes collection; if you must run pytest outside the venv, prefix
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
```

Root `.gitignore`: append under `# Experiment outputs`:
```
SimpleTES/
```
(the README's `git clone ... ../SimpleTES` lands inside this repository).

- [ ] **Step 7: Run the tests and commit**

```bash
python -m pytest -q | tail -1        # 38 passed
cd /Users/controlroom/Dream-RSI
git add reconstruction/pyproject.toml reconstruction/scripts/run_dream_rsi.py reconstruction/README.md .claude/CLAUDE.md .gitignore
git commit -m "Package reconstruction/ as dream-rsi and drop the sys.path hacks

pyproject.toml (hatchling, python >= 3.10, extras extract/lasso/dev,
dream-rsi console script) replaces requirements.txt. tests/conftest.py and
the path insert in scripts/run_dream_rsi.py go away because the editable
install makes see importable from any cwd. README and CLAUDE.md document
the venv-based install; SimpleTES/ is ignored at the root."
```
(`git rm` already staged the two deletions.)

---

### Task 2: Mechanical ruff autofix and format — one commit, no logic change

**Files:**
- Modify: every `.py` under `reconstruction/see`, `scripts`, `tools`, `tests` that ruff touches (no file is hand-edited in this task)

**Interfaces:**
- Consumes: `[tool.ruff]` from Task 1.
- Produces: a codebase where `ruff format --check .` passes and `ruff check .` reports only the hand-fix categories (`E501`, `E731`, `B905`, `RUF059`).

- [ ] **Step 1: See the starting count**

```bash
ruff check . --statistics | tail -3
```
Expected: about 88 findings, about 62 fixable (measured with ruff 0.16.5; 0.16.8 may differ by a few).

- [ ] **Step 2: Apply safe fixes, then format**

```bash
ruff check --fix .
ruff format .
ruff check . --output-format concise
```
Expected: `ruff format` reports files reformatted; the final `ruff check` lists only `E731` (2), `B905` (2), `RUF059` (1) and any `E501` the formatter could not shorten (likely none — long string literals are the only lines it cannot split). Anything else → stop and report; a safe fix should never leave a new category behind.

- [ ] **Step 3: Prove no behavior changed**

```bash
python -m pytest -q | tail -1        # 38 passed
git diff --stat | tail -1
git diff -- see/loop.py see/objective.py see/world.py
```
Expected: 38 passed. Read those three diffs in full — the files carry the paper's logic — and confirm every hunk is a rewrap, an `Optional[X]` → `X | None`, an import reorder, or a removed `# noqa`. Any hunk that adds, removes or reorders a statement means a fix was not safe: revert that file and report.

- [ ] **Step 4: Commit**

```bash
git add -A reconstruction/see reconstruction/scripts reconstruction/tools reconstruction/tests
git commit -m "Apply ruff autofixes and ruff format

Mechanical only: annotation modernization (Optional -> X | None, typing ->
collections.abc), import ordering, unused noqa removal and the formatter's
line wrapping at 100 columns. No statement was added, removed or reordered;
38 tests pass before and after."
```

---

### Task 3: Hand lint fixes (`E731`, `B905`, `RUF059`, residual `E501`)

**Files:**
- Modify: `reconstruction/scripts/verify_lasso.py` (`npz` lambda near line 78, `zip` near line 163), `reconstruction/see/policies/portfolio.py` (`cap` lambda near line 136), `reconstruction/tools/extract_listings.py` (`zip(OUTPUTS, texts)` in `main`), `reconstruction/tests/test_objective.py` (`report, execs = ...` near line 84)

Line numbers shifted in Task 2; locate each by the `ruff check . --output-format concise` output.

**Interfaces:**
- Consumes: Task 2's formatted tree.
- Produces: `ruff check .` clean.

- [ ] **Step 1: Replace the two lambda assignments with `def`s**

`scripts/verify_lasso.py`, in `downstream()`:
```python
    def npz(name):
        return tuple(np.load(os.path.join(task, "eval_data", name))[k] for k in "Xy")
```
`see/policies/portfolio.py`, in `plan_grid()`:
```python
        def cap(w, r):
            return (
                min(max(1, w), context.hard_max_branch_count),
                min(max(0, r), context.hard_max_refine_count),
            )
```

- [ ] **Step 2: Make both `zip` calls strict**

`scripts/verify_lasso.py` (per-program runs are paired with the glmnet baseline runs; both have `--repeats` entries):
```python
        ratios = [b["geo_mean_sol_ms"] / r["geo_mean_sol_ms"] for r, b in zip(runs, base, strict=True)]
```
`tools/extract_listings.py`, in `main()` (three names, three texts, guarded by `EXPECTED_LINES`):
```python
    for name, text in zip(OUTPUTS, texts, strict=True):
```

- [ ] **Step 3: Drop the unused unpacked variable**

`tests/test_objective.py`, in `test_plan_grid_restricts_replay_and_flags_out_of_support`:
```python
    _, execs = beta_sweep(Wide, [t], betas=(0.5,))
```

- [ ] **Step 4: Wrap any residual `E501` by hand**

For each line ruff still reports, break at the last comma or binary operator before column 100 and let `ruff format .` settle the indentation. Do not shorten identifiers or strings' content.

- [ ] **Step 5: Verify and commit**

```bash
ruff format . && ruff check . && python -m pytest -q | tail -1
git add -A reconstruction
git commit -m "Fix the lint findings ruff cannot autofix

Two lambda assignments become defs, both zip() calls are strict (their
operands are paired by construction), and an unused unpacked variable is
dropped."
```
Expected: `All checks passed!`, `38 passed`.

---

### Task 4: pyright basic clean — one guarded module loader, tested

**Files:**
- Create: `reconstruction/tests/test_loader.py`
- Modify: `reconstruction/see/loader.py`, `reconstruction/see/tasks.py` (`simpletes_task`), `reconstruction/scripts/verify_lasso.py` (`load_evaluator`, `downstream`, `main`), `reconstruction/scripts/run_dream_rsi.py` (`main`), `reconstruction/see/objective.py` (`beta_sweep`), `reconstruction/see/toy.py` (`ScriptedDiscoveryAgent.__call__`), `reconstruction/tests/test_objective.py` (`Illegal.solve`), `reconstruction/tools/extract_listings.py` (`extract`, `main`)

**Interfaces:**
- Consumes: Task 3's clean tree.
- Produces: `see.loader.load_module_from_path(name: str, path: str) -> types.ModuleType` — imports a file as a module, raises `ValueError(f"{path}: not a loadable Python module (expected a .py file)")` when the path has no loader. Task 5's tests import tool scripts with it.

- [ ] **Step 1: Write the failing tests**

`reconstruction/tests/test_loader.py`:
```python
"""Policy files are loaded by path; a non-Python path must fail with a clear error."""
import pytest

from see.loader import load_policy
from see.policy.api import LLMDesignedMethod

POLICY = '''
from see.policy.api import LLMDesignedMethod

NAME = "Probe"


class Probe(LLMDesignedMethod):
    pass
'''


def test_non_python_policy_path_fails_loudly(tmp_path):
    bad = tmp_path / "policy.txt"
    bad.write_text(POLICY)
    with pytest.raises(ValueError, match="policy.txt"):
        load_policy(str(bad))


def test_load_policy_returns_the_named_class(tmp_path):
    good = tmp_path / "method.py"
    good.write_text(POLICY)
    cls = load_policy(str(good))
    assert cls.__name__ == "Probe" and issubclass(cls, LLMDesignedMethod)
```

- [ ] **Step 2: Run them to verify the first fails for the right reason**

```bash
python -m pytest -q tests/test_loader.py
```
Expected: `1 failed, 1 passed`; the failure is `AttributeError: 'NoneType' object has no attribute 'loader'` (the audit's reachable crash), not an import error.

- [ ] **Step 3: Implement the helper in `see/loader.py`**

Replace the module body from `def load_policy` up to (not including) `def overrides_plan_grid` with:
```python
def load_module_from_path(name: str, path: str) -> types.ModuleType:
    """Import the file at ``path`` as module ``name`` without writing bytecode next to it."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path}: not a loadable Python module (expected a .py file)")
    mod = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep archives clean
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    return mod


def load_policy(path: str):
    """Return the class named by the module-level ``NAME`` in ``path``."""
    path = os.path.abspath(path)
    key = hashlib.sha1(path.encode()).hexdigest()[:10]
    mod = load_module_from_path(f"see_policy_{key}", path)
    name = getattr(mod, "NAME", None)
    cls = getattr(mod, name, None) if isinstance(name, str) else None
    if not (isinstance(cls, type) and issubclass(cls, LLMDesignedMethod)):
        raise TypeError(f"{path}: NAME must name an LLMDesignedMethod subclass, got {name!r}")
    return cls
```
and add `import types` to the imports (ruff will sort it).

- [ ] **Step 4: Run the loader tests**

```bash
python -m pytest -q tests/test_loader.py
```
Expected: `2 passed`.

- [ ] **Step 5: Use the helper at the other two sites**

`see/tasks.py`, in `simpletes_task`, replace the four lines from `spec = importlib.util.spec_from_file_location(` through `spec.loader.exec_module(mod)` with:
```python
    mod = load_module_from_path(f"simpletes_{name}_evaluator", os.path.join(local, "evaluator.py"))
```
add `from see.loader import load_module_from_path` to the imports and remove `import importlib.util` if nothing else uses it.

`scripts/verify_lasso.py`: add `from see.loader import load_module_from_path` after the `numpy` import; in `load_evaluator` replace the three `spec`/`mod` lines with
```python
    return load_module_from_path("lasso_evaluator", dst)
```
(delete the now-unreachable `return mod`); in `downstream` replace the three `spec`/`gr` lines with
```python
    # also pins OMP/BLAS threads to 1, as SimpleTES does
    gr = load_module_from_path("simpletes_generate_results", os.path.join(task, "generate_results.py"))
```
Remove `import importlib.util` if unused.

- [ ] **Step 6: The remaining pyright sites, each a two-line change**

`scripts/run_dream_rsi.py`, `scripts/verify_lasso.py` and `tools/extract_listings.py`, in each `main()`:
```python
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
```

`see/objective.py`, in `beta_sweep`, the `kw = dict(...)` line becomes
```python
    kw: dict[str, Any] = dict(use_plan=use_plan, max_rounds=max_rounds, beta1=beta1, beta2=beta2, record=record)
```
and `Any` joins the `from typing import ...` line (create the line if Task 2 removed it).

`see/toy.py`, in `ScriptedDiscoveryAgent.__call__`, replace the `branch, attempt = map(int, re.match(...).groups())` line with:
```python
        m = re.match(r"attempt_b(\d+)_a(\d+)", node)
        if m is None:
            raise ValueError(f"not an attempt directory: {node!r}")
        branch, attempt = map(int, m.groups())
```

`tests/test_objective.py`, in `test_crashing_or_illegal_policy_is_scored_minus_infinity`:
```python
        def solve(self, question, budget=None) -> SimResult:
            question.reset()
            question.probe_batch(["b0a5"])
            raise AssertionError("unreachable: the batch above is illegal and must raise")
```

`tools/extract_listings.py`, in `extract()`, directly after the `if number == 1:` block (the one that does `current = {}` and `listings.append(current)`) and before `if number in current:` insert:
```python
                if current is None:
                    raise ValueError(f"code row before line 1 (page {page_index + 1})")
```

- [ ] **Step 7: Verify everything**

```bash
pyright
ruff format . && ruff check .
python -m pytest -q | tail -1
python tools/extract_listings.py | head -c 120; echo
```
Expected: `0 errors, 0 warnings, 0 informations`; `All checks passed!`; `40 passed`; extraction still prints its JSON summary with `"lines": [28, 273, 847]`.

- [ ] **Step 8: Commit**

```bash
git add -A reconstruction
git commit -m "Make pyright basic pass; load modules by path through one guarded helper

see.loader.load_module_from_path replaces three copies of the
spec_from_file_location pattern and raises ValueError for a path with no
loader — --method policy.txt used to crash with AttributeError; two tests
pin it. The other fixes are annotation guards: __doc__ may be None, a
re.match may be None, an untyped kwargs dict, an illegal-policy test
override that now declares it never returns, and a listing row that
arrives before line 1."
```

---

### Task 5: Extraction integrity (`--check`, manifest) and the no-skip gate (`check_junit.py`)

**Files:**
- Create: `reconstruction/tools/check_junit.py`, `reconstruction/tools/generated.sha256`, `reconstruction/tests/test_check_junit.py`, `reconstruction/tests/test_extract_manifest.py`
- Modify: `reconstruction/tools/extract_listings.py` (`main`, new manifest functions), `.claude/CLAUDE.md` (Commands block)

**Interfaces:**
- Consumes: `see.loader.load_module_from_path` (Task 4).
- Produces: in `extract_listings.py` — `digests(texts: list[str]) -> dict[str, str]`, `read_manifest(path: str) -> dict[str, str]`, `write_manifest(texts: list[str], path: str) -> None`, `check_manifest(texts: list[str], path: str) -> list[str]` (names whose digest drifted), CLI flags `--check` / `--write-manifest` / `--manifest PATH`; in `check_junit.py` — `main(argv) -> int` exiting 1 on any skip/error/failure or a wrong test count. Task 7's CI calls both.

- [ ] **Step 1: Write the failing manifest tests**

`reconstruction/tests/test_extract_manifest.py`:
```python
"""The committed digest manifest catches any drift in the paper's regenerated listings."""
import os

from see.loader import load_module_from_path

TOOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "extract_listings.py"
)
el = load_module_from_path("extract_listings_under_test", TOOL)
TEXTS = ["one\n", "two\n", "three\n"]


def test_manifest_round_trips_digests(tmp_path):
    path = str(tmp_path / "generated.sha256")
    el.write_manifest(TEXTS, path)
    assert el.read_manifest(path) == el.digests(TEXTS)
    assert open(path).read().splitlines()[0].endswith("  exploration_prompt.md")


def test_check_reports_nothing_when_text_matches_the_manifest(tmp_path):
    path = str(tmp_path / "generated.sha256")
    el.write_manifest(TEXTS, path)
    assert el.check_manifest(TEXTS, path) == []


def test_check_names_the_listing_whose_text_changed(tmp_path):
    path = str(tmp_path / "generated.sha256")
    el.write_manifest(TEXTS, path)
    tampered = [TEXTS[0], TEXTS[1], TEXTS[2] + " "]
    assert el.check_manifest(tampered, path) == ["lasso_path_dream_rsi.py"]
```
(`extract_listings.py` imports `pymupdf` lazily inside `extract()`, so loading the module needs no PDF library.)

- [ ] **Step 2: Run them to verify they fail**

```bash
python -m pytest -q tests/test_extract_manifest.py
```
Expected: `3 failed` with `AttributeError: module 'extract_listings_under_test' has no attribute 'write_manifest'`.

- [ ] **Step 3: Add the manifest functions and flags to `tools/extract_listings.py`**

Add `import hashlib` to the imports and, after the `EXPECTED_LINES = [28, 273, 847]` line:
```python
MANIFEST = os.path.join(HERE, "generated.sha256")  # sha256sum format: "<hex>  <name>"


def digests(texts):
    return {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in zip(OUTPUTS, texts, strict=True)
    }


def read_manifest(path=MANIFEST):
    out = {}
    with open(path) as f:
        for line in f:
            hexdigest, _, name = line.strip().partition("  ")
            out[name] = hexdigest
    return out


def write_manifest(texts, path=MANIFEST):
    with open(path, "w") as f:
        for name, hexdigest in digests(texts).items():
            f.write(f"{hexdigest}  {name}\n")


def check_manifest(texts, path=MANIFEST):
    """Names whose regenerated text differs from the manifest; empty means no drift."""
    want = read_manifest(path)
    return [name for name, hexdigest in digests(texts).items() if want.get(name) != hexdigest]
```
Replace `main()` with:
```python
def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--manifest", default=MANIFEST)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="compare the extraction with the manifest, write nothing, exit 1 on drift",
    )
    mode.add_argument(
        "--write-manifest",
        action="store_true",
        help="also rewrite the manifest (only when the paper PDF is deliberately updated)",
    )
    args = ap.parse_args(argv)
    texts = extract(args.pdf)
    if [t.count("\n") for t in texts] != EXPECTED_LINES:
        sys.exit(
            f"unexpected listing sizes {[t.count(chr(10)) for t in texts]}; "
            f"expected {EXPECTED_LINES} (different PDF build?)"
        )
    if args.check:
        drift = check_manifest(texts, args.manifest)
        print(json.dumps({"manifest": args.manifest, "drift": drift}))
        sys.exit(1 if drift else 0)
    os.makedirs(args.out, exist_ok=True)
    for name, text in zip(OUTPUTS, texts, strict=True):
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    if args.write_manifest:
        write_manifest(texts, args.manifest)
    non_ascii = sorted({c for t in texts for c in t if ord(c) > 126})
    print(
        json.dumps(
            {"out": args.out, "files": OUTPUTS, "lines": EXPECTED_LINES, "non_ascii_left": non_ascii}
        )
    )
```
The default path (no flag) behaves exactly as before.

- [ ] **Step 4: Run the manifest tests, then create the real manifest**

```bash
python -m pytest -q tests/test_extract_manifest.py      # 3 passed
python tools/extract_listings.py --write-manifest       # writes generated/ and tools/generated.sha256
cat tools/generated.sha256
python tools/extract_listings.py --check; echo "exit=$?"
(cd generated && shasum -a 256 -c ../tools/generated.sha256)
```
Expected: three lines of `<64 hex>  <name>`; `--check` prints `"drift": []` and `exit=0`; `shasum -c` prints `OK` three times (the manifest is plain sha256sum format).

- [ ] **Step 5: Prove `--check` catches drift**

```bash
cp tools/generated.sha256 /tmp/manifest.bak
sed -i '' '1s/^./z/' tools/generated.sha256      # corrupt the first digest (macOS sed; GNU: sed -i '1s/^./z/')
python tools/extract_listings.py --check; echo "exit=$?"
cp /tmp/manifest.bak tools/generated.sha256
```
Expected: `"drift": ["exploration_prompt.md"]` and `exit=1`. Restore the manifest before continuing.

- [ ] **Step 6: Write the failing junit-gate tests**

`reconstruction/tests/test_check_junit.py`:
```python
"""CI must go red on a skipped test, a failed test, or a test count that quietly changed."""
import os

from see.loader import load_module_from_path

TOOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "check_junit.py"
)
cj = load_module_from_path("check_junit_under_test", TOOL)
XML = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
    '<testsuite name="pytest" errors="0" failures="{failures}" skipped="{skipped}" tests="{tests}">'
    "</testsuite></testsuites>"
)


def _report(tmp_path, **counts):
    path = tmp_path / "report.xml"
    path.write_text(XML.format(**counts))
    return str(path)


def test_a_skipped_test_fails_the_gate(tmp_path):
    report = _report(tmp_path, tests=38, skipped=5, failures=0)
    assert cj.main([report, "--expect", "38"]) == 1


def test_a_failed_test_fails_the_gate(tmp_path):
    report = _report(tmp_path, tests=38, skipped=0, failures=1)
    assert cj.main([report, "--expect", "38"]) == 1


def test_a_clean_report_with_the_expected_count_passes(tmp_path):
    report = _report(tmp_path, tests=38, skipped=0, failures=0)
    assert cj.main([report, "--expect", "38"]) == 0


def test_an_unexpected_test_count_fails_the_gate(tmp_path):
    report = _report(tmp_path, tests=37, skipped=0, failures=0)
    assert cj.main([report, "--expect", "38"]) == 1
```
The XML shape is what pytest writes (`<testsuites>` wrapping one `<testsuite>` with `tests`, `skipped`, `errors`, `failures` attributes).

- [ ] **Step 7: Run them to verify they fail**

```bash
python -m pytest -q tests/test_check_junit.py
```
Expected: an error at collection — `FileNotFoundError` for `tools/check_junit.py`.

- [ ] **Step 8: Write `tools/check_junit.py`**

```python
"""Fail when a pytest JUnit report has skips, errors, failures, or the wrong number of tests.

    python tools/check_junit.py report.xml --expect 47

CI runs this after pytest so a test that quietly skips (for example because
generated/ is missing) can never turn the build green.
"""
import argparse
import sys
import xml.etree.ElementTree as ET

COUNTS = ("tests", "skipped", "errors", "failures")


def summarize(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in COUNTS}


def problems(summary, expect):
    found = [f"{k}={summary[k]}" for k in ("skipped", "errors", "failures") if summary[k]]
    if summary["tests"] != expect:
        found.append(f"tests={summary['tests']} (expected {expect})")
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--expect", type=int, required=True, help="exact number of tests expected")
    args = ap.parse_args(argv)
    found = problems(summarize(args.report), args.expect)
    print("junit: " + (", ".join(found) if found else f"{args.expect} tests, no skips"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 9: Run the junit tests, then the whole suite through the gate**

```bash
python -m pytest -q tests/test_check_junit.py           # 4 passed
python -m pytest -q --junitxml=/tmp/report.xml | tail -1  # 47 passed
python tools/check_junit.py /tmp/report.xml --expect 47; echo "exit=$?"
```
Expected: `junit: 47 tests, no skips`, `exit=0`.

- [ ] **Step 10: Gate self-test — the skip case must go red (spec §8.3)**

```bash
mv generated /tmp/generated.bak
python -m pytest -q --junitxml=/tmp/report.xml | tail -1
python tools/check_junit.py /tmp/report.xml --expect 47; echo "exit=$?"
mv /tmp/generated.bak generated
```
Expected: `42 passed, 5 skipped`; `junit: skipped=5`; `exit=1`. Record these three lines in the commit message.

- [ ] **Step 11: Update CLAUDE.md and commit**

In `.claude/CLAUDE.md`'s Commands block change `# 38 tests, ~13s` to `# 47 tests, ~13s` and add, after the `extract_listings.py` line:
```
python tools/extract_listings.py --check   # digest drift vs tools/generated.sha256; exit 1 on drift
```
In `reconstruction/README.md` change `# 38 tests` to `# 47 tests`.

```bash
ruff format . && ruff check . && pyright && python -m pytest -q | tail -1
git add -A reconstruction .claude/CLAUDE.md
git commit -m "Pin the regenerated listings to a digest manifest and fail CI on skipped tests

extract_listings.py --check compares a fresh extraction with
tools/generated.sha256 (sha256sum format) and exits 1 on drift;
--write-manifest refreshes it when the PDF changes on purpose.
tools/check_junit.py reads pytest's JUnit XML and exits 1 on any skip,
error, failure or unexpected test count, so the five loop tests that skip
without generated/ can no longer pass a build silently. Gate self-test:
without generated/ the suite reports 42 passed, 5 skipped and the checker
exits 1."
```

---

### Task 6: pre-commit and project agent permissions

**Files:**
- Create: `.pre-commit-config.yaml` (repo root), `.claude/settings.json`
- Modify: `.claude/CLAUDE.md` (Commands block), `reconstruction/README.md` ("Use" block)

**Interfaces:**
- Consumes: ruff/pyright config from Task 1; a tree that already passes both (Tasks 2–5).
- Produces: hooks that run on every commit; an allow-list so agents run the gate commands unprompted.

- [ ] **Step 1: Write `.pre-commit-config.yaml`**

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v6.0.0
    hooks:
      - id: trailing-whitespace
        exclude: ^(assets/|papers/)
      - id: end-of-file-fixer
        exclude: ^(assets/|papers/)
      - id: check-yaml
      - id: check-added-large-files
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.16.8
    hooks:
      - id: ruff-check
        args: [--fix]
        files: ^reconstruction/
      - id: ruff-format
        files: ^reconstruction/
  - repo: local
    hooks:
      - id: pyright
        name: pyright (reconstruction)
        entry: pyright --project reconstruction
        language: system
        pass_filenames: false
        files: ^reconstruction/.*\.py$
```
pyright runs as a `local`/`system` hook on purpose: the mirror's isolated environment cannot resolve `numpy` or `pymupdf`, the venv's pyright can, and it is the same binary CI uses. The `ruff` revision matches the pin in `pyproject.toml`.

- [ ] **Step 2: Install and run on everything**

```bash
cd /Users/controlroom/Dream-RSI
reconstruction/.venv/bin/pre-commit install
reconstruction/.venv/bin/pre-commit run --all-files
```
Expected: every hook `Passed`, except that `trailing-whitespace` / `end-of-file-fixer` may modify root Markdown or YAML files on the first run — inspect `git diff`, keep only whitespace changes, run again, all `Passed`.

- [ ] **Step 3: Write `.claude/settings.json`**

```json
{
  "permissions": {
    "allow": [
      "Bash(python -m pytest*)",
      "Bash(python3 -m pytest*)",
      "Bash(pytest*)",
      "Bash(PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python*)",
      "Bash(ruff *)",
      "Bash(pyright*)",
      "Bash(pre-commit *)",
      "Bash(python tools/extract_listings.py*)",
      "Bash(python3 tools/extract_listings.py*)",
      "Bash(python tools/check_junit.py*)",
      "Bash(python -m see demo*)",
      "Bash(python3 -m see demo*)"
    ]
  }
}
```

- [ ] **Step 4: Document and commit**

Add to the Commands block of `.claude/CLAUDE.md`, after the `pip install` line:
```
pre-commit install                    # once per clone; ruff, ruff-format, pyright run on every commit
```
and the same line (without the comment's second clause) to the README "Use" block after `pip install`.

```bash
reconstruction/.venv/bin/pre-commit run --all-files && git add .pre-commit-config.yaml .claude/settings.json .claude/CLAUDE.md reconstruction/README.md
git commit -m "Add pre-commit hooks and project permissions for the gate commands

ruff, ruff-format and pyright run on every commit at the versions pinned in
pyproject.toml; .claude/settings.json lets agents run exactly the gate
commands without prompts."
```
(The commit itself exercises the hooks; if one rewrites a file, add it and commit again.)

---

### Task 7: CI workflow, pull request, fresh-clone acceptance

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `tools/extract_listings.py --check`, `tools/check_junit.py` (Task 5), the pinned dev extra (Task 1).
- Produces: a green check on the pull request; the track's acceptance evidence.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push:
    branches: [main]
  pull_request:
concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true
jobs:
  gate:
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest, python: "3.10" }
          - { os: ubuntu-latest, python: "3.13" }
          - { os: macos-latest,  python: "3.13" }
    runs-on: ${{ matrix.os }}
    defaults:
      run:
        working-directory: reconstruction
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python }}
          cache: pip
          cache-dependency-path: reconstruction/pyproject.toml
      - run: python -m pip install -e ".[extract,dev]"
      - run: ruff check . && ruff format --check .
      - run: pyright
      - run: python tools/extract_listings.py --check
      - run: python tools/extract_listings.py
      - run: python -m pytest -q --junitxml=report.xml
      - run: python tools/check_junit.py report.xml --expect 47
```

- [ ] **Step 2: Validate the YAML and commit**

```bash
reconstruction/.venv/bin/pre-commit run check-yaml --files .github/workflows/ci.yml
git add .github/workflows/ci.yml
git commit -m "Run the quality gate in GitHub Actions

Three jobs (ubuntu 3.10, ubuntu 3.13, macos 3.13): editable install, ruff
check and format, pyright, extraction digest check, extraction, pytest
with a JUnit report, and check_junit.py so a skipped test fails the build."
```

- [ ] **Step 3: Ask the owner, then push and open the pull request**

Pushing is outward-facing: confirm with the owner first ("Ready to push `spec/quality-gate-and-layout` and open a PR against `main` so CI can run — go ahead?"). Then:
```bash
git push -u origin spec/quality-gate-and-layout
gh pr create --base main --title "Track A: quality gate and repository layout" --body-file docs/superpowers/specs/2026-09-24-quality-gate-and-layout-design.md
gh pr checks --watch
```
Expected: three `ci / gate (...)` checks pass. If a job fails, read its log (`gh run view --log-failed`), fix on the branch with a failing test where applicable, push again. Do not weaken a check to make it pass.

- [ ] **Step 4: Fresh-clone acceptance test (spec §8.5)**

From the scratchpad directory (not inside the repo):
```bash
git clone --quiet --branch spec/quality-gate-and-layout https://github.com/crichalchemist/Dream-RSI fresh-dream-rsi
cd fresh-dream-rsi/reconstruction
python3 -m venv .venv && . .venv/bin/activate
pip install -q -e ".[extract,dev]"
python tools/extract_listings.py > /dev/null
python -m pytest -q --junitxml=report.xml | tail -1
python tools/check_junit.py report.xml --expect 47
```
Expected: `47 passed` and `junit: 47 tests, no skips`. Only the README's commands were used.

- [ ] **Step 5: Record the evidence**

Append to the pull request description (`gh pr edit --body-file -` or the web UI) a short "Verification" section with the four observed results: the mechanical-commit diff stat (Task 2), the skip self-test lines (Task 5 step 10), the drift self-test (Task 5 step 5), and the fresh-clone output. Track A is complete when the owner merges.
