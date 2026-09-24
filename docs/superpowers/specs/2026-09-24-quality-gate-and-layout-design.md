# Quality gate and repository layout (track A)

Date: 2026-09-24. Status: approved design, awaiting implementation plan.

## 1. Context

`reconstruction/` is an independent reimplementation of the Dream-RSI exploration layer,
produced in one cloud coding session and merged as-is (PR #1, commit c98fff5). A six-lens audit
on 2026-09-24 (`docs/superpowers/research/2026-09-24-maturity-audit.md`) found the mechanical
core faithful and deterministic but the outer loop untrustworthy under real conditions, the test
suite thin where it matters, and no packaging, lint, type check or CI at all. Its first
recommendation, adopted here: *nothing else is safe to change until the existing 38 tests are a
hard floor on a clean clone and CI runs the extraction step.*

The owner's goals for the codebase, in order of dependency: a trustworthy platform, a Claude
Code plugin that dreams over Claude's own history (fed by claude-remember's data), reproduction
of the paper's numbers, and extension of the method. Track A is the foundation for all four.

Decisions already made with the owner during brainstorming:

- Repo layout: **plugin root = repo root** (approach 1). Nothing moves. `reconstruction/` stays
  the Python project; plugin directories (`.claude-plugin/`, `commands/`, `skills/`, `hooks/`)
  join the root in track P1, not now. A plugin install clones the whole repo (1.66 MiB packed),
  which the plugin needs anyway because `generated/` must be regenerated from the PDF at first run.
- Rejected: plugin in a `plugin/` subdirectory (forces `reconstruction/` and the PDF to move,
  because the plugin loader drops symlinks that leave the plugin directory and a `git-subdir`
  source would not contain `papers/`); a separate thin plugin repo (two repos in lockstep,
  network at first run). Approach 1 can evolve into the second later because track A makes the
  package installable regardless.
- Git submodules are ruled out as a distribution mechanism: `/plugin install` leaves submodule
  directories empty (verified on three installed official plugins).

## 2. Goals and non-goals

Goal: after track A, a fresh clone that follows the README reaches 38 green tests; CI enforces
lint, format, types, extraction integrity and "no skipped tests" on every push and pull request;
the package is installable in editable mode; agents can run the gate without permission prompts.

Non-goals (owned by other tracks):

- Fixing audit defects 3.1–3.6 (tracks B and C, each fix preceded by a failing test).
- Removing the `generated/`-dependent skip in `tests/test_loop.py:13` (track B, prompt injection).
- A `PromptSet` / environment override so a non-editable install can find `generated/` (track E).
- Plugin manifest, commands, hooks (track P1).
- A wheel or sdist build, a lock file, PyPI publication.
- A SimpleTES/Lasso CI job (track F: needs g++, OpenMP, Eigen and an AGPL clone).
- Documentation beyond the command lines that track A itself changes (track G).
- Choosing a license. `reconstruction/` has no license file and `pyproject.toml` leaves the
  `license` field unset; the owner must decide before anything is published.

## 3. Layout and packaging

### 3.1 `reconstruction/pyproject.toml`

PEP 621 metadata, `hatchling` build backend (pure Python, one line of configuration).

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
dev = ["pytest", "ruff", "pyright", "pre-commit"]

[project.scripts]
dream-rsi = "see.__main__:main"

[tool.hatch.build.targets.wheel]
packages = ["see"]
```

- Distribution name `dream-rsi`; import package `see`. The import path is frozen because the
  paper's Listing 2 prompt tells the policy-development agent to import `see.policy.api` and
  `see.policy.observation_signal`; `see` is also an existing PyPI distribution, so the two names
  must differ.
- `requires-python >= 3.10`: the code uses no 3.10-only runtime syntax today (54 `Optional[...]`
  annotations, all under `from __future__ import annotations`), but 3.8 and 3.9 are end-of-life,
  nothing older than 3.10 exists on the owner's machine, and the floor lets ruff's `UP` rules
  modernize annotations in the mechanical commit (section 4).
- `dependencies = []` states the fact that the core is standard-library only. Extras group the
  three optional tool sets exactly as `requirements.txt` did; `requirements.txt` is deleted so
  there is one source of truth.
- `dream-rsi` console script is additive; `python -m see` keeps working and stays in the docs.
- `generated/` stays outside `see/` and is never packaged: it is the paper's text (© Google),
  regenerated locally from `papers/Dream-RSI.pdf`. With an editable install `see/prompts.py`'s
  `__file__`-relative path still resolves. A non-editable install cannot find the prompts; track
  A documents that limitation in the README and builds no wheel.

### 3.2 Remove the `sys.path` hacks

`tests/conftest.py` (line 4), `scripts/run_dream_rsi.py:16` and `scripts/verify_lasso.py` insert
`reconstruction/` on `sys.path`. With the editable install they are unnecessary and they hide
import errors. Delete them; `tests/conftest.py` contains nothing else and is deleted.
The replay-sweep subprocess is already cwd-independent (`see/loop.py:38`, `:220`), so no other
code depends on the working directory.

### 3.3 Install lines

`reconstruction/README.md` "Use" and `.claude/CLAUDE.md` "Commands" become:

```bash
cd reconstruction
pip install -e ".[extract,dev]"          # add ,lasso for scripts/verify_lasso.py
python tools/extract_listings.py         # prerequisite: writes generated/
python -m pytest -q tests                # 38 tests
python -m see demo --workdir /tmp/drsi
```

The note that `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed on the owner's machine (a broken
globally installed pytest plugin) stays in `.claude/CLAUDE.md` only; it is not repository config.

### 3.4 `.gitignore`

Add `SimpleTES/` at the repo root: the README instructs `git clone ... ../SimpleTES`, which lands
inside this repository and is currently unignored.

## 4. Lint, format and types

### 4.1 ruff

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "RUF"]

[tool.ruff.lint.per-file-ignores]
"tools/extract_listings.py" = ["RUF001"]   # the glyph map must contain those characters
```

Today this configuration reports 88 findings, 62 auto-fixable. They are resolved in two commits:

1. **Mechanical commit:** `ruff check --fix .` then `ruff format .`, nothing else. Precondition
   and postcondition: 38 tests green. `git diff` contains no change to control flow. This is
   the only whole-codebase reformat; it is kept separate so later diffs stay reviewable.
2. **Hand fixes:** 13 `E501` (wrap), 2 `E731` (`def` instead of lambda assignment), 2 `B905`
   (`zip(..., strict=...)`), 1 `RUF059` (unused unpacked variable). No behavior change.

### 4.2 pyright

```toml
[tool.pyright]
include = ["see", "scripts", "tools", "tests"]
pythonVersion = "3.10"
typeCheckingMode = "basic"
```

29 errors today. Annotation-only fixes are free. Three findings change behavior — an
`importlib.util.spec_from_file_location` result used without a `None` check in
`see/loader.py:14`, `see/tasks.py:37` and `scripts/verify_lasso.py:55`, which today crash with
`AttributeError` on a non-`.py` path — and each gets a failing test first: a `--method policy.txt`
style input must raise `ValueError` naming the path. Those are the only tests track A adds.

mypy is not used (not installed; the audit chose pyright).

### 4.3 pre-commit

`.pre-commit-config.yaml` at the repo root, hooks scoped to `reconstruction/`:

- `astral-sh/ruff-pre-commit`: `ruff` (with `--fix`) and `ruff-format`.
- `RobertCraigie/pyright-python`: `pyright`.
- `pre-commit/pre-commit-hooks`: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`,
  `check-added-large-files`.

No pytest in pre-commit; CI owns tests. `pre-commit install` is documented in the README.

## 5. Tests as a floor

### 5.1 pytest configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

`-ra` prints every skip reason in every run, locally and in CI.

### 5.2 Extraction integrity: `extract_listings.py --check`

A committed manifest `reconstruction/tools/generated.sha256` lists the SHA-256 of each file the
tool produces (`exploration_prompt.md`, `policy_improvement_prompt.md`,
`lasso_path_dream_rsi.py`), one `<hex>  <name>` line each. `--check` extracts into a temporary
directory, compares each digest with the manifest, prints every mismatch and exits 1; it writes
nothing under `generated/`. `--write-manifest` regenerates the manifest (used once now and when
the paper PDF is deliberately updated). Default behavior (write `generated/`) is unchanged.
Extraction is already byte-reproducible with the pinned PyMuPDF line count guard, so the manifest
is stable.

### 5.3 No silent skips: `tools/check_junit.py`

A ~20-line script: `python tools/check_junit.py report.xml --expect 38` parses the JUnit XML
that pytest writes with `--junitxml`, and exits 1 if `skipped > 0`, `errors > 0`, `failures > 0`
or `tests != 38`. It has one test of its own (a fixture XML with one skip must fail). The
expected count is a CLI argument so it is updated deliberately when tests are added.

## 6. CI: `.github/workflows/ci.yml`

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
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip
          cache-dependency-path: reconstruction/pyproject.toml
      - run: pip install -e ".[extract,dev]"
      - run: ruff check . && ruff format --check .
      - run: pyright
      - run: python tools/extract_listings.py --check
      - run: python tools/extract_listings.py
      - run: python -m pytest -q --junitxml=report.xml
      - run: python tools/check_junit.py report.xml --expect 38
```

Three jobs. macOS is included because the owner develops on macOS and the audit found nothing
platform-specific yet; it is the cheapest way to keep it that way. No Lasso job (track F).

## 7. Agent guardrails

`.claude/settings.json` (project scope, committed) allows exactly the gate commands so agents
run them without prompts and nothing broader:

```json
{
  "permissions": {
    "allow": [
      "Bash(python -m pytest*)", "Bash(python3 -m pytest*)", "Bash(pytest*)",
      "Bash(PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python*)",
      "Bash(ruff *)", "Bash(pyright*)", "Bash(pre-commit *)",
      "Bash(python tools/extract_listings.py*)", "Bash(python3 tools/extract_listings.py*)",
      "Bash(python tools/check_junit.py*)",
      "Bash(python -m see demo*)", "Bash(python3 -m see demo*)"
    ]
  }
}
```

`.claude/CLAUDE.md` gains the new install line and the `--check` step; `.agents/AGENTS.md` and
`.gemini/GEMINI.md` are symlinks to it and follow automatically.

## 8. Verification

Track A is done when all of the following have been observed, not inferred:

1. **Baseline:** 38 tests green on the current tree (observed 2026-09-24).
2. **Mechanical commit:** 38 green before and after; `git diff --stat` lists only reformatted
   files; a reviewer reading the diff finds no control-flow change.
3. **Gate self-test, run locally and recorded in the implementation notes:**
   - delete `generated/`, run the CI step sequence *without* the two extraction steps →
     `check_junit.py` exits 1 reporting 5 skips; re-run extraction → green;
   - change one byte of `generated/exploration_prompt.md`, run `--check` → exits 1 naming the file;
   - run `check_junit.py` on a fixture XML with one failure → exits 1.
4. **CI:** the workflow is green on a pull request for all three matrix jobs.
5. **Fresh clone:** `git clone` into a scratch directory, follow only the README "Use" section,
   38 green. This is the acceptance test for the whole track.

## 9. Sequencing

Six commits, each leaving 38 tests green:

1. `pyproject.toml`; delete `requirements.txt`; remove `sys.path` hacks; README/CLAUDE.md
   install lines; `.gitignore` `SimpleTES/`.
2. Mechanical `ruff check --fix` + `ruff format`.
3. Hand lint fixes; pyright fixes with the three guard tests.
4. `extract_listings.py --check` / `--write-manifest`, `tools/generated.sha256`,
   `tools/check_junit.py` and its test.
5. `.pre-commit-config.yaml`; `.claude/settings.json`.
6. `.github/workflows/ci.yml`; open a pull request; verify the three jobs and the fresh-clone test.

## 10. Deferred items surfaced by this design

- Non-editable installs cannot locate `generated/` → track E (`PromptSet` with a path override
  or `importlib.resources`; the paper's text must never ship in a wheel).
- The `test_loop.py` skip should become prompt injection so the loop tests never skip → track B.
- License decision for `reconstruction/` → owner.
- `.claude/settings.json` allow-list will need `python -m see sweep` and the plugin commands when
  P1 lands.
