# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Orientation

Two layers. The root is the paper's public project page (`README.md`, `papers/Dream-RSI.pdf`,
`CITATION.cff`, `assets/`). `reconstruction/` is an independent reimplementation of the paper's
exploration layer, built from the PDF and the public SimpleTES benchmark only — the authors' code is
unreleased. Everything the paper leaves unspecified, and every place it contradicts itself, is
recorded in `reconstruction/GAPS.md`. Read it before trusting any number this code produces.

`reconstruction/generated/` holds the paper's own listings (© Google), regenerated locally from
the PDF. It is gitignored and must never be committed.

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

- Without `generated/`, `see demo` and `see/prompts.py` raise `FileNotFoundError` and 5 tests in
  `tests/test_loop.py` skip (33 passed, 5 skipped instead of 38 passed).
- Use the venv. The system interpreter on this machine has a broken global pytest plugin
  (langsmith) that crashes collection; if you must run pytest outside the venv, prefix
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Paper-task scripts need `git clone --depth 1 https://github.com/wq-will/SimpleTES ../SimpleTES`
  plus g++, OpenMP and system Eigen (`libeigen3-dev`). `scripts/verify_lasso.py` scores Listing 3
  with SimpleTES's evaluator; `scripts/run_dream_rsi.py` runs the loop with real coding agents and
  spends real API budget (~110 discovery calls per round at the paper's default grid).

## Architecture

**Outer loop** — `see/loop.py:DreamRSI.run` alternates two stages per iteration `t`:
1. `online(t)`: load the deployed policy (`see/loader.py:load_policy`), ask it for a grid via
   `plan_grid` (validated by `see/objective.py:validate_plan`), then a `LiveQuestion` drives the
   discovery agent + evaluator over that grid. The finished tree is frozen into
   `trace_pool/iterNNNN/` as `trace.json` + `live_cycle_manifest.json`.
2. `offline(t)`: build M policy versions (`LoopConfig.versions`; version 0 is the current policy,
   so the deployed policy never regresses on the pool). The policy-development agent edits
   `policy_dev/method.py`; each version is replay-scored over every tree in the pool in a
   subprocess with a timeout (crash or illegal batch scores −∞); the argmax is copied to
   `deployed/iterNNNN.py` and `state.json` is updated.

Workdir layout is documented in the `see/loop.py` module docstring; the directory names are the
ones the paper's Listing 2 prompt quotes, so do not rename them.

**One `question` API, two backends.** A policy interacts only with a `Question`. `see/world.py`
provides `Trace` (the frozen tree: root plus disjoint branch chains, cell `(branch, attempt)`) and
`ReplayQuestion`; `see/live.py` provides `LiveQuestion`, which answers the same calls by running a
real agent in `runs/iterNNNN/tree/attempt_*/` workspaces. The policy cannot tell them apart, which
is what makes replay evaluation valid. Replay only reveals what the recorded tree contains — a
policy that goes wider or deeper than the behaviour policy is truncated (GAPS.md §6).

**Policy contract is frozen.** `see/policy/api.py` (`LLMDesignedMethod`, `GridPlan`,
`GridPlanningContext`, `Observation`, `CellMeta`, `SimResult`) and
`see/policy/observation_signal.py` are the import paths the paper's Listing 2 prompt tells the
LLM to use. LLM-written policies in `policy_dev/method.py` import them by name; renaming anything
public there breaks every generated policy. `see/policies/parallel_refine.py` is the paper's
initial policy π₁; `see/policies/portfolio.py` is a hand-written stand-in for LLM output.

**Two objectives that disagree.** `see/objective.py` implements both the paper's Eq. (1)
(`eq1_value`) and Listing 2's beta-sweep `pareto.auc − λ·parallel_penalty` (`beta_sweep`).
`LoopConfig.objective` selects one; default `"pareto"`. The paper never says which selected its
reported policies (GAPS.md §4).

**Agents and tasks are argv and adapters.** `see/live.py:CommandAgent` runs any CLI whose argv
contains `"{prompt}"`; `AGENT_PRESETS` has `gemini` and `claude`. `see/live.py:TaskSpec` is the
task interface; `see/tasks.py` adapts SimpleTES tasks (AGPL, not vendored); `see/toy.py` and
`see/synthetic.py` exist for tests and the demo only. `see/prompts.py` instantiates the two
prompts from `generated/`, which is why extraction is a prerequisite.

## Conventions

- Every choice the paper does not specify gets a row in `GAPS.md` §3; every contradiction found
  in the paper goes in §4. Keep the README status table in step with what is actually implemented.
- Test names state the paper claim or business outcome they protect
  (`test_eq1_matches_the_paper_formula`, `test_broken_versions_score_minus_infinity_and_are_never_deployed`).
  Keep that style.
- Run everything from `reconstruction/`; `tests/conftest.py` and the scripts put it on `sys.path`.
- `LoopConfig.serialize_eval=True` by default: evaluations are serialized because timing-based
  tasks interfere with each other. Do not parallelize evaluation for Lasso or kernel tasks.
- An interrupted iteration cannot be resumed: `online()` raises if `runs/iterNNNN/` already
  exists. Delete the partial directory, do not merge into it.
- `*.local.md` files are private maintainer notes and are gitignored.
