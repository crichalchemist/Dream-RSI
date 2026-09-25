# Dream-RSI, reconstructed from the paper

An independent reimplementation of the Dream-RSI exploration layer, built only
from `papers/Dream-RSI.pdf` and the public SimpleTES benchmark. It is not the
authors' code, which is still unreleased. Everything the paper leaves
unspecified is listed with the choice made here in [GAPS.md](GAPS.md). Read
that file before trusting any number this code produces.

## Status

| Part | Source | State |
|---|---|---|
| Replay simulator and policy API (`see.policy.api`, `see.policy.observation_signal`) | Sec. 3, Listing 2 | implemented and tested; same import paths as the paper's prompt |
| Objectives: Eq. (1) and the beta-sweep `pareto.reward` | Sec. 3, Listing 2 | implemented; AUC/attainment details inferred and pinned (`tests/test_ledger.py`); `pareto` scores the beta grid, `eq1` the default-beta episode |
| Parallel-refine initial policy | Sec. 4 | implemented |
| Online rollout: workspaces, parallel workers, agent + evaluator | Sec. 3, Listing 1 | implemented; tested with scripted agents and the real Lasso evaluator; an agent timeout, an interrupt or a fault elsewhere in the batch kills the agent's whole process group, and nothing an agent forked outlives its call (a child that calls `setsid` escapes; GAPS §3 lists the residues); a timed-out attempt is scored as the program it left but recorded as a `timeout` failure |
| Outer loop: online, pool, M versions, argmax deploy | Sec. 3, Fig. 1 | implemented; tested end to end with scripted agents; replay ≡ live pinned across seeds and both policies; deploy integrity verified by digest when a version is scored, when it is deployed and again when it is loaded; `state.json` is written once per iteration, after the deploy; an interrupt freezes the partial tree under `runs/`, never the pool, and a restart refuses |
| Prompts (Listings 1, 2) and discovered Lasso solver (Listing 3) | appendix | regenerated from the PDF, verbatim |
| Lasso and math tasks | SimpleTES | adapter (`see/tasks.py`) |
| KernelBench tasks | KernelBench | no adapter |
| Runs with real LLM agents | Gemini CLI or other | wired (`scripts/run_dream_rsi.py`), not run here |

The core package is standard-library Python; `pyproject.toml` extras cover the tools
(`extract`), the SimpleTES scripts (`lasso`) and the quality gate (`dev`). The package is
installed in editable mode only: `see/prompts.py` finds `generated/` next to the source tree, so
a wheel install cannot locate the prompts.

## Use

```bash
cd reconstruction
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[extract,lasso,dev]"
pre-commit install
python tools/extract_listings.py          # prerequisite: writes generated/ (2 prompts + the Lasso solver)
python -m pytest -q                       # whole suite; exact count pinned in .github/workflows/ci.yml
python -m see demo --workdir /tmp/drsi    # whole loop on a toy task, scripted agents
```

Check the paper's Lasso solver against SimpleTES's own evaluator. It needs a `g++` with
OpenMP first on `PATH` and Eigen 3: on Linux, `g++` and `libeigen3-dev`; on macOS, where
Apple's `g++` is clang without OpenMP, MacPorts `gcc13` (`sudo port select --set gcc mp-gcc13`)
and `eigen3`, passed as `--eigen-include /opt/local/include/eigen3` (the directory holding
`Eigen/`; it covers the search score, not `--downstream`):

```bash
git clone --depth 1 https://github.com/wq-will/SimpleTES ../SimpleTES
python scripts/verify_lasso.py --simpletes ../SimpleTES --repeats 2  # macOS: --eigen-include DIR
```

Run the loop on a paper task with real coding agents. This spends real API
budget: about 110 discovery calls per round at the paper's 3.1-Pro setting. On macOS add
`--eigen-include` as above.

```bash
python scripts/run_dream_rsi.py --simpletes ../SimpleTES --task lasso_path \
    --workdir runs/lasso --discovery-agent gemini --policy-agent gemini
```

Replay-score any policy file over an existing trace pool:

```bash
python -m see sweep --method my_policy.py --pool runs/lasso/trace_pool --out /tmp/sweep
```

## Layout

```
tools/extract_listings.py   PDF -> generated/ (glyph-coordinate recovery)
see/policy/api.py           Observation, CellMeta, SimResult, GridPlan, LLMDesignedMethod, ...
see/policy/observation_signal.py   success semantics, failure taxonomy, helper signals
see/world.py                Trace (frozen tree) and the question API; ReplayQuestion
see/objective.py            Eq. (1), attainment, Pareto AUC, beta sweep
see/policies/               parallel_refine.py (pi_1), portfolio.py (example adaptive policy)
see/live.py                 LiveQuestion: agent + evaluator per probe, attempt_* workspaces
see/loop.py                 DreamRSI outer loop
see/pool.py, loader.py      trace pool and policy-file loading
see/tasks.py                SimpleTES task adapter
see/toy.py, synthetic.py    toy task, scripted agents, synthetic trees (tests/demo only)
scripts/                    verify_lasso.py, run_dream_rsi.py
```

`generated/` is not committed. It holds the paper's own text (© 2026 Google),
reproduced from the PDF already in this repository.
