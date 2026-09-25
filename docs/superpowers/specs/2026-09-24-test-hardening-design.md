# Track B — Test hardening for `reconstruction/`

Design, 2026-09-24. Branch `spec/test-hardening` from `main` at d788c8f (track A merged as PR #2).
Source of the findings: `docs/superpowers/research/2026-09-24-maturity-audit.md`, re-verified
against d788c8f on 2026-09-24 (workflow `wf_4e4d9eaa-66a`; every line number below is as of
d788c8f, after track A's reformat).

## 1. Context

Track A made `reconstruction/` a packaged, lint- and type-clean tree behind a strict gate: any
skipped, errored or failed test, or a test count other than the one pinned in
`.github/workflows/ci.yml`, is red. The audit's verdict stands: the core (trace, replay,
objective) is faithful and deterministic; the outer loop is untrustworthy for real runs. Six
findings survived the audit and all six are confirmed on `main`, one of them worse than stated.
None is pinned by a test. Because the gate forbids skips and pytest reports an expected failure
as a skip, a pinning test can only land green: its fix lands with it.

## 2. Goals and non-goals

Goals, in execution order:

1. No test depends on an environmental artifact: the five loop tests stop skipping without
   `generated/`.
2. Each surviving audit finding is pinned by a test named for the claim it protects and lands
   with the smallest internal fix that makes it pass.
3. A property test asserts the paper's core validity claim, replay ≡ live, on-policy, across
   seeds and both shipped policies.
4. Every paper-silent choice the fixes make gets a `GAPS.md` §3 row; the docs say what the code
   does.

Non-goals, recorded for later tracks:

- Resumable iterations (owner decision: refuse loudly instead; §4.1).
- The trace-pool tamper window during `offline()` and the discovery agent's view of sibling
  attempt directories (`LiveQuestion._run_attempt`, `see/live.py:192-193`, runs the agent with
  the whole tree as cwd) — both are questions about what the paper intends → track D.
- Prompts as a proper seam for non-editable installs → track E (track A spec §10).
- The argparse-description one-liner repeated in four entry points → track G.
- Real-agent validation (owner decision: toy loop and demo only; track F buys paper numbers).

Constraints carried from track A: nothing public in `see/policy/api.py` or
`see/policy/observation_signal.py` is renamed; no workdir directory is renamed; `generated/`
is never committed; the gate stays strict (no skips, no xfail); commit messages are plain
imperative with no agent-attribution trailers; no push without the owner's go-ahead.

## 3. Test infrastructure

### 3.1 Prompt injection fixture

`see/prompts.py` reads the two paper prompts through a module constant `GENERATED`
(`see/prompts.py:13-18`, `_read`); `exploration_prompt` substitutes six `$`-style placeholders
(`node_dir`, `history_dir`, `baseline_dir`, `eval_program`, `problem_file`,
`direction_guidance`) and `policy_improvement_prompt` replaces three brace placeholders
(`{method_file}`, `{history_dir}`, `{trace_pool}`). `tests/test_loop.py:12-18` skips the whole
module when `generated/policy_improvement_prompt.md` is absent.

The suite gets its first `tests/conftest.py` with one fixture, `stub_prompts`: it writes
`exploration_prompt.md` (each `$` placeholder once) and `policy_improvement_prompt.md` (each
brace placeholder once) into `tmp_path` and monkeypatches `see.prompts.GENERATED` to that
directory. `_read` stays real. The scripted toy agents never read prompt text, so no assertion is
lost. The `pytestmark` skip in `tests/test_loop.py` is deleted and its module-scoped loop
fixture requests `stub_prompts`. After this, nothing in `tests/` reads `generated/`; the real
prompts remain covered by `tools/generated.sha256` and `extract_listings.py --check`.

### 3.2 Conventions

- Test names state the claim they protect (`test_interrupted_iteration_is_refused_not_merged`).
- Every test builds its world from `see.toy`, `see.synthetic` and `tmp_path`; none reads a file
  in the repository.
- The property test (§5) is one function looping over seeds internally, so the suite's count
  does not move with the seed list.
- The exact test count is pinned only in `.github/workflows/ci.yml`. `.claude/CLAUDE.md` and
  `reconstruction/README.md` say "the whole suite" and point at the workflow.

## 4. The six fixes

Each subsection names the defect as verified, the test(s), the fix, and the ledger row.

### 4.1 Interrupted iteration (audit HIGH)

Defect. `DreamRSI.online` (`see/loop.py:129-188`) refuses to start only when
`trace_pool/iterNNNN` exists (`:131-133`), which is written last (`:176`); `runs/iterNNNN` is
created first with `exist_ok=True` (`:138-141`, and per attempt at `see/live.py:177-178`). The
`try/except Exception` around `policy.solve()` (`:159-163`) does not catch `KeyboardInterrupt`.
A killed iteration therefore leaves a populated `runs/iterNNNN/tree` and no `trace_pool` entry;
a restart silently overwrites in-scope attempts, leaves out-of-scope attempt directories on disk
unreferenced by `trace.json`, and exposes them to later iterations through the `history/iterNNNN`
symlink (`:142-145`). `state.json`'s counter never advances (`:120-127`). CLAUDE.md promises the
opposite: refuse, delete the partial directory, never merge.

Tests.
- `test_interrupted_iteration_is_refused_not_merged`: pre-seed `runs/iter0001/tree` with one
  attempt inside and one outside the restart grid; `online(1)` raises `RuntimeError` naming
  `runs/iter0001`, before any agent call; `state.json` and the seeded files are unchanged.
- `test_interrupt_leaves_a_partial_run_that_a_restart_refuses`: a scripted discovery agent that
  raises a `BaseException` subclass on its second call (a stand-in for `KeyboardInterrupt`,
  which pytest handles specially; it takes the same uncaught path through `except Exception`);
  the interrupt propagates out of `online(1)`; `runs/iter0001` exists, `trace_pool/iter0001`
  does not, `state.json` is unchanged; a second `online(1)` raises the refusal.

Fix. The guard checks `runs/iterNNNN` as well as `trace_pool/iterNNNN`; the message names the
existing directory and says to delete it, not merge into it. No completion markers, no resume,
no change to the symlink; exception handling is unchanged.

Ledger. §3 row: the paper is silent on crash handling; the reconstruction refuses to reuse a
partial iteration. CLAUDE.md's existing bullet becomes true as written.

### 4.2 Replay fallback symmetry (audit MED)

Defect. `validate_plan` (`see/objective.py:72-81`) returns `None` for a rejected plan and its
docstring says "None means use the fallback grid"; the live path does so
(`see/loop.py:136-137`, `LoopConfig.fallback_grid`). Replay (`run_episode`,
`see/objective.py:116-161`) passes `None` through to `ReplayQuestion`
(`see/world.py:329-350`), which then grants the policy the trace's full support. A candidate
whose plan is rejected is scored on a wider tree than it would ever see live.

Test. `test_rejected_plan_replays_on_the_fallback_grid_not_the_whole_trace`: a policy whose
`plan_grid` returns an invalid plan, replayed via `run_episode` on a `synthetic_trace` wider than
the context's fallback; no probe lies outside the fallback grid, and the episode's objective
values equal those of the same policy handed the fallback as an explicit plan.

Fix. `run_episode` applies `context.fallback_branch_count` / `context.fallback_refine_count`
(fields that already exist, `see/policy/api.py:66-67`) clipped to trace support, exactly as an
explicit plan is clipped. The live path builds its context from `LoopConfig.fallback_grid`, so
the two agree by construction. This changes replay scores for rejected-plan candidates and can
change which offline version is deployed for runs that hit this path; that is the intent.

Ledger. §3 row: rejected or absent plans replay on the fallback grid, mirroring the live path.

### 4.3 Malformed evaluator result (audit MED)

Defect. `LiveQuestion` already contains two per-cell failures: an agent that raises becomes a
`no_program` cell (`see/live.py:192-197`) and an evaluator that raises becomes an
`evaluator_crashed` cell (`:242-249`). An evaluator that returns without raising but not a
mapping with a numeric `combined_score` fails while the observation is built, inside
`pool.map` (`:152-164`), and the whole batch is lost.

Test. `test_one_malformed_evaluator_result_fails_one_cell_not_the_batch`: a `TaskSpec` whose
`evaluate` returns garbage for exactly one of three cells; `probe_batch` returns three
observations, two with real scores and one `evaluator_crashed` cell whose error names the
malformation.

Fix. `_evaluate` validates the return value inside the existing per-cell guard and reuses
`evaluator_crashed` with an error text such as `malformed evaluator result: missing
combined_score`. No new failure class; `see/policy/observation_signal.py` is untouched.

Ledger. §3 row: the paper is silent on evaluator misbehaviour; one cell's evaluator failure never
discards its siblings.

### 4.4 Deploy integrity (audit MED)

Defect. `offline()` (`see/loop.py:190-214`) copies the argmax candidate from
`policy_dev/history/` to `deployed/iterNNNN.py` (`:207`) without verifying that the bytes copied
are the bytes `_sweep` (`:235-285`) scored; `_archive` (`:216-233`) records round, index, score
and validity but no digest. The module's own guarantee ("the deployed policy never scores below
π_t on H_t") is a claim about bytes and is currently unverifiable.

Tests.
- `test_deployed_policy_digest_matches_the_scored_candidate`: after a normal `offline(t)`, the
  archive record, the sweep summary and `state.json` all carry the same sha256, and it equals
  the digest of `deployed/iterNNNN.py`.
- `test_tampered_candidate_is_not_deployed`: run `offline(t)` once normally, rewrite the
  archived winner's `method.py`, then call the deploy step again with the original scored
  record; it raises and `deployed/` is unchanged.

Fix. `_archive` computes the sha256 of each archived `method.py` and stores it in its record;
the sweep summary (`beta_sweep.json`, `see/__main__.py:30-33`) carries it. The copy at the end
of `offline()` moves into a method `_deploy(t, record)` that recomputes the digest of the file
it is about to copy, raises on mismatch, copies, and writes the deployed digest to `state.json`
beside the score; `offline()` calls it with the argmax record. Frozen directories are not made
read-only (it would collide with the delete-the-partial-directory recovery).

Ledger. §3 row: the deployed file's identity is pinned by digest. §7 open item: `trace.json`
under `trace_pool/` is equally unpinned while `_sweep` replays against it.

### 4.5 Budget (audit MED)

Defect, as re-verified. `Question.probe_batch` (`see/world.py:251-302`) already refuses any
cell outside the planned grid for both backends, so calls are bounded at runtime by the plan.
The hole is the plan bound: `LoopConfig.hard_max_grid` defaults to `(32, 30)`
(`see/loop.py:43-60`), which admits 32 × 31 = 992 calls; the paper's settings are 10 × 11 = 110
(the default, encoded by `fallback_grid = (10, 10)`) and 32 × 20 = 640. No `GAPS.md` row
explains the 992.

Test. `test_default_hard_caps_admit_no_more_calls_than_the_paper`: with the default context, no
plan accepted by `validate_plan` has `W × (R + 1) > 640`; `(32, 19)` is accepted, `(33, 19)`
and `(32, 20)` are rejected.

Fix. `hard_max_grid` defaults to `(32, 19)`; the `--hard-max` defaults in `see/__main__.py:88`
and `scripts/run_dream_rsi.py:37` follow. No product cap is added: Listing 2 passes two
independent per-dimension caps to the policy, and a product bound would diverge from the paper's
own validation rule.

Ledger. §3 row: hard caps are the paper's largest reported setting; the per-dimension form is
the paper's, the values are ours.

## 5. Replay ≡ live property test

`tests/test_replay_equivalence.py::test_replay_reproduces_live_episodes_across_seeds_and_policies`.
For each seed in a fixed list and each of `see/policies/parallel_refine.py` and
`see/policies/portfolio.py`: build the toy task in `tmp_path`, run a live episode through
`LiveQuestion` with `ScriptedDiscoveryAgent(seed)`, freeze the tree exactly as `online()` does,
replay the same policy with the same configuration on the frozen `Trace` through
`ReplayQuestion`, and assert the two episodes are identical round by round (batch cells,
revealed scores, failure classes, parent deltas), then `budget_spent`, `decision_rounds`,
`best_so_far`, `eq1_value` and `beta_sweep`. The claim is on-policy; off-policy truncation stays
covered by the existing tests in `tests/test_objective.py`. The seed list is sized so the whole
suite stays under about five seconds, and a failure names the seed, the policy and the first
differing round. The existing single-seed `test_on_policy_replay_reproduces_the_live_episode`
remains.

## 6. Ledger and docs

- `reconstruction/GAPS.md` §3: five new rows (§4.1–4.5). §7: two open items (trace-pool tamper
  window; sibling-attempt visibility).
- `.claude/CLAUDE.md`: the interruption bullet stays (now true); Commands block drops the
  literal count and points at `ci.yml`; Architecture gains one clause: the deployed policy's
  sha256 is recorded in `state.json` and verified before the copy.
- `reconstruction/README.md`: count line and the status table's outer-loop row updated.
- `.github/workflows/ci.yml`: `--expect` moves from 47 to the new total (eight new tests as
  designed → 55; the plan fixes the exact number).

## 7. Verification

1. The gate at the new count: ruff, format, pyright, `--check`, pytest with zero skips,
   `check_junit.py --expect N`.
2. `python -m see demo --workdir /tmp/drsi` completes.
3. Manual interrupted-run check on the toy loop: start the demo, interrupt it during iteration 1,
   restart with the same workdir → the refusal names `runs/iter0001`; delete it → the restart
   proceeds.
4. The property test's runtime is measured and recorded in the plan; the suite stays under about
   five seconds.

## 8. Sequencing

Eight tasks, one commit each, on `spec/test-hardening`, in the order of §2: fixture; interrupted
iteration; replay fallback; malformed evaluator; deploy integrity; budget; property test;
ledger and docs (with the workflow count). Each task is a red-then-green pair except the first
and last. Execution follows track A's process: subagent-driven development with a task review
per task, a final whole-branch review, one fix wave, and a pull request only on the owner's
go-ahead.

## 9. Deferred items surfaced by this design

- Resumable iterations need per-attempt completion markers and a rule for attempts outside the
  new grid; the analysis is in the context map (workflow `wf_4e4d9eaa-66a`) → a later track.
- `beta_sweep.json` and `state.json` gain a digest key; a schema for the two files would let
  track D pin them → track D.
- The prompt stub in `tests/conftest.py` is the minimal seam; track E's `PromptSet` replaces it
  when prompts get a path override.
