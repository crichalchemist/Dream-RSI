# Provenance and gap ledger

What the paper (`papers/Dream-RSI.pdf`, 2026-09-14 build) pins down, what public
dependencies supply, what this reconstruction had to choose, where the paper
contradicts itself, and what was measured here. Section and line references are
to the paper and its listings.

## 1. Specified exactly by the paper

| Item | Source | Where it lives here |
|---|---|---|
| Exploration prompt, 28 lines | Listing 1 | `generated/exploration_prompt.md` (regenerated from the PDF) |
| Policy-improvement prompt, 273 lines | Listing 2 | `generated/policy_improvement_prompt.md` |
| Discovered Lasso solver, 847 lines | Listing 3 | `generated/lasso_path_dream_rsi.py` |
| Tree structure, legal set A(T) = {r} ∪ leaves, batch size ≤ W | Sec. 3 | `see/world.py` |
| Replay reveal rule, termination (empty batch, K2, tree exhausted) | Sec. 3 | `see/world.py` |
| Replay score, Eq. (1) | Sec. 3 | `see/objective.py: eq1_value` |
| Selection π_{t+1} = argmax over M versions, π_t included | Sec. 3 | `see/loop.py: offline` |
| Policy API names, `Observation`/`CellMeta` fields, success semantics, batch legality | Listing 2 | `see/policy/api.py`, `see/world.py` |
| Parallel penalty: effective_sequential_rounds / total_probes, ceil(k/W) per batch | Listing 2 l.17-21 | `see/objective.py` |
| `plan_grid` contract, `GridPlan(branch_count, refine_count, reason)`, cap checks | Listing 2 l.190-242 | `see/policy/api.py`, `see/objective.py: validate_plan` |
| Initial policy π_1 = parallel refine | Sec. 4 | `see/policies/parallel_refine.py` |
| Budgets: 10 workers × 11 steps = 110 calls/round (3.1 Pro), 32 × 20 = 640 (3.7 Flash); 5 rounds (Lasso), 10 (math) | Sec. 4, 4.1, 4.2 | `LoopConfig` defaults |
| Task definitions | App. A | via SimpleTES evaluators |

Recovery method for the listings: glyph coordinates, not text extraction. Line
numbers come out contiguous (28/273/847). Indentation and alignment come from the
monospace grid. Soft wraps are rejoined, and all 70 wraps in Listing 3 fall on
token boundaries. Typographic glyphs are mapped back to ASCII.
Details: `tools/extract_listings.py`.

## 2. Supplied by public dependencies, not by the paper

- **SimpleTES** (github.com/wq-will/SimpleTES, AGPL-3.0). The paper's Lasso task *is*
  SimpleTES's benchmark: the same 17 synthetic instances, the same `CPP_CODE`/`COMPILE_FLAGS`
  wire format and the same evaluator. SimpleTES's `generate_results.py` loads exactly the six
  held-out datasets of Fig. 3(a). SimpleTES also ships evaluators for the three math tasks
  (sum-difference, circle packing n=26, first autocorrelation). The `n_valid`/`n_total`
  fields of Listing 2's `Observation` match this evaluator's output, so Dream-RSI very
  likely wraps SimpleTES evaluators. SimpleTES's vendored Eigen lacks `Eigen/Core` (its
  `.gitignore` drops it), so compilation needs system Eigen.
- **KernelBench** supplies the four kernel tasks. The paper names "VGG16, LayerNorm, ConvDiv,
  ConvMax" but not their problem ids or levels, and the last two names match several
  KernelBench problems. The GPU model is not stated either.
- **Gemini CLI** is the discovery agent. Its invocation flags, tools, timeouts and sampling
  settings are not given.

## 3. Reconstruction choices (the paper is silent)

| Item | What the paper gives | Chosen here |
|---|---|---|
| M (versions per offline phase), K1, K2 | symbols only | `LoopConfig.versions=4`, K1/K2 unset |
| β1, β2 in Eq. (1) | "fixed coefficients ≥ 0" | 0, 0 (set per task; score-scale dependent) |
| Which objective selects π_{t+1} | Eq. (1) in Sec. 3, `pareto.reward` in Listing 2 | Listing 2 by default, `objective="eq1"` switch |
| "Per-trace attainment" | named only | (best − baseline) / (trace ceiling − baseline), clipped to [0, 1] |
| `pareto.auc` | "rewards high attainment with few total probes" | area under the (work fraction, attainment) Pareto frontier of the sweep, linear from (0,0), flat to work = 1 |
| Beta grid, λ | "a fixed beta grid", λ unnamed | 0.0, 0.1, …, 1.0; λ = 0.1 |
| Failure taxonomy | only `"ok"` and `"compile_other"` named | ok, correctness, resource, code, shape, compile_other, timeout, no_program (repairable), env (hard); regex classifier in `observation_signal.py` |
| Helper semantics (`branch_promising`, …) | names only | `see/policy/observation_signal.py` |
| `GridPlanningContext` field names | `history`, `hard_max_*`, `trace_*`, "fallback/hard caps and worker cap" | adds `fallback_branch_count`, `fallback_refine_count`, `max_parallelism` |
| `live_cycle_manifest.json` schema, `_current` | file names only | `see/loop.py: online`; `_current` links the newest completed cycle |
| Workspace snapshot | "resumes the parent's saved workspace" | the parent's program file is copied into the child's `attempt_*` dir |
| History given to the discovery agent | "access to H_{t−1}" | `runs/iterNNNN/history/` links to every earlier tree |
| Direction provider / `$direction_guidance` | one line in Listing 2, a placeholder in Listing 1 | pluggable callable, default empty. Sec. 5.1 reports that injected guidance *hurt*, so the main runs' content is unknown |
| Concurrent evaluation | not discussed | serialised by default; parallel timing measurements of Lasso or kernels would interfere |
| Policy-development agent | "a fixed LLM-based agent"; model not named | any CLI agent; revises the latest version |
| Sandboxing LLM-written policy code | not discussed | replay sweep runs in a subprocess with a timeout; crashes score −∞ |
| Interrupted iteration handling | not discussed | `online(t)` refuses when `runs/iterNNNN`, `trace_pool/iterNNNN` or the iteration's first archive `policy_dev/history/r{round+1}_tNN_m0` already exists, naming every one that does in a single message (delete them, never merge into them); the archive is checked because the round counter is persisted only on success, so a restart after a crash in `offline()` would otherwise recreate and overwrite the aborted iteration's archives (see the archive-name guard row) |
| Rejected/absent grid plan in replay | Listing 2's fallback/hard caps (for the live runner); nothing about replay under a rejected plan | `run_episode` clips a rejected or absent plan to `context.fallback_branch_count`/`fallback_refine_count`, mirroring `online()`'s live substitution of `LoopConfig.fallback_grid`; neither path clips the fallback grid to the hard caps, and replay clips it only to the trace's own grid, flagging an oversize fallback `out_of_support=True` exactly as an explicit plan would be |
| Malformed (non-raising) evaluator result | not discussed | one cell's malformed result — not a `Mapping`, a `combined_score` that isn't a `numbers.Real`, or an `error` that is neither absent, `None`, nor a `str` — is recorded as an unevaluated `evaluator_crashed` cell for that cell only; sibling cells in the same batch keep their real results; using `numbers.Real` lets NaN, ±inf and numpy scalar scores pass unchanged, but now fails a cell whose score is `np.bool_` or a 0-d array (no shipped evaluator produces either) |
| Deploy-time code integrity | not discussed | the argmax candidate's sha256, computed when it was archived, before scoring, is re-verified against the file about to be copied to `deployed/iterNNNN.py`; a mismatch raises before the offline log is persisted, and since `run()` advances `state["iteration"]` only after `offline()` returns, a restart recomputes the same t; the run aborts with `runs/iterNNNN` and `trace_pool/iterNNNN` already written by the completed `online()` — the interrupted-iteration guard then refuses either, so recovery deletes both, and the aborted iteration's `policy_dev/history/r*_tNN_m*` directories, before restarting (the round counter is persisted only on success, so a restart would otherwise recreate and overwrite those archive names), then re-runs the whole iteration (a fresh online rollout and fresh offline candidates, not just a re-sweep); on success `state.json` records the deployed digest beside the score, absent until the first `offline()` completes |
| `hard_max_grid` default | Listing 2 names `hard_max_branch_count`/`hard_max_refine_count` (the two per-dimension caps) but gives no values | lowered to `(32, 19)`, so `branch_count × (refine_count + 1) ≤ 640`, the paper's largest reported per-round budget; the two caps stay independent per-dimension bounds, matching Listing 2's own validation rule rather than adding an unspecified product cap |
| Interrupt semantics (SIGINT, SIGTERM mid-iteration) | not discussed | the batch in flight is cancelled: every running agent's process group is killed (a batch is at most `max_parallelism` wide, so every attempt in it is running; a job not yet dequeued by a worker is cancelled), and killed attempts are neither evaluated nor recorded (no new fail class); what was collected before the interrupt is frozen as `runs/iterNNNN/partial/{trace.json,live_episode.jsonl,live_cycle_manifest.json}` with the manifest's `error` naming the interrupt and `"partial": true`; `trace_pool/` is never written on that path, so replay scoring never sees a truncated tree; SIGTERM is mapped to `KeyboardInterrupt` by the CLI entry points (`install_signal_handlers()`), SIGKILL runs none of this; an in-process evaluator cannot be interrupted, so the return waits for evaluations already running |
| Agent and sweep child-process lifetime | not discussed | `CommandAgent` runs each agent CLI in its own session (`start_new_session=True`) and on timeout SIGTERMs the whole group, then SIGKILLs it after `kill_grace` (5 s); the result records `timed_out` and `eval/score.json` records `agent_timed_out`; the killed group's pipes are closed, not drained (a descendant outside the group could hold them open); an exception escaping the call in the calling thread (Ctrl-C during the policy-agent call) kills the group before propagating; `terminate()` claims only the calls still running, so a call whose process had already finished keeps its result; a timed-out agent's attempt is scored as whatever program it left (its parent's copy if it wrote nothing) — whether that should count as `no_program` is open (C2); `_sweep` starts `see sweep` in its own session and kills the group on timeout or when an interrupt escapes; a failure to spawn the sweep subprocess (a host error such as `EMFILE`) propagates instead of scoring the version invalid, because it describes the host, not the version; POSIX only (`os.killpg`); a child that calls `setsid` itself leaves the group and is not killed |
| Archive-name guard on restart | not discussed | `archive_name(round, t, m)` is the single source of the `rNNNN_tNN_mM` names; `online(t)` refuses when `policy_dev/history/<archive_name(round+1, t, 0)>` exists because `_archive` increments the round in memory and it is persisted only by `_deploy` and at the end of `offline()`; recovery deletes that iteration's `r*_tNN_m*` directories together with `runs/iterNNNN` and `trace_pool/iterNNNN` |

## 4. Places where the paper contradicts itself

1. **Action space.** In Sec. 3, C ⊆ A(T) = {r} ∪ leaves, and selecting r opens the
   earliest-created unrevealed branch. So the formal model opens at most one branch per
   round. The paper's own baseline opens 10 (or 32) workspaces in its first round, which
   that model cannot express. Listing 2 exposes one legal root cell per branch
   ("several roots per batch"), and so does this reconstruction.
2. **Objective.** Eq. (1) (best score − β1·N + β2·N/rounds, one episode per world) and
   Listing 2's `pareto.auc − λ·parallel_penalty` (a beta sweep, normalised attainment) have
   different forms and units. The paper never says which one selected the reported policies.
   Listing 2 also mentions "legacy AUC-only sweep[s]", which implies the objective changed
   during the experiments.
3. **Version count.** Sec. 3 builds π^{m+1} for m = 0..M−1, which gives M+1 versions, yet
   selects over m ∈ {0..M−1}. Implemented: M versions in total, π^0 included.
4. **"Independent" workspaces.** Sec. 4 describes the baseline's workspaces as
   independent. Listing 1 makes every discovery call read *every* proposal in the current
   tree and in all history, so branches share context.
5. **The dagger row in Fig. 3(a).** "SimpleTES †" is never defined. On all five datasets
   checked, SimpleTES's released best program runs within 25% of that row here, and
   1.4–2.1× slower than the plain "SimpleTES" row (§5 below). The plain row is probably
   quoted from the SimpleTES paper, i.e. measured on other hardware.
6. **Listing 3's origin.** The paper does not say which backbone found it. On all five
   datasets checked, the measured solver/sklearn speed ratios are closer to the
   Gemini-3.1-Pro row than to the Flash row (§5).

## 5. Measured here

Machine: 4-core Intel Xeon @ 2.8 GHz with AVX-512. Compiled with g++ 13.3 and Eigen 3.4.0,
using the SimpleTES flags plus the listing's own `-fopenmp -ffast-math`. Single-threaded,
as both SimpleTES scripts pin OMP to 1.

**Search score** (SimpleTES evaluator, 17 instances, fresh random instances per run, 2 runs):

| Program | valid | geo-mean ms | vs glmnet port |
|---|---|---|---|
| Listing 3 (Dream-RSI) | 17/17, 17/17 | 19.25, 18.24 | 1.78×, 1.92× |
| SimpleTES released best | 17/17, 17/17 | 24.93, 25.92 | 1.37×, 1.35× |
| glmnet port (seed) | 17/17, 17/17 | 34.25, 35.06 | 1.00× |

The transcribed solver's maximum objective gap is 2.1e-11 (tolerance 1e-6). A transcription
error that still compiled would almost certainly show up as a correctness failure here.

**Held-out datasets** (SimpleTES `benchmark()`, min of 5; `scripts/verify_lasso.py --downstream --gisette`):

| Dataset | sklearn ms | SimpleTES best ms | Listing 3 ms | SimpleTES/Listing 3 | paper, plain row (Pro, Flash) | paper, † row (Pro, Flash) |
|---|---|---|---|---|---|---|
| Gisette 5100×5000 | 6470.8 | 6598.7 | 1731.8 | (3.81, both invalid*) | 1.11, 2.88 | 3.05, 7.92 |
| DNA 1700×180 | 61.3 | 33.3 | 42.8 | 0.78 | 0.32, 0.51 | 0.75, 1.20 |
| Leukemia 38×7129 | 147.6 | 27.4 | 23.8 | 1.15 | 0.51, 0.74 | 0.93, 1.34 |
| Colon 62×2000 | 157.3 | 16.2 | 11.6 | 1.40 | 0.71, 0.95 | 1.19, 1.60 |
| Duke 44×7129 | 264.6 | 32.0 | 26.2 | 1.22 | 0.56, 0.77 | 0.96, 1.32 |

\* On Gisette neither solver meets the 1e-6 objective tolerance (gaps 1.1e-6 and 1.6e-6).
The paper reports downstream times without saying whether those solutions passed.
sklearn/Listing 3 ratios measured here (3.7, 1.4, 6.2, 13.6, 10.1) sit closer to the paper's
Gemini-3.1-Pro row (4.0, 1.9, 7.5, 14.0, 11.5) than to its Flash row
(10.3, 3.0, 10.8, 18.8, 15.8). RCV1 was not run: densified, it is several GB.

**Evaluation noise.** Re-evaluating an unchanged program moves the Lasso score by about
2% (0.0288–0.0294). Replay stores one sample per node, so the replay simulator inherits
that noise without any estimate of it.

## 6. Reading the reported results

- **Fig. 3(a) averages milliseconds arithmetically**, so RCV1 (82–95% of every row's
  sum) decides the "Avg." column. Per dataset, Dream-RSI (3.1 Pro) is *slower* than its
  own controlled baseline on 5 of 6 held-out sets. It is faster only on RCV1. Under a
  geometric mean (the aggregation the paper uses for its search score) the headline
  "1.22× faster" becomes 0.89×:

  | Row | arithmetic ms | geometric ms |
  |---|---|---|
  | Fixed exploration, 3.1 Pro | 3587.1 | 159.1 |
  | Dream-RSI, 3.1 Pro | 2931.0 | 179.4 |
  | Fixed exploration, 3.7 Flash | 2516.7 | 127.6 |
  | Dream-RSI, 3.7 Flash | 2350.6 | 117.9 |
  | SimpleTES (plain row) | 3804.8 | 121.3 |

  The efficiency claim (317 vs 550 calls; 1879 vs 3200) is unaffected.
- **Table 1.** On autocorrelation (lower is better) Dream-RSI scores 1.456375, *worse* than
  its controlled baseline (1.456001), AlphaEvolve (1.455700) and SimpleTES (1.453675).
  Circle packing ties its baseline and the best prior systems at 2.635983. Sum-difference
  improves by 0.0014.
- **All results are single runs**, with no seeds, variance or confidence intervals.
  The 162× call reduction against SimpleTES also compares gpt-oss-120b with Gemini 3.1
  Pro, so the model confounds it. Only the comparison with fixed exploration is
  controlled.
- **Replay is off-policy evaluation without correction.** A recorded tree only contains
  what the behaviour policy chose to run. A candidate that would go wider or deeper is
  truncated to the recorded support ("cannot earn replay reward"), so the estimator
  favours policies close to the ones that generated the pool. The paper does not
  discuss this.

## 7. Still needed to reproduce the paper's numbers

- Gemini 3.1 Pro / 3.7 Flash through Gemini CLI with the paper's (unpublished) settings.
  Swapping in another agent gives a different experiment.
- Lasso, 5 rounds: 550 discovery calls on Pro (up to 3200 on Flash), plus M policy-development
  calls per round and one ~20 s evaluation per attempt.
- The unknowns in §3, above all which objective selected the deployed policies, plus
  λ, β, M and the direction provider. These determine what "Dream-RSI" means operationally.
- KernelBench problem ids and a GPU for the kernel results.
- Whether the trace-pool tamper window during `offline()` should close the same way
  `policy_dev/history/` now does: `_archive`'s sha256 (§3 above) covers only the archived
  `method.py`, not `trace_pool/iterNNNN/trace.json`, and `_sweep` (`see/loop.py:259-310`)
  replay-scores every version directly against the whole pool with no digest on any frozen tree;
  the sweep subprocess writes its own `sha256` of the method file into `beta_sweep.json`
  (`see/__main__.py:17,34`), but `_archive`/`_deploy` never compare it to the archive digest, so
  that the subprocess scored the archived bytes is inferred from ordering, not verified.
- Whether the discovery agent's filesystem view is meant to extend past the shared-proposal
  reading §4 item 4 already documents: `LiveQuestion._run_attempt` (`see/live.py:176`) runs the
  agent with `cwd=self.tree_dir` (`:194`), so it can read every sibling `attempt_*` directory's
  actual program and eval output — not just the proposals Listing 1 names. On an agent crash
  (`:195-198`), only the crashed attempt's own program file is removed, so other files that
  the crashed agent wrote stay visible to later sibling attempts.
