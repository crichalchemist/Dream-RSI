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
| Tree structure, batch size ≤ W | Sec. 3 | `see/world.py` |
| Replay reveal rule, termination on K2 or an exhausted tree | Sec. 3 | `see/world.py` |
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
  `.gitignore` drops it), so compilation needs system Eigen or a host Eigen root passed as
  `--eigen-include` (§3, "Lasso host toolchain").
- **KernelBench** supplies the four kernel tasks. The paper names "VGG16, LayerNorm, ConvDiv,
  ConvMax" but not their problem ids or levels, and the last two names match several
  KernelBench problems. The GPU model is not stated either.
- **Gemini CLI** is the discovery agent. Its invocation flags, tools, timeouts and sampling
  settings are not given. `scripts/run_dream_rsi.py` records each launch's agent argv and CLI
  version in `<workdir>/launches.jsonl`. The D2a run used the Antigravity CLI instead, because
  the Gemini CLI refused this account's login (§5, "D2a run").

## 3. Reconstruction choices (the paper is silent)

| Item | What the paper gives | Chosen here | How to change |
|---|---|---|---|
| M (versions per offline phase), K1, K2 | symbols only | `LoopConfig.versions=4`, K1/K2 unset | `LoopConfig.versions` (`run_dream_rsi.py --versions`, `see demo --versions`); `LoopConfig.max_live_rounds` (no flag); `LoopConfig.max_replay_rounds` (`see sweep --max-rounds`) |
| β1, β2 in Eq. (1) | "fixed coefficients ≥ 0" | 0, 0 (set per task; score-scale dependent); Eq. (1)'s max runs over the revealed subtree including the root, so a policy that reveals only worse cells scores the root (pinned by `test_ledger.py::test_eq1_scores_the_root_when_nothing_revealed_beats_it`) | `LoopConfig.beta1`/`beta2` (no runner flag); `see sweep --beta1/--beta2` |
| Which objective selects π_{t+1} | Eq. (1) in Sec. 3, `pareto.reward` in Listing 2 | Listing 2 by default; `objective="eq1"` selects Eq. (1); any other name is refused when `LoopConfig` is built and by `score_of`, so a typo cannot silently select Eq. (1) | `LoopConfig.objective`, one of `see.objective.OBJECTIVES`; `run_dream_rsi.py --objective` |
| "Per-trace attainment" | named only | (best − baseline) / (trace ceiling − baseline), clipped to [0, 1]; on a trace whose ceiling does not beat the root, 1.0 for every episode, best or none, because nothing recorded beats the root (pinned in `test_objective.py::test_attainment_normalises_between_root_and_ceiling`) | fixed in `see/objective.py` (`attainment`) |
| `pareto.auc` | "rewards high attainment with few total probes" | area under the (work fraction, attainment) Pareto frontier of the sweep, linear from (0,0), flat to work = 1 | fixed in `see/objective.py` (`pareto_auc`) |
| Beta grid, λ | "a fixed beta grid", λ unnamed | 0.0, 0.1, …, 1.0; λ = 0.1 | `LoopConfig.betas`/`lam` (no runner flag); `see sweep --betas/--lam` |
| Failure taxonomy | only `"ok"` and `"compile_other"` named | ok, correctness, resource, code, shape, compile_other, timeout, no_program (repairable), env (hard); regex classifier in `observation_signal.py` | fixed in `see/policy/observation_signal.py` (frozen surface) |
| Helper semantics (`branch_promising`, …) | names only | `see/policy/observation_signal.py` | fixed in `see/policy/observation_signal.py` (frozen surface) |
| `GridPlanningContext` field names | `history`, `hard_max_*`, `trace_*`, "fallback/hard caps and worker cap" | adds `fallback_branch_count`, `fallback_refine_count`, `max_parallelism` | fixed in `see/policy/api.py` (frozen surface) |
| `live_cycle_manifest.json` schema, `_current` | file names only | `see/loop.py: online`; `_current` links the newest completed cycle | fixed in `see/loop.py` (`_manifest`, `online`) |
| Workspace snapshot | "resumes the parent's saved workspace" | the parent's program file is copied into the child's `attempt_*` dir | fixed in `see/live.py` (`_resume_from`) |
| History given to the discovery agent | "access to H_{t−1}" | `runs/iterNNNN/history/` links to every earlier tree | fixed in `see/loop.py` (`online`) |
| Direction provider / `$direction_guidance` | one line in Listing 2, a placeholder in Listing 1 | pluggable callable, default empty. Sec. 5.1 reports that injected guidance *hurt*, so the main runs' content is unknown | `DreamRSI(directions=...)`; no flag |
| Concurrent evaluation | not discussed | serialised by default; parallel timing measurements of Lasso or kernels would interfere | `LoopConfig.serialize_eval` (no flag) |
| Policy-development agent | "a fixed LLM-based agent"; model not named | any CLI agent; revises the latest version | `DreamRSI(policy_agent=...)`; `run_dream_rsi.py --policy-agent` (a preset name or a JSON argv) |
| Sandboxing LLM-written policy code | not discussed | replay sweep runs in a subprocess with a timeout; crashes score −∞ | `LoopConfig.sweep_timeout`/`kill_grace` (no flag); the mechanism is fixed in `see/loop.py` (`_sweep`) |
| Interrupted iteration handling | not discussed | `online(t)` refuses when `runs/iterNNNN`, `trace_pool/iterNNNN` or any archive of the iteration `policy_dev/history/r*_tNN_m*` already exists, naming every one that does in a single message (delete them, never merge into them); the archives are checked because the round counter is persisted only on success, so a restart after a crash in `offline()` would otherwise recreate and overwrite the aborted iteration's archives, starting with `r{round+1}_tNN_m0` (see the archive-name guard row); `run_dream_rsi.py` runs the same check before it appends to `launches.jsonl`, so a refused restart records no launch and the report's caps still come from the launch that ran (pinned in `test_run_dream_rsi.py::test_a_refused_restart_records_no_launch`) | fixed in `see/loop.py` (`check_iteration`, which `online` calls first) |
| Rejected/absent grid plan in replay | Listing 2's fallback/hard caps (for the live runner); nothing about replay under a rejected plan | `run_episode` clips a rejected or absent plan to `context.fallback_branch_count`/`fallback_refine_count`, mirroring `online()`'s live substitution of `LoopConfig.fallback_grid`; neither path clips the fallback grid to the hard caps, and replay clips it only to the trace's own grid, flagging an oversize fallback `out_of_support=True` exactly as an explicit plan would be | the rule is fixed in `see/objective.py` (`run_episode`); the grid is `LoopConfig.fallback_grid` (`run_dream_rsi.py --grid`, `see sweep --fallback`) |
| Malformed (non-raising) evaluator result | not discussed | one cell's malformed result — not a `Mapping`, a `combined_score` that isn't a `numbers.Real`, or an `error` that is neither absent, `None`, nor a `str` — is recorded as an unevaluated `evaluator_crashed` cell for that cell only; sibling cells in the same batch keep their real results; using `numbers.Real` lets NaN, ±inf and numpy scalar scores pass unchanged, but now fails a cell whose score is `np.bool_` or a 0-d array (no shipped evaluator produces either) | fixed in `see/live.py` (`_evaluate`) |
| Deploy-time code integrity | not discussed | the argmax candidate's sha256, computed when it was archived, before scoring, is re-verified against the file about to be copied to `deployed/iterNNNN.py`; a mismatch raises before the offline log is persisted, and since `offline()` advances `state["iteration"]` only in its final write, after `_deploy`, a restart recomputes the same t; the run aborts with `runs/iterNNNN`, `trace_pool/iterNNNN` and the iteration's archives `policy_dev/history/r*_tNN_m*` already written — the interrupted-iteration guard then refuses, naming `runs/iterNNNN`, `trace_pool/iterNNNN` and every `r*_tNN_m*` archive of the iteration in one message, so recovery deletes exactly what was named before restarting (the round counter is persisted only on success, so a restart would otherwise recreate and overwrite those archive names), then re-runs the whole iteration (a fresh online rollout and fresh offline candidates, not just a re-sweep); on success `state.json` records the deployed digest beside the score in the single write `offline()` makes at its end, together with the offline log and the iteration counter (`online()` writes nothing, so a crash anywhere in `offline()` leaves the file as the previous iteration left it; pinned in `test_loop.py::test_state_is_written_once_per_iteration_after_the_deploy`), absent until the first `offline()` completes; from then on `online(t)` re-hashes `state['deployed']` against that digest before loading it and refuses a file that changed since it was deployed (the hash and the load are two reads of the file, so a change between them is not caught); `_archive` refuses a sweep report whose `sha256` (hashed by the subprocess before it loaded the method, again a separate read) differs from the archive digest, so that the sweep scored the archived bytes is verified rather than inferred from ordering | fixed in `see/loop.py` (`_archive`, `_deploy`, `check_iteration`) |
| `hard_max_grid` default | Listing 2 names `hard_max_branch_count`/`hard_max_refine_count` (the two per-dimension caps) but gives no values | lowered to `(32, 19)`, so `branch_count × (refine_count + 1) ≤ 640`, the paper's largest reported per-round budget; the two caps stay independent per-dimension bounds, matching Listing 2's own validation rule rather than adding an unspecified product cap | `LoopConfig.hard_max_grid`; `run_dream_rsi.py --hard-max`; `see sweep --hard-max` |
| Interrupt semantics (SIGINT, SIGTERM, SIGHUP mid-iteration) | not discussed | the batch in flight is cancelled: every running agent's process group is killed (a child that has called `setsid` has left the group and survives; a batch is at most `max_parallelism` wide, so every attempt in it is running; a job not yet dequeued by a worker is cancelled), and killed attempts are neither evaluated nor recorded (no new fail class); what was collected before the interrupt is frozen as `runs/iterNNNN/partial/{trace.json,live_episode.jsonl,live_cycle_manifest.json}` with the manifest's `error` naming the interrupt and `"partial": true`; `trace_pool/` is never written on that path, so replay scoring never sees a truncated tree; SIGTERM and SIGHUP are mapped to `KeyboardInterrupt` by the CLI entry points (`install_signal_handlers()`; a SIGHUP inherited ignored, as under nohup, stays ignored); SIGKILL and any unmapped signal run none of this, and because the agents and the sweep run in their own sessions they do not receive the terminal's hangup themselves; an interrupt before the first attempt (during planning or the baseline evaluation) leaves nothing under `runs/`, because the baseline is evaluated before `runs/iterNNNN/` is created — the few statements between creating the run directory and the first attempt can leave an empty `tree/`, which the restart guard refuses like any other leftover; an in-process evaluator cannot be interrupted, so the return waits for evaluations already running; with `eval_repeats` k > 1, an attempt already queued on the evaluation lock runs all k evaluations, so the wait can be up to k runs per queued attempt | fixed in `see/loop.py` (`install_signal_handlers`, `online`) |
| Agent and sweep child-process lifetime | not discussed | `CommandAgent` runs each agent CLI in its own session (`start_new_session=True`) and on timeout SIGTERMs the whole group, then SIGKILLs it after `kill_grace` (5 s); the result records `timed_out` and `eval/score.json` records `agent_timed_out`; the killed group's pipes are closed, not drained (a descendant outside the group could hold them open); an exception escaping the call in the calling thread (Ctrl-C during the policy-agent call) kills the group before propagating; `terminate()` claims only the calls still running and gives every claimed group one shared `kill_grace`, so a call whose process had already finished keeps its result and the cost of an interrupt does not grow with the batch width; once the CLI has exited, the rest of its group is SIGKILLed without grace: within a second if a member still holds the pipes (the call then returns the CLI's real result instead of waiting for EOF), and after every normal completion, so nothing an attempt started outlives its call; if the pipes are still held one slice after that sweep, the holder is a process outside the group, which is not killed by design: the call returns the CLI's real exit status within about two seconds with the held output dropped, and the reaped leader's group id is never signalled again; a claimed call stops waiting for its pipes within a second even when a descendant outside the group still holds them (the agent's remaining output is dropped, as on the timeout path); on Darwin a group whose only members are zombies answers EPERM, which is treated like an empty group; a worker fault (an exception `_run_attempt` raises before or after the agent call, such as a host error) abandons the batch the way an interrupt does, cancelling its queued attempts and killing its running agents, but through `kill_running()`, which leaves the agent open for `offline()` and the next iteration; the fault ends the episode: every later `probe_batch` on that `LiveQuestion` fails at once, so a policy that swallows the fault and keeps probing gets nothing further, and if its `solve` then returns normally the manifest still records `batch abandoned: <fault>` and the cycle is frozen with that error (pinned in `test_loop.py::test_a_fault_the_policy_swallows_still_reaches_the_manifest`); an agent spawned in the microseconds after that kill runs to its timeout unless interrupted, and the fault does not propagate until it returns (untested, as are the pre-spawn check that narrows the window and queued-attempt cancellation on the fault path); an interrupt that lands while the batch's attempts are being submitted or its workers joined kills the agents before propagating (pinned by interrupts injected at the executor); a batch's cells reach the tree in one statement, so a partial freeze never holds half a batch (a whole batch without its episode row remains possible when the interrupt lands between `_execute` returning and the episode row being appended); agent and sweep output is decoded with replacement, so a process killed between the bytes of one character cannot raise out of the call; a timed-out agent's attempt is scored as whatever program it left but recorded as a `timeout` failure with the agent's own message as its error (the evaluator's error wins when there is one), so it is never a success and never raises the trace ceiling; its `valid` flag is the evaluator's `validity` when one is reported, else false; one that changed nothing is an untouched attempt (see that row), `no_program` and never evaluated (pinned in `test_command_agent.py::test_a_timed_out_agent_that_edited_its_program_is_still_scored_as_a_timeout`, `::test_a_timed_out_agent_that_left_a_broken_program_keeps_the_evaluators_verdict` and `::test_agent_timeout_kills_the_whole_process_group`); `_sweep` starts `see sweep` in its own session and kills the group on timeout or when an interrupt escapes; a failure to spawn the sweep subprocess (a host error such as `EMFILE`) propagates instead of scoring the version invalid, because it describes the host, not the version; POSIX only (`os.killpg`); a child that calls `setsid` itself leaves the group and is not killed | `CommandAgent(timeout=, kill_grace=)`; `run_dream_rsi.py --agent-timeout`; the mechanism is fixed in `see/live.py` |
| Archive-name guard on restart | not discussed | `archive_name(round, t, m)` is the single source of the `rNNNN_tNN_mM` names; `online(t)` refuses when any `policy_dev/history/r*_tNN_m*` of iteration t exists, because an iteration runs once per workdir and `_archive` increments the round in memory, persisted only at the end of `offline()`, so a restart would recreate `<archive_name(round+1, t, 0)>` first; recovery deletes the named `r*_tNN_m*` directories together with `runs/iterNNNN` and `trace_pool/iterNNNN` | fixed in `see/loop.py` (`archive_name`, `check_iteration`, which `online` calls first) |
| Per-round call budget | Sec. 4: 10 workers × 11 attempts = 110 calls (Pro), 32 × 20 = 640 (Flash) | the fallback grid's size and the hard caps' product; no product cap and no budget is enforced, so a plan may be (32, 0) or (1, 19), and the D2a run left it so: no plan pushed on a budget there (the largest, the deployed version's live 4 x 4, was 20 calls under the run's 6 x 4 caps of 30), and the caps already bound a round at W × (R + 1); a complete rerun (D2c) revisits it; the demo hardcodes (4, 5) and (8, 8) (pinned in `test_ledger.py::test_the_default_fallback_grid_is_the_papers_110_call_round` and `test_objective.py::test_default_hard_caps_admit_no_more_calls_than_the_paper`) | `LoopConfig.fallback_grid`/`hard_max_grid`; `run_dream_rsi.py --grid/--hard-max`; `see sweep --fallback/--hard-max` |
| Legal set A(T) | Sec. 3: {r} ∪ leaves, one branch opened per round | Listing 2's: one root cell per unopened branch, so several branches may open in one batch (§4 item 1; pinned in `test_world.py::test_initial_legality_is_roots_only_in_creation_order`) | fixed in `see/world.py` (`legal_roots`) |
| `probe_batch([])` | Sec. 3 lists the empty batch among the terminations | illegal (`IllegalBatch`: "stop by not probing"); in replay it scores the version −∞ like any other exception, live it propagates to the policy; the D2a run left it so, as no version and no live batch there left a batch empty, and D2c revisits it (pinned in `test_world.py::test_illegal_batches_are_refused`, `test_objective.py::test_crashing_or_illegal_policy_is_scored_minus_infinity`) | fixed in `see/world.py` (`probe_batch`) |
| Two objectives, two episode sets | Eq. (1) per episode; Listing 2 sweeps beta | `pareto.reward` is computed from the beta-grid episodes (one per trace per beta) and `eq1.V` from one extra episode per trace at the policy's baked-in default beta, the one a live rollout uses; an error in either set invalidates both (pinned in `test_ledger.py::test_pareto_comes_from_the_beta_grid_and_eq1_from_the_default_beta_episode` and `::test_an_error_in_either_episode_set_invalidates_both_objectives`) | betas and λ as above; the split is fixed in `see/objective.py` (`beta_sweep`) |
| Trace ceiling | not named | the best successful cell's score, else the baseline; failed and timed-out cells keep their scores in the trace but never raise it; an out-of-support plan is clipped to the trace and scored as the clipped plan, with `out_of_support` reported for information; the D2a run left it so, as none of its episodes was clipped (every version planned 4 x 3, the tree's own grid, in replay: m0 its fixed grid, m1 and m2 their bootstrap for an empty history, which a one-trace pool's replay always has; see the live-plan signal row), and D2c revisits it (pinned in `test_ledger.py::test_the_trace_ceiling_counts_only_successful_cells` and `::test_an_out_of_support_plan_is_flagged_and_scored_as_its_clipped_grid`) | fixed in `see/world.py` (`Trace.ceiling`, `ReplayQuestion`) |
| The floor and π₁ | Sec. 4: π₁ is parallel refine | `policy_dev/history/baseline/` is re-swept on the current pool every `offline()` for reference and is never a candidate; `LoopConfig.initial_policy` is copied to `deployed/iter0001.py` only when the workdir is created, so a resumed run keeps the policy it started with (pinned in `test_ledger.py::test_the_floor_is_reswept_for_reference_and_never_deployed` and `::test_the_initial_policy_seeds_only_a_fresh_workdir`) | `LoopConfig.initial_policy` (no flag); the floor is fixed in `see/loop.py` (`offline`) |
| Lasso host toolchain | not discussed; SimpleTES's evaluator compiles with a hardcoded `g++ -O3 -march=native -std=c++17`, with `-I<task dir>/eigen` when that directory exists, else `-I/usr/include/eigen3` | the adapter does not copy SimpleTES's vendored `eigen/` (it lacks `Eigen/Core`); `eigen_include` makes `task/src/eigen` a symlink to a host Eigen root, the directory the evaluator tries first, refusing a root without `Eigen/` and an existing link that points elsewhere (a restart reuses its own link); the compiler is whatever `g++` resolves to on `PATH`: on macOS Apple's `g++` is clang without OpenMP, so MacPorts gcc is selected with `port select` (pinned in `test_tasks.py::test_eigen_include_is_where_the_evaluator_looks_first` and `::test_a_restart_reuses_its_eigen_link_and_refuses_a_different_one`) | `simpletes_task(eigen_include=)`; `run_dream_rsi.py --eigen-include`; `verify_lasso.py --eigen-include` (search score only); the compiler: `PATH` |
| Untouched attempt | not discussed; Listing 1's agent resumes the parent's saved workspace | the sha256 of the program copied into the attempt's directory is taken before the agent runs (the copy, not the source: agents run in the whole tree, so another agent may rewrite the source meanwhile); an attempt whose program still matches it when the agent returns, timed out or not, is `no_program` with the error "agent left its program unchanged (…)" (`score.json` records `untouched`, true or false, for every attempt): it is not evaluated, scores 0.0, is never a success and never raises the trace ceiling; its file stays, so its children resume the same bytes; a program the check cannot read is left to the evaluator, as before. In the D2a run, 1 of 16 attempts in iteration 1 and all 4 in iteration 2's trace were untouched, and the latter scored `ok` at up to +12.7% on evaluation noise alone (§5, "D2a run"; pinned in `test_loop.py::test_an_agent_that_changes_nothing_is_no_program_and_never_evaluated`, `::test_a_source_edited_mid_attempt_does_not_hide_an_untouched_program` and `::test_a_program_the_check_cannot_read_is_left_to_the_evaluator`) | fixed in `see/live.py` (`_run_attempt`) |
| Agent exit code | not discussed | recorded as `agent_returncode` in `score.json`, beside `agent_stderr`, the stderr tail `CommandAgent` returns (at most 4000 characters; stdout, the agent's reply, can quote the program and is not kept); never a failure class: the D2a run's best cell exited 3 on a quota error after editing its program, so failing on the exit code would have discarded it, while the untouched rule already fails the attempts that changed nothing (pinned in `test_loop.py::test_score_json_keeps_the_agents_stderr_tail`) | fixed in `see/live.py` (`_run_attempt`) |
| Evaluation repeats | not discussed | one evaluation per program by default; with k > 1 (k odd) each program is evaluated k times and the median run is kept, with every run's score in `repeat_scores`, and the first run that reports an error ends the repeats and is kept as it is, so a failure is never averaged away; the baseline is scored the same way, but once per workdir (`baseline_eval.json` is cached, so a restart with another k keeps the first baseline); an attempt's k runs hold the evaluation lock together. In the D2a run, eight evaluations of the unchanged baseline spread 14.0%, about the size of the run's improvement (§5, "D2a run"; pinned in `test_loop.py::test_repeats_score_the_median_run_and_stop_at_the_first_error`, `::test_an_even_zero_or_negative_repeat_count_is_refused_when_the_config_is_built` and `::test_the_baseline_is_scored_with_the_same_repeats_as_the_attempts`) | `LoopConfig.eval_repeats`; `run_dream_rsi.py --eval-repeats` |
| Live-plan signal in replay | Listing 2 lets `plan_grid` read the replay support fields `trace_branch_count`/`trace_refine_count`, and says a plan beyond them "cannot earn replay reward" | after each replay episode is scored and its metrics are read, `run_episode` calls `plan_grid` again with both trace fields cleared and validates the plan as `online()` does (a rejected or absent plan becomes the fallback grid); the episode records that grid as `live_plan`, `beyond_support` when it is wider or deeper than the recorded tree, and `live_plan_error` if the call raised; that grid is the plan the policy makes for the trace's own cycle without the trace fields, given the history that cycle had (`see/pool.py:context_factory` passes only the manifests of earlier cycles), not the plan it would run next live, which `online()` makes with every manifest in the pool (see the next live plan row): the signal flags a policy whose plan reads the trace fields and misses one whose plan changes only once later history is present; informational only, it never changes the episode's score, error or validity, even for a policy that kept the question from `solve` and probes it from `plan_grid`; `beta_sweep.json` carries `beyond_support` beside `out_of_support`; it means something only under live caps, as the loop's sweep and `see sweep` pass them, and is always false under `default_context`, whose caps are the trace's own grid. In the D2a run no version replayed on a clipped episode (see the trace ceiling row), and on D2a's pool the signal reads 0 although the deployed version m2 planned 4 x 4 live over 4 x 3 trees: replayed with `see sweep --fallback 4 3 --hard-max 6 4`, every m2 episode has `live_plan` 4 x 3 and `beyond_support` false, because a one-trace pool's replay has no history and m2 plans 4 x 3 without history whether the trace fields are set or cleared (with iteration 1's manifest in history it plans 4 x 3 with them set and 4 x 4 with them cleared); the next live plan row measures that per version (§5, "D2a run"; pinned in `test_ledger.py::test_a_policy_that_clamps_to_the_trace_fields_is_flagged_with_its_reward_unchanged`, `::test_a_live_plan_that_raises_is_recorded_not_an_episode_error` and `::test_a_policy_that_probes_its_kept_question_from_plan_grid_cannot_change_its_own_score`) | fixed in `see/objective.py` (`run_episode`, `beta_sweep`) |
| Next live plan | not discussed; replay scores a version only on the recorded trees, and nothing looks at the plan it would run live before it is deployed | `see sweep` records `next_live_plan` in `beta_sweep.json` for each version: the grid `online()` would run in the next iteration with that version deployed, planned once, after the sweep is scored, by a fresh instance at its baked-in default beta (`policy_cls(None)`), with every manifest in the pool as history and no trace fields (`see/pool.py:next_context`, the context `DreamRSI._context(self.manifests())` builds), and validated as `online()` does (a rejected or absent plan becomes the fallback grid, with `fallback` true); its `max_parallelism` is `--max-parallelism`, which the loop's sweep passes from `LoopConfig.max_parallelism`, else the newest recorded tree's; `beyond_support` is true when no single recorded tree is at least as wide and as deep as the plan (two trees that cover its width and its depth only between them do not cover it); a `plan_grid` that raises, on which `online()` would stop, is recorded as `error`; report only: it changes no score, validity or selection. `report_run.py` shows it per version and counts the versions whose next live plan no recorded tree covers, and how many of them were deployed; a deployed version's is the grid the next live cycle runs. On D2a's pool, `see sweep --fallback 4 3 --hard-max 6 4` gives m2 a next live plan of 4 x 4, beyond the one 4 x 3 tree, while none of its 12 episodes is beyond support and its reward is still 0.532062; D2a's own sweeps predate the field, so its report reads "not measured" (pinned in `test_ledger.py::test_d2as_deployed_version_plans_a_next_cycle_its_one_tree_does_not_cover`, `::test_a_plan_that_widens_with_history_is_flagged_only_as_the_next_live_plan`, `::test_a_next_live_plan_is_covered_only_by_one_tree_at_least_as_wide_and_deep`, `::test_the_next_live_plan_is_made_at_the_default_beta_and_falls_back_as_online_does`, `::test_a_next_live_plan_that_raises_is_recorded_and_the_sweep_scores_as_without_it`, `::test_the_next_live_plan_uses_the_given_parallelism_else_the_newest_trees`, `test_loop.py::test_the_next_context_is_the_one_online_plans_with`, `::test_the_deployed_versions_next_live_plan_is_the_grid_online_runs_next` and `::test_the_loop_sweeps_with_its_own_parallelism`) | `see sweep --max-parallelism`; fixed in `see/objective.py` (`next_live_plan`) and `see/__main__.py` (`cmd_sweep`) |

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

**D2a run (2026-09-25).** One real-agent run on the Lasso task on a second machine, Intel(R)
Core(TM) i7-7700K CPU @ 4.20GHz (8 threads), g++ (MacPorts gcc13 13.4.0_1+stdlib_flag) 13.4.0,
Eigen 3.4.1, SimpleTES 47d3413da1d8. The Antigravity CLI (`agy`) 1.2.11 with
`gemini-3.1-pro-high` drove discovery, and Claude CLI 2.1.283 drove policy development. Both
were isolated from the owner's global CLI configuration. The paper's Gemini CLI refused this
account's login ("no longer supported for Gemini Code Assist for individuals"). The run took 1
launch, with a 4 x 3 fallback grid (16 calls), 6 x 4 hard caps (30 calls) and three policy
versions per iteration. It completed 1 of 2 iterations.

The account's individual Gemini 3.1 Pro quota ran out during iteration 1's last round
(agy logged HTTP 429 from 18:02). Two of that round's four agents exited with code 3: b1a3 after
editing its program (the iteration's best cell) and b2a3 before writing one.

Iteration 2, under the deployed version, was stopped during its second round:
- All four first-round agents exited 3 on the quota.
- Before the second round, concurrent `agy` calls sharing one home had corrupted the token file
  at the hourly refresh. The second round's agents could not authenticate and exited 1.
- No agent in iteration 2 changed its program.
- Its frozen trace holds 4 probes. The second round's 4 attempts were evaluated on disk but
  never returned to the policy.

A restart's pre-flight was refused by the same quota. The workdir alone does not show these
causes: the loop then kept each agent's exit code but not its stderr (D2b adds its tail to
`score.json`), so they come from agy's own logs, which stay outside the workdir.

The full report and the evidence subset are in `evidence/d2a-lasso/`, and the stopped
iteration's are in `evidence/d2a-lasso/restart1/`. The workdir archive stays off the repository
(sha256 `281b49ad4c5c41129e3bb1cd1fe64625fdb1ec99c3baccffa39ea8ea5db3cb2c`).

- Per-round calls:
  - iteration 1 (the initial policy) planned 4 x 3 without the fallback and spent 16 probes in
    4 rounds;
  - the stopped iteration 2 (the deployed version) planned 4 x 4 (20 calls). Its frozen trace
    holds 4 probes, and 8 attempts were evaluated on disk.

  The caps here are 16 and 30; the paper's are 110 and 640.
- Out-of-support replay: none of the 3 versions replayed on clipped episodes (12 episodes each).
  For the deployed version this holds by construction, not by behaviour.
  - Replay hands `plan_grid` the recorded tree's grid (`trace_branch_count`,
    `trace_refine_count`, set in `see/pool.py`); a live context leaves both None.
  - Every replay episode of version m2 planned 4 x 3 because a one-trace pool's replay has no
    history, and without history m2 plans 4 x 3 whether those fields are set or cleared.
  - Live, in iteration 2, the same version planned 4 x 4, deeper than any recorded tree.

  Measuring out-of-support plans needs a signal that depends neither on those fields nor on
  replay's history, which stops before the replayed trace. The next live plan (§3) is that
  signal: re-swept over this pool, m2's is 4 x 4 and flagged, with its reward unchanged.
- Untouched programs:
  - 1 of 16 attempts left its resume source byte for byte (iteration 1, b2a3). Its agent exited
    3 on the quota before writing a program, so the evaluator re-ran its parent's program
    (`compile_other`, 0 against 0). This agent did not choose to leave the program unchanged, and
    iteration 1 has no case where one did.
  - In the stopped iteration, all 8 attempts evaluated on disk did (b0a0–b3a1, all `ok`). Every
    agent exited non-zero, 3 on the quota and then 1 on auth, and each program is the baseline's.
    The report counts the 4 in the frozen trace.
- Empty batches: none among the versions, and none among the live iterations.
- Health of iteration 1:
  - 16 attempts and 11 successes; fail classes ok 11, timeout 3, compile_other 2;
  - 2 agent timeouts, 2 agents that exited non-zero, 0 attempts without a program,
    0 evaluator crashes;
  - baseline 0.0134266, best 0.0167488.

  Two of the 11 successes scored 0, with 16 and 14 of 17 instances valid (b0a0, b0a2).
  SimpleTES reports validity 1.0 and no error for them, so the loop classes them `ok`.

  Version rewards were m0 0.475, m1 0.4529 and m2 0.532062; m2 was deployed.
- Evaluation noise in the run: the stopped iteration's eight evaluations of the unchanged
  baseline scored 0.0132689–0.0151303, a 14.0% spread. That runs from 1.2% below to 12.7% above
  the baseline's own 0.0134266 at run start. Iteration 1's best is 24.7% above that baseline
  score, but only 10.7% above the highest of those re-evaluations.
- Smoke test on this host (`host.json`, two runs each), against the Xeon's 2% above:

  | Program | Valid | Geo-mean ms | Spread |
  |---|---|---|---|
  | Listing 3 | 17/17, 17/17 | 63.03, 62.36 | 1.1% |
  | glmnet port | 17/17, 17/17 | 74.24, 70.50 | 5.3% |
  | SimpleTES best | 17/17, 17/17 | 62.88, 66.39 | 5.6% |

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
  discuss this. Listing 2 also lets a policy read the recorded grid (`trace_*`), in replay
  only, so a policy can clamp its plan there and plan wider live, and replay then scores a
  plan the policy would not run. Each replay episode records the plan made with those fields
  cleared for that reason; it flags a policy whose plan reads them, but not one whose plan
  changes only with the later history `online()` passes (§3, "Live-plan signal in replay").
  Each version's sweep therefore also records its next live plan, the grid `online()` would run
  with every recorded cycle in history, and flags it when no recorded tree covers it. That flag
  is reported only: replay still scores the version on the recorded support (§3, "Next live
  plan").

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
  `method.py`, not `trace_pool/iterNNNN/trace.json`, and `_sweep` (`see/loop.py`)
  replay-scores every version directly against the whole pool with no digest on any frozen tree
  (the method-file side is closed: `_archive` compares the `sha256` the sweep subprocess hashed
  before loading the method, `see/__main__.py:17,34`, to the archive digest).
- Whether the discovery agent's filesystem view is meant to extend past the shared-proposal
  reading §4 item 4 already documents: `LiveQuestion._run_attempt` (`see/live.py`) runs the
  agent with `cwd=self.tree_dir`, so it can read every sibling `attempt_*` directory's actual
  program and eval output — not just the proposals Listing 1 names. On an agent crash, only the
  crashed attempt's own program file is removed, so other files that the crashed agent wrote stay
  visible to later sibling attempts.
