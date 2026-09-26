# D2's verdicts and the next run's instrumentation (track D2b)

Date: 2026-09-25. Branch: `spec/verdicts-instrumentation-d2b`, from `main` at 2d33b52 (track D2a
merged as PR #7). Owner decisions are recorded inline as **Decision**.

## 1. Context

The D1 spec (`2026-09-25-fidelity-ledger-d1-design.md`, section 9) deferred four behaviour changes
to D2 and gated them on one real-agent run:
- a per-round call budget;
- scoring an out-of-support replay as zero rather than as the clipped plan;
- treating an attempt whose program is byte-identical to its resume source as `no_program`;
- whether `probe_batch([])` should end the episode.

Track D2a (`2026-09-25-real-agent-run-d2a-design.md`) made that run on the Lasso task and
committed its evidence under `reconstruction/evidence/d2a-lasso/`. Its section 8 set the rule D2b
applies: for each item, "observed: change it" or "not observed: leave it, and say why", grounded in
the counts.

The run finished 1 of 2 iterations. The account's individual Gemini 3.1 Pro quota ran out in
iteration 1's last round, and iteration 2 failed on quota and then on a token-file race (GAPS
section 5, "D2a run"). D2a handed D2b six more items:
- from its spec's section 12:
  - a quota;
  - each agent call's stderr;
  - redaction at copy time;
- from GAPS section 5: an out-of-support signal that does not depend on the replay-only `trace_*`
  fields;
- from its final review:
  - the launch record written before the restart guard;
  - a quieter host.

**Decision (scope).** D2b is code only. It covers the verdicts that D2a's counts support and the
instrumentation the next run needs. **D2c** is a complete two-iteration rerun on the owner's
`dream-rsi` Google Cloud project with paid quota, and it revisits the verdicts that one run could
overturn. The GCP auth and quota check is a separate no-spend spike, not part of this spec.

**Decision (noise).** D2a's eight re-evaluations of the unchanged baseline spread 14.0%, about the
size of the improvement. D2b adds a repeat setting whose default of 1 keeps the paper's single
evaluation.

**Decision (approach).** The behaviour changes live in the loop, and repeats are a `TaskSpec`
wrapper applied from `LoopConfig`. Two alternatives were rejected:
- writing repeats separately into both evaluation sites, which leaves two code paths to keep in
  step;
- computing the answers in the report only, which would leave noise in the trace as `ok`
  improvements.

## 2. Goals, non-goals, constraints

Goals:

1. Each of D1's four deferred changes has a verdict grounded in D2a's counts, and the ledger
   records it.
2. An attempt whose agent changed nothing is no longer scored as a fresh evaluation of an
   unchanged program.
3. The next run can answer what D2a could not:
   - why an agent call failed;
   - whether a policy would plan beyond the recorded tree live;
   - how noisy its evaluations are.
4. The evidence copy redacts what D2a's evidence commit had to redact by hand, and a refused
   restart leaves no launch record.

Non-goals:
- a real-agent run or any Google Cloud configuration (D2c);
- a per-round budget setting;
- scoring an out-of-support replay as zero;
- a failure class for non-zero agent exits;
- regenerating D2a's committed reports, which stay a record of that run.

Constraints carried from tracks A to D2a:
- The gate stays strict:
  - `ruff format --check .` and `ruff check .`;
  - `pyright` basic with 0 errors;
  - `python tools/extract_listings.py --check`;
  - `python -m pytest -q` with zero skips;
  - `tools/check_junit.py --expect N`.
- No `xfail`, `skip`, `# noqa` or `# type: ignore`.
- Nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes.
- No workdir directory is renamed.
- The agent callable's signature is unchanged.
- Every paper-silent choice gets a GAPS section 3 row with its "How to change" cell and a
  hand-computed pin.
- One commit per task, with plain imperative messages and no attribution trailers.
- SimpleTES is AGPL and stays unvendored.

## 3. The verdicts

| Item | Verdict | Evidence |
|---|---|---|
| Untouched program | **Change** (section 4.1) | Observed. 1 of 16 attempts in iteration 1 (b2a3). All 4 attempts in iteration 2's trace (8 on disk). Iteration 2's four unedited copies of the baseline scored 0.0143934 to 0.0151303 against 0.0134266 and were classed `ok`: evaluation noise recorded as improvement. |
| Non-zero agent exit | **Record only**, with the stderr tail (section 4.3) | Every untouched case had exited non-zero, but so had b1a3, the run's best cell (0.0167488, `agent_returncode` 3). It edited its program before the quota stopped it. Failing on exit code would discard it; the untouched rule removes the harm without doing so. |
| Evaluation noise | **Repeat setting**, default 1 (section 4.2) | Owner decision (section 1). The best score is 24.7% over the run-start baseline but only 10.7% over the highest re-evaluation of that same baseline. |
| Per-round call budget | **Leave; no setting** | Not observed. The largest plan was the deployed version's live 4 x 4, 20 calls, under the 6 x 4 cap of 30. The hard caps already bound a round's spend at W x (R + 1). D2c sets them for its quota. |
| Out-of-support replay, zero versus clip | **Leave the clip; add the live-plan signal** (section 5) | Not observable as posed. Listing 2 tells a policy it may read the replay support fields, and versions m1 and m2 clamp their plans to them. So 0 of 36 episodes were clipped, by construction. Listing 2's "cannot earn replay reward" stays read as truncation (GAPS section 6). |
| `probe_batch([])` | **Leave** (scored minus infinity) | Not observed: no version and no live batch in either report. |

Every "not observed" rests on 16 live attempts and one live plan. D2c revisits the budget, zero
versus clip and the empty batch with a complete run.

## 4. Loop changes

### 4.1 The untouched check (`see/live.py:LiveQuestion._run_attempt`)

After the resume source is copied into the attempt's directory, the sha256 of the copied file is
taken. The copy is what the agent starts from. Hashing it, rather than re-reading the source
afterwards, means a concurrent agent that edits the source file mid-attempt cannot change the
comparison: agents run with the whole tree as their working directory.

After the agent returns, and the batch was not cancelled, the attempt's program is checked. If it
exists and matches the digest, the attempt takes the existing `no_program` path:
- it is not evaluated;
- its score is 0.0;
- it is a failure, and it never raises the trace ceiling;
- `score.json` gains `"untouched": true`;
- the error reads `agent left its program unchanged (<first 200 characters of the stderr tail>)`.

This also covers a timed-out agent that touched nothing, which settles D1's second clause ("skip
the evaluation of a timed-out attempt that touched nothing").

The program file stays on disk. Its bytes equal the source's, so `_resume_from` resolves exactly as
before. A missing program and a crashed agent's deleted program behave as today.

### 4.2 Evaluation repeats (`see/live.py`, `see/loop.py`, the runner)

`see.live.repeated(task: TaskSpec, k: int) -> TaskSpec` returns
`dataclasses.replace(task, evaluate=...)`. The new `evaluate`:
- runs the inner evaluator up to k times;
- returns the first result whose `error` is not None, with the scores of the runs so far,
  including the failing one, in `repeat_scores`, so a failure is never averaged away;
- otherwise, sorts the k results by `combined_score` and returns the middle one, with all k scores
  in run order in `repeat_scores`.

An evaluator that raises or returns a malformed result is handled by `LiveQuestion._evaluate`'s
existing guard, as today.

`LoopConfig.eval_repeats: int = 1` is refused in `__post_init__` unless it is an odd integer of at
least 1. With an odd k, the middle result is one real evaluation, whatever the task's direction.
`DreamRSI` replaces its task with `repeated(task, k)` when k > 1. Both evaluation sites then score
the same way:
- the baseline (`DreamRSI.baseline_score`);
- every attempt (`LiveQuestion._evaluate`).

All k runs of an attempt happen inside the existing evaluation lock.

The runner gains `--eval-repeats` (default 1), and the launch record's `config` gains
`eval_repeats`. The baseline is evaluated once per workdir and cached in `baseline_eval.json`, so a
restart with a different k keeps the first baseline. The GAPS row says so; it is not refused.

### 4.3 The stderr tail (`see/live.py:LiveQuestion._run_attempt`)

`score.json` gains `agent_stderr`: the stderr tail that `CommandAgent` already returns, bounded at
4000 characters. For a timed-out or crashed agent it is the existing message. stdout is not kept:
it is the agent's reply, which can quote the program (AGPL), and it is large. The tail may hold
account names or home paths, so section 6.2 redacts it at copy time.

## 5. The live-plan signal (`see/objective.py:run_episode`)

This runs only when the policy plans (`use_plan=True`), and only after the replay episode is
finished (after `solve`, or after its error). `run_episode`:
- calls `policy.plan_grid` again with
  `dataclasses.replace(context, trace_branch_count=None, trace_refine_count=None)`, the context
  `online()` gives it live;
- validates the result with `validate_plan`. A rejected or absent plan becomes the context's
  fallback grid, as `online()` runs it.

`Episode` (internal, not the policy API) gains three fields:
- `live_plan`: `{"branch_count", "refine_count", "fallback"}`, the grid the policy would run live;
- `beyond_support`: true when that grid is wider or deeper than the trace's recorded grid;
- `live_plan_error`: the traceback if the extra call raised, else None.

The extra call never changes the episode's score, error or validity. It runs after scoring, and an
exception is recorded, not raised. Each episode already gets its own policy instance.

The executions file carries the new fields through `dataclasses.asdict`. `beta_sweep.json`'s
version summary gains `"beyond_support": any(...)` beside `out_of_support`.

The signal means something only under live caps. The loop's sweep and `see sweep` pass
`--fallback` and `--hard-max`. Under `default_context`, whose hard caps equal the trace's grid, it
is always false; the GAPS row says so.

## 6. Runner, redaction and report

### 6.1 The launch record follows the guard

`online()`'s two refusals move into `DreamRSI.check_iteration(t)`, which `online()` calls first,
with the same messages:
- the iteration's `runs/`, `trace_pool/` or archive directories exist;
- the deployed policy's digest has changed.

`scripts/run_dream_rsi.py:main` runs in this order:
1. build the loop;
2. `check_iteration(loop.state["iteration"] + 1)`;
3. `record_launch`;
4. `run()`.

A refused restart raises the same error and appends nothing to `launches.jsonl`, so the report's
"caps come from the last launch line" stays true.

### 6.2 Redaction at copy time (`scripts/report_run.py --copy-evidence`)

Every copied text file gets four rewrites:
- a compiler source-quote line (a gutter line of the form `NN | code`) becomes
  `[source line withheld]`;
- the home-directory prefix becomes `~`;
- the system temp prefix (`tempfile.gettempdir()`, e.g. `/var/folders/…/T` on macOS) becomes
  `$TMPDIR`;
- an email address becomes `[address withheld]`.

In `.json` and `.jsonl` files the rules apply to string values, where a quoted line sits between
escaped newlines. A file is rewritten only when a rule matched, and a JSON file stays valid JSON.
Files quoting `CPP_CODE` are still withheld whole.

`report.md` states the count per rule (for example "3 source lines withheld in 2 files"). The same
rules apply to `report.md` and `report.json` themselves, which now quote stderr tails.

### 6.3 The report reads the new fields

- Section 2 (out-of-support) gains per-version columns:
  - the number of episodes whose live plan is beyond support;
  - the live grid asked for.
- Section 3 (untouched) shows the loop's own `untouched` flag beside the report's byte check, and
  flags any row where they disagree (a later agent may have edited the file). The untouched table
  gains the agent's last stderr line.
- The health section gains:
  - the `eval_repeats` in force, from the launch record;
  - with k > 1, each iteration's median in-attempt spread, (max − min) / median of
    `repeat_scores`.

On a workdir that predates a field, each of these reads "not measured". D2a's committed reports
are not regenerated.

## 7. Ledger and docs

**GAPS section 3, new rows.** Each has its "How to change" cell and a pin:
1. **Untouched attempt**: the digest at copy time, `no_program`, not evaluated. Fixed in
   `see/live.py`.
2. **Evaluation repeats**:
   - odd k with a default of 1;
   - the median run, and the first error stops;
   - the baseline is cached per workdir.

   How to change: `LoopConfig.eval_repeats`, `--eval-repeats`.
3. **Live-plan signal in replay**: informational, and meaningful only under live caps. Fixed in
   `see/objective.py`.
4. **Agent exit code**: recorded with its stderr tail, never a failure class. Fixed in
   `see/live.py`.

**Existing rows record their verdicts.** These are the four rows whose text defers to "D2":
- "Agent and sweep child-process lifetime": its clause "whether an untouched program should count
  as `no_program` is D2" now points to the new untouched row;
- "Per-round call budget": "a budget knob is D2" becomes not enforced after D2a; the hard caps
  bound spend;
- "`probe_batch([])`": "ending the episode instead is D2" becomes left as minus infinity, not
  observed;
- "Trace ceiling": "zero-versus-clip is D2" becomes the clip kept after D2a, not observed; D2c
  revisits it.

**GAPS section 6**, the off-policy bullet, gains a clause: a policy that reads `trace_*` can clamp
its plan in replay and plan wider live, so replay scores a plan the policy would not run.

**`.claude/CLAUDE.md`**: "The policy cannot tell them apart" is qualified. It holds for the
`Question`, while `GridPlanningContext`'s `trace_*` fields are set only in replay.

**README**: the Lasso section names `--eval-repeats`, and the status table stays in step.

**CI**: `--expect` is pinned to the measured count.

## 8. Tests

About 16 new tests, hand-computed on the toy task with scripted agents and named for the outcome
they protect:

- **Untouched.**
  - `test_an_agent_that_changes_nothing_is_no_program_and_never_evaluated`, with an evaluator call
    counter.
  - `test_a_timed_out_agent_that_touched_nothing_is_never_evaluated`.
  - `test_a_source_edited_mid_attempt_does_not_hide_an_untouched_program`.
  - `test_score_json_keeps_the_agents_stderr_tail`.
- **Repeats.**
  - `test_repeats_score_the_median_run_and_stop_at_the_first_error`.
  - `test_an_even_or_zero_repeat_count_is_refused_when_the_config_is_built`.
  - `test_the_baseline_is_scored_with_the_same_repeats_as_the_attempts`.
  - `test_the_launch_record_carries_eval_repeats`.
- **Live plan.**
  - `test_a_policy_that_clamps_in_replay_but_plans_wider_live_is_flagged_with_its_reward_unchanged`.
  - `test_a_live_plan_that_raises_is_recorded_not_an_episode_error`.
- **Guard.**
  - `test_a_refused_restart_records_no_launch`.
- **Redaction.**
  - `test_quoted_source_lines_inside_json_strings_are_withheld_and_the_json_stays_valid`.
  - `test_home_temp_and_address_are_rewritten_and_counted`.
- **Report.**
  - The new columns on a workdir that has the fields.
  - "not measured" on one that does not.
  - A flagged disagreement between the loop's flag and the byte check.

The suite has 123 tests today, so about 139 afterwards; the last task measures the count and pins
it.

## 9. Success criteria

1. The gate passes on the final tree with zero skips, and the CI pin equals the measured count.
2. Every new test fails on the tree before its task (stash the source file, run, pop).
3. The GAPS table is in step with the code, checked row by row by the task's reviewer:
   - the four new rows;
   - the four rows that deferred to D2;
   - the section 6 clause.

   No "is D2" remains in GAPS.
4. `python -m see demo` runs end to end at the default `eval_repeats` of 1.
5. D2a's committed evidence is unchanged.

## 10. Task order

1. The untouched check and the stderr tail (`see/live.py`), with their GAPS rows.
2. `repeated`, `LoopConfig.eval_repeats`, `--eval-repeats` and the launch record field, with their
   GAPS row.
3. The live-plan signal (`see/objective.py`, the sweep summary), with its GAPS row.
4. `DreamRSI.check_iteration` and the runner's reorder.
5. `report_run.py`: redaction at copy time and the new report columns.
6. The verdict clauses in existing rows, GAPS section 6, CLAUDE.md, README and the CI pin.

Process as before: this spec, then the writing-plans skill, then subagent-driven execution with a
task review each, a whole-branch review at the end, and the finishing menu.

## 11. Hand-forward to D2c

- **Quota and auth.** A no-spend probe of paid Gemini, through an API key or Vertex AI billed to
  the `dream-rsi` project, with both agy and the Gemini CLI. Nothing is enabled, billed or created
  in the project without the owner's go.
- **Host.** A CPU VM in that project, with the toolchain from D2a's section 3 on Linux (`g++` with
  OpenMP and `libeigen3-dev`).
- **Settings.** The choice of k for `--eval-repeats`, and caps sized to the quota.
- **Versions.** Pinned agent CLI versions; agy updated itself from 1.2.10 to 1.2.11 during D2a.
- **Verdicts.** The three "not observed" verdicts, re-read from D2c's report with the live-plan
  signal in place.

## 12. Open questions

None that block the plan.

## 13. Amendments while planning (2026-09-25)

The plan (`docs/superpowers/plans/2026-09-25-d2b-verdicts-and-instrumentation.md`) was generated
from a spike: every task was built and gated in a throwaway worktree, one commit per task, and the
plan's replace blocks rebuild each task's files byte for byte. The spike changed the design in
these places.

1. **Seven tasks, not six.** Section 6's report work is split in two:
   - Task 5, redaction at copy time;
   - Task 6, the report's new columns.

   A reviewer could accept one and reject the other. The ledger and docs task becomes Task 7.
2. **`score.json` records `untouched` on every attempt**, as true or false, not only when true.
   Without that the report could not tell "checked, and not untouched" from a workdir that
   predates the flag, which section 6.3 must show as "not measured".
3. **A program the untouched check cannot read is left to the evaluator.** An agent that replaces
   its program with a directory gets one failed cell, as before the check existed. Hashing it
   unguarded would instead raise a worker fault that abandons the whole batch. A new test pins
   this.
4. **Where edits land.**
   - Task 1 changes the "Agent and sweep child-process lifetime" row's clause "whether an
     untouched program should count as `no_program` is D2", together with the code that settles
     it. Section 7 had put it in the last task.
   - Task 3 adds the live plan to `test_objective.py`'s rejected-plan comparison.
   - Task 4 changes the "Interrupted iteration handling" and "Deploy-time code integrity" rows to
     name `check_iteration`.
   - Task 5 changes `.claude/CLAUDE.md`'s evidence line.
   - Task 7 makes every README change.
5. **Two tests are exempt from section 9's criterion 2.** They guard behaviour Task 1 must keep,
   so they pass before it as well:
   - `test_command_agent.py::test_a_timed_out_agent_that_edited_its_program_is_still_scored_as_a_timeout`;
   - `test_loop.py::test_a_program_the_check_cannot_read_is_left_to_the_evaluator`.
6. **Test names and count.** Section 8's list becomes the names in the plan, 19 new tests in all
   (123 → 142).
   - The timed-out untouched case is pinned by `test_agent_timeout_kills_the_whole_process_group`,
     updated in place, and by the timed-out root in
     `test_an_agent_that_changes_nothing_is_no_program_and_never_evaluated`.
   - The "not measured" readings are pinned on D2a's committed evidence, a real pre-D2b workdir
     (`test_a_workdir_from_before_d2b_reads_not_measured`).
7. **The CI pin moves once**, in Task 7 (123 → 142). Each earlier task's gate checks its own
   measured count.
