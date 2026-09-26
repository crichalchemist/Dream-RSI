# D2c prep: plan counts, instrumentation out of the agent's view, flakes

Date: 2026-09-26. Branch `spec/d2c-prep` from `main` a63277f (PR #9 merged).

## 1. Goal

Three changes to make before the next paid real-agent run (D2c):

- **A. Plan counts.** A policy whose `plan_grid` returns numpy integer counts runs a whole live
  cycle and then crashes `online()` in `_freeze` (`TypeError: Object of type int64 is not JSON
  serializable`). The attempts have already run, which in a real run means paid agent calls. The
  crash also leaves a half-written `trace_pool/iterNNNN/`. This was verified on the toy loop
  during PR #9's review. The per-episode `plan` field can crash the sweep's
  `policy_execution_traces.jsonl` write in the same way.
- **B. Instrumentation out of the agent's view.** Listing 2 points the policy-development agent
  at earlier rounds' `proposal_results/beta_sweep.json` and
  `proposal_results/policy_execution_traces.jsonl`. D2b's live-plan fields and PR #9's
  `next_live_plan` are written into those files, so the agent reads them. The paper's agent
  never saw them. A field saying `beyond_support: true` would nudge the agent toward narrower
  plans, a soft selection gate applied through the prompt. It would also contaminate the count
  it reports.
- **C. Flakes.** One test fails whenever `FORCE_COLOR` is set. Two timing tests fail under host
  load.

## 2. Owner rulings (2026-09-26)

1. D2c keeps the next-live-plan flag **report only**; selection does not use it. Gating would cap
   the grid at iteration 1's size. The decision is revisited with D2c's data.
2. **Hide the live-plan family** from the policy-development agent:
   - the per-episode `live_plan`, `beyond_support` and `live_plan_error`;
   - the sweep-level `beyond_support`;
   - `next_live_plan`.

   `out_of_support` stays where it is. D2a's agent saw it, so D2c stays comparable with D2a, and
   it only restates Listing 2's rule that a plan beyond the trace earns no replay reward.
3. **Whole numbers are accepted.** Any whole-number count, whether a numpy integer or a float
   such as 4.0, is recorded as a plain int. A fractional or non-numeric count makes the plan
   invalid, so the fallback grid runs, as it does for an out-of-range count.
4. The sidecar approach (§4) was chosen over two alternatives:
   - a hidden full copy plus a filtered copy, rejected because the copies can drift;
   - editing the prompt, rejected because Listing 2 is kept verbatim.

## 3. Part A: plan counts

### 3.1 One rule, in `validate_plan`

`see/objective.py:validate_plan` normalises both counts before its range check:

- **Accepted.** An integer of any type (`operator.index` succeeds: `int`, numpy integers)
  becomes that `int`. A real number with an integral value (`numbers.Real` and
  `float(x).is_integer()`: `4.0`, `np.float64(4.0)`) becomes `int(x)`.
- **Rejected.** Anything else is rejected, and `validate_plan` returns `None`, which means the
  fallback grid, exactly as for an out-of-range count. That includes 4.5, `nan`, `inf`,
  strings and `None`.
- **Returned.** It returns the plan with the normalised counts (a `GridPlan` built with the same
  `reason`), so every caller sees plain ints.

The rule applies wherever `validate_plan` runs: `online()`, `run_episode`'s scoring path, and the
live-plan helper `_live_grid`. `_live_grid`'s own `operator.index` coercion (PR #9) is removed.
It is now redundant, and it disagreed with `online()`: it turned a 4.0 plan, which `online()`
runs, into a recorded error. Fallback counts come from the context and are already ints.

Consequences:

- A numpy-count policy runs live and freezes cleanly. In replay it is scored on its plan, and its
  `plan`, `planned_grid` and `effective_grid` are written as ints.
- A fractional plan runs the fallback grid live and in replay alike. Today it runs with undefined
  meaning. `planned_grid` is `None` and `used_fallback` is true, as for any rejected plan.
- The live-plan signal and the next live plan now record a fractional plan as the fallback grid
  with `fallback: true`, no longer as an error.

### 3.2 `online()` fails before spending when the plan cannot be recorded

**The check.** After planning, and before the baseline evaluation or any agent call, `online()`
serialises the manifest fields it already knows with `json.dumps`: `beta`, `planned_grid` (the
validated plan, including `reason`) and `effective_grid`. If that raises, `online()` raises
`RuntimeError`, naming the iteration and the error. At that point `runs/iterNNNN/` does not yet
exist, so nothing is left behind and no attempt has run.

**Why it is still needed after §3.1.**
- The sweep already scores such a version −∞. `beta_sweep.json` holds `default_beta`, and the
  episodes file holds each plan's `reason`, so a version whose beta or reason cannot be written
  fails its sweep and is never deployed.
- What the sweep cannot catch are the initial policy, which is never swept, and a plan whose
  `reason` or counts become unwritable only once the full history exists. The next live plan
  records only counts.
- The check is the last guard before paid calls.

**Considered and not taken:** an atomic freeze, written to a temp directory and then renamed.
With §3.1 and this check, the known cause of a half-written pool entry fails before the cycle
runs. The restart guard already names a partial `trace_pool/iterNNNN/` for deletion.

### 3.3 Ledger

A new GAPS §3 row, **"Grid counts"**:
- Listing 2 says `GridPlan` "accepts arbitrary integers" and that the runner checks the range.
- The rule is §3.1, and the early check is §3.2.
- The row gives both consequences and its pins.

The "Live-plan signal in replay" and "Next live plan" rows replace their PR #9 sentence about
numpy counts and non-integers with a pointer to the new row.

## 4. Part B: instrumentation out of the agent's view

### 4.1 `see sweep --instrumentation DIR`

- **The option.** It is new and optional, and defaults to `--out`. `cmd_sweep` writes a third
  file, `DIR/instrumentation.json`, after the two agent-facing files:

  ```json
  {
    "beyond_support": false,
    "next_live_plan": {"branch_count": 4, "refine_count": 4, "fallback": false, "beyond_support": true},
    "episodes": [
      {"trace_id": "iter0001", "beta": 0.0,
     "live_plan": {"branch_count": 4, "refine_count": 3, "fallback": false},
     "beyond_support": false, "live_plan_error": null}
    ]
  }
  ```

  `episodes` follows the order of `policy_execution_traces.jsonl`, so line i of that file and
  `episodes[i]` describe the same episode.
- **The agent-facing files.**
  - `beta_sweep.json` no longer has `beyond_support` or `next_live_plan`.
  - Each line of `policy_execution_traces.jsonl` no longer has `live_plan`, `beyond_support` or
    `live_plan_error`.
  - Everything else is unchanged, including `out_of_support` in both files.
- **Where the split happens.** `cmd_sweep` separates the fields when it writes the files.
  `beta_sweep` and `run_episode` keep computing them in memory, so their return values, and the
  tests that read them, are unchanged.

### 4.2 Where the loop puts it

- **The paths.** `DreamRSI._sweep(method, rdir)` passes
  `--instrumentation <workdir>/instrumentation/<basename(rdir)>`. A version therefore writes to
  `instrumentation/rNNNN_tNN_mM/instrumentation.json`, and the floor writes to
  `instrumentation/baseline/`.
- **Why that is out of view.** The policy-development agent runs with `cwd=policy_dev/`, and its
  prompt names `policy_dev/history/` and `trace_pool/`. `instrumentation/` is outside all three.
- **Placement, not access control.** An agent with a shell could still go looking. GAPS says so.
- **The workdir layout docstring** in `see/loop.py` gains the directory.
- **Failed sweeps.** A sweep that fails (a crash or a timeout, which `_sweep` rewrites as −∞)
  may leave no instrumentation file. The report then reads "not measured" for that version.

### 4.3 Restart guard

- **What it checks.** `check_iteration(t)` also refuses when any `instrumentation/r*_tNN_m*`
  exists, and names it with the others.
- **Why.** Without it, a restart would silently overwrite an aborted iteration's instrumentation,
  which is exactly the "never merge into them" case the convention forbids.
- **Where it is written down.** The "Interrupted iteration handling" and "Archive-name guard on
  restart" rows and `.claude/CLAUDE.md`'s restart convention list the new directory.

### 4.4 `report_run.py`

- **Where it reads.**
  - Per version, the live-plan counts and the next live plan come from
    `<workdir>/instrumentation/<version>/instrumentation.json`.
  - A missing file reads "not measured". That covers D2a's evidence, which predates it, and
    failed sweeps.
  - The inline fields were only ever written by toy runs, so there is no reader for them.
- **What it shows.** The report's outputs, headline, columns and wording are unchanged.
- **Evidence.** `--copy-evidence` adds `instrumentation/*/instrumentation.json` to its allowlist.
  The existing redaction applies to it, and matters here because `live_plan_error` tracebacks
  carry paths.

### 4.5 Ledger and docs

- **GAPS "Live-plan signal in replay" and "Next live plan".** The fields live in
  `instrumentation/<round>/instrumentation.json`, not in the two files Listing 2 names. The
  policy-development agent is not pointed at them, which is placement and not access control.
  PR #9's sentence saying the agent reads `next_live_plan` is replaced.
- **README status row and `.claude/CLAUDE.md`'s "One question API" paragraph** say where the
  fields are written.

## 5. Part C: flakes

- **`test_loop.py::test_the_demo_entry_point_maps_sigterm_to_the_partial_freeze`.**
  - It asserts `"KeyboardInterrupt: signal 15" in err`.
  - Under `FORCE_COLOR`, Python 3.13 colours the child's traceback, and the ANSI codes split
    the string.
  - Fix: run the child with `PYTHON_COLORS=0` in its environment. That variable takes precedence
    over `FORCE_COLOR`.
  - This is the only test that matches a subprocess's traceback text; checked with grep on
    2026-09-26.
- **The two timing tests.** They are
  `test_command_agent.py::test_timeout_returns_even_when_an_escapee_holds_the_pipes` and
  `::test_agent_timeout_kills_the_whole_process_group`. Both use a 0.3 s agent timeout, and both
  failed once each under load averages of 6–8 during D2b.
  1. Reproduce first. Run each test repeatedly while CPU-bound busy processes load every core,
     and record which assertion fails and the measured timings.
  2. Then fix what the reproduction shows. The likely cause is the timeout expiring before the
     stand-in has forked, which would call for a readiness handshake. Alternatively, size the
     margins from the measured overhead. Either way, keep the claim each test states.
  3. If neither reproduces in 50 loaded runs, change nothing. Record that in the D2c hand-forward
     with the command used.

## 6. Testing

TDD throughout: each pin fails before its change. After GREEN, each pin is mutation-checked
against its most plausible regression; a pin that survives its mutant is rewritten.

Following the next-live-plan journal, the new paths are also fed each **input shape** real
policies produce:
- `int`;
- `np.int64`;
- `4.0` and `np.float64(4.0)`;
- `4.5`;
- `nan`;
- `"4"`;
- a `reason` that is not JSON-serialisable;
- a `beta` of `np.float32`.

Pins (names are finalised in the plan):

- **A1** `test_objective.py`: `validate_plan` turns whole-number counts of each type into ints
  and rejects fractional and non-numeric ones, by hand-computed cases.
- **A2** `test_loop.py`: a numpy-count policy runs a toy live cycle and freezes. The frozen
  manifest and trace hold ints, and `trace_pool/iter0001/` is complete.
- **A3** `test_loop.py`: an unwritable `beta` or `reason` makes `online()` raise before the first
  attempt. `runs/iter0001/` does not exist, and no agent was called.
- **A4** `test_ledger.py`: a 4.5 plan replays on the fallback grid, mirroring `online()`.
  `run_episode`'s `plan` is `None`.
- **B1** `test_ledger.py`: a sweep's `beta_sweep.json` and `policy_execution_traces.jsonl` hold
  none of the live-plan family, and `instrumentation.json` holds them in episode order. The D2a
  pin (m2 → 4 x 4, flagged, reward unchanged) reads the new file.
- **B2** `test_loop.py`: after a toy run, no file under `policy_dev/` contains `live_plan`,
  `beyond_support` or `next_live_plan`. The toy policies' own sources contain none of these
  strings; checked 2026-09-26. Each version and the floor have
  `instrumentation/<round>/instrumentation.json`. The deployed-version pin (its next live plan is
  the grid `online()` runs next) reads the new file.
- **B3** `test_loop.py`: the restart guard names an iteration's `instrumentation/r*_tNN_m*`.
- **B4** `test_report_run.py`:
  - the existing live-plan and next-live-plan report tests pass reading the new file;
  - a version without one reads "not measured";
  - `--copy-evidence` copies and redacts it.
- **C** The colour test passes with `FORCE_COLOR=3` set. The timing tests: see §5.

Existing tests that read the moved fields from `beta_sweep.json` or the episodes file are
switched to the new file; none is deleted. The CI pin moves to the collected count, which is
taken with `NO_COLOR=1` so that no ANSI codes reach `ci.yml`.

## 7. Out of scope

- Selecting on the next-live-plan flag (ruling 1).
- Showing `fallback` in the report's next-live-plan column (declined in PR #9's review).
- An atomic freeze (§3.2).
- Other D2c hand-forward items, which stay on that list: redaction on the run host, the Linux
  temp-path rule, reviewing `agent_stderr`, spread labels, and counting default-beta live plans.
