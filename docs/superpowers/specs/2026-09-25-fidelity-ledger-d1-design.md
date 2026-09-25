# Fidelity and ledger, part 1 (track D1): rulings, pins and four consistency fixes

Date: 2026-09-25. Branch: `spec/fidelity-ledger-d1`, from `spec/runtime-safety-c2` at aaaf056 (PR #5
open at the time of writing; the branch is rebased onto `main` once #5 merges).
Owner decisions are recorded inline as **Decision**.

## 1. Context

The maturity audit's track D ("fidelity and ledger") asks that any number the platform publishes
can be traced to a recorded choice. It names two audit items, 3.6 (the per-round call budget and
the `hard_max_grid` default) and 3.9 (GAPS §1 files the legal set as "specified exactly" while the
code implements Listing 2's departure), and a list of conventions nobody has ruled on: attainment
on a flat trace, the root inside Eq. (1)'s max, the default-beta episodes, clip-to-support,
`probe_batch([])`, the floor versus `initial_policy`, objective validation, and a "how to change"
column for GAPS §3.

Tracks B and C closed parts of it: track B lowered the hard caps to (32, 19) so the product is at
most 640 and pinned that ceiling, and made replay use the fallback grid for a rejected plan; track
C1/C2 documented the child-process lifetime and routed two state questions here (`_deploy` saves
the new policy before `run()` advances the iteration counter; a fault a policy swallows leaves the
manifest without an error). Track B's queue also holds the replay-equivalence claim that is stated
in a comment but never asserted.

A fact check of the code before this design (nine readers, one per ruling) corrected three
assumptions: no test pins that the default fallback grid yields 110 calls in 11 rounds (the audit
and the track B plan both assumed one); `python -m see demo` hardcodes its own grids, (4, 5) and
(8, 8), rather than sharing the `LoopConfig` defaults; and `scripts/run_dream_rsi.py --objective`
already restricts its value to two choices while `LoopConfig` itself accepts any string.

**Decision (shape).** Track D is cut in two. **D1 (this spec)** is ledger-first: rule on each open
convention, record it in GAPS §3 with a "How to change" column, pin it with a hand-computed test,
and fix the four places where the code contradicts its own ledger. **D2** holds every change that
would move a number already in the ledger, and waits for one real-agent run to show which
departures from the paper matter (§9).

**Decision (the four fixes).** All four consistency fixes are in D1, each with a test: the
timed-out attempt's fail class, objective validation, one `state.json` write per iteration, and the
swallowed fault in the manifest.

**Decision (approach).** A wider GAPS §3 table plus a standalone pin module `tests/test_ledger.py`,
rather than a conventions registry in code (machinery for 24 rows that the paper-critique sections
of GAPS do not fit) or rows and fixes without pins (rulings without tests drift the way the C1 spec
drifted from its code within a day).

## 2. Goals, non-goals, constraints

Goals:

1. Every reconstruction choice that affects a published number has a GAPS §3 row that says what
   the paper gives, what was chosen, and how to change it.
2. Every such row is pinned by a test with a hand-computed expectation.
3. A timed-out attempt is recorded as the failure it is; a mistyped objective fails before budget
   is spent; `state.json` is written once per iteration, after the deploy; a fault a policy
   swallows still reaches the manifest.
4. The replay-equivalence test asserts what its comment claims.

Non-goals: any change to the numbers the ledger already publishes (all of §9 is D2); the GAPS split
into a contributor ledger and a paper-reader critique (track G, sequenced after D); the C1 spec's
own C2 and C3 lists (track C); a per-round budget knob; resumability.

Constraints carried from tracks A to C: the gate stays strict (`ruff format --check .`,
`ruff check .`, `pyright` basic with 0 errors, `python tools/extract_listings.py --check`,
`python -m pytest -q` with zero skips, `tools/check_junit.py --expect N`); no `xfail`, `skip`,
`# noqa`, `# type: ignore`; nothing public in `see/policy/api.py` or
`see/policy/observation_signal.py` changes and no new `fail_class` string is introduced (the
timed-out fix reuses the existing `timeout` class); no workdir directory is renamed; the agent
callable's signature is unchanged; every paper-silent choice gets a GAPS §3 row; one commit per
task, plain imperative messages, no attribution trailers; fixes land with their tests, shown
failing on the previous tree.

**Decision (validation).** Pins are hand-computed cases on the toy world (`see/toy.py`,
`see/synthetic.py`), never expectations copied from a run. Existing pins are extended, not
duplicated: `tests/test_objective.py` already covers the 640 ceiling, the flat-trace attainment
for `best=None`, Eq. (1)'s formula, the rejected-plan fallback and the −∞ for an illegal batch;
`tests/test_world.py` covers the legal roots, the empty batch and the classifier.

## 3. The four fixes

### 3.1 Timed-out attempt (`see/live.py:_run_attempt`)

Today a timed-out agent's attempt evaluates whatever program is in the node (its parent's copy if
the agent touched nothing), the error comes only from the evaluator, so the cell is a success with
fail class `ok` and the parent's score. The classifier has a `timeout` class this path never
produces.

After the fix: when the agent result carries `timed_out` and the evaluator reported no error, the
attempt's `error` is the agent's own message ("agent timed out after Ns"), which
`classify_failure` maps to `timeout`. The program is still evaluated and its score recorded, so an
agent that finished an improvement before hanging keeps its number in the trace; but the cell is
not a success: `is_success` is false, it does not raise the trace ceiling (§4 ruling 8), and a
policy sees a repairable `timeout` failure. If the evaluator itself errored, its error wins; the
program is broken either way and `score.json`'s `agent_timed_out` keeps the cause. Children resume
from the file it left, as before. `error.txt` is written, as for any error.

Tests: `test_agent_timeout_kills_the_whole_process_group` expects `("b0a0", 1.0, True, "timeout")`
with the error set; a new test evaluates a timed-out attempt that left a broken program and expects
the evaluator's class, not `timeout`.

### 3.2 Objective validation (`see/objective.py`, `see/loop.py`, `scripts/run_dream_rsi.py`)

`OBJECTIVES = ("pareto", "eq1")` in `see/objective.py`. `LoopConfig.__post_init__` raises
`ValueError` naming the choices when `objective` is not one of them; `score_of` raises the same for
an unknown name instead of falling through to Eq. (1). The runner's `--objective` takes
`choices=OBJECTIVES`. Tests: a mistyped objective fails at `LoopConfig` construction; `score_of`
returns `pareto.reward` and `eq1.V` for the two names and raises for a third.

### 3.3 One `state.json` write per iteration (`see/loop.py`)

Today `_deploy` saves (deployed, round, digest), `offline()` saves again (the log), and `run()`
saves a third time (the iteration counter), so a crash between the first and the third leaves the
deployed policy one iteration ahead of the counter. After the fix: `_deploy` does not save;
`offline()` sets `state["iteration"] = t` and saves once at its end; `run()` only loops. The
one-time save in `__init__` stays. A crash anywhere inside `offline()` leaves `state.json` as the
previous iteration left it, which is what the restart guard already assumes. Tests: `_save_state`
is called once per `offline()` and never by `online()`; after `online(1)` and `offline(1)` the file
holds iteration 1, `deployed/iter0002.py`, its digest and the log entry with `live`, `offline` and
`selected`; a fault injected right after a successful `_deploy` leaves the file byte-identical to
the one before `offline(1)`.

### 3.4 Swallowed fault in the manifest (`see/live.py`, `see/loop.py`)

`_cancel` records the first abandoning exception as `LiveQuestion.fault` (`"<Type>: <message>"`); a
later refusal of the same abandoned batch does not overwrite it. When
`solve` returns normally with `q.fault` set, `online()` records
`error = f"batch abandoned: {q.fault}"` in the manifest; the tree is frozen into the pool with that
error, exactly as when the fault propagates. Test: a policy whose `solve` catches the batch's
exception and returns; the host fault is a missing prompt directory (`FileNotFoundError` raised
by `exploration_prompt` in the worker before any agent call); the manifest's error starts with
"batch abandoned: FileNotFoundError", `probes` is 0 and the agent was never called.

## 4. Rulings

Each ruling is a GAPS §3 row or a clause in an existing one, and a pin.

1. **Per-round call budget.** The paper's 110 (Pro) and 640 (Flash) are the fallback grid's size,
   10 × (10 + 1), and the hard caps' product, 32 × (19 + 1). No product cap and no budget is
   enforced: a policy may plan (32, 0) or (1, 19). The demo uses (4, 5) and (8, 8). Pin: a scripted
   live episode under the default fallback grid runs exactly 110 attempts in 11 rounds. Budget
   enforcement is D2.
2. **Legal set and the empty batch.** GAPS §1's rows "Tree structure, legal set A(T)" and "Replay
   reveal rule, termination" are split: the tree structure, batch size ≤ W, the reveal rule, K2 and
   exhaustion stay in §1; "one root per unopened branch (Listing 2), not Sec. 3's {r} ∪ leaves"
   becomes a §3 row pointing at §4 item 1; `probe_batch([])` becomes a §3 row: illegal
   (`IllegalBatch`), and in replay it scores the version −∞ like any other exception. Already
   pinned (`test_world.py::test_illegal_batches_are_refused`,
   `test_objective.py::test_crashing_or_illegal_policy_is_scored_minus_infinity`).
3. **Attainment on a flat trace** is 1.0 for every episode, whatever `best` is, because nothing
   recorded beats the root. The existing row gains the sentence; the existing test gains the
   `best` case.
4. **Eq. (1)'s quality term** is the max over the revealed subtree including the root, so a policy
   that reveals only worse cells scores the root. New clause; pin: `best` below the root gives the
   root.
5. **Two objectives, two episode sets.** `pareto.reward` is computed from the beta-grid episodes
   (one per trace per beta), `eq1.V` from the extra default-beta episode per trace, and an error in
   either set invalidates both. New row; pin on a scripted sweep with a policy whose default beta
   differs from every grid beta.
6. **Out-of-support plans** are clipped to the trace grid and scored as the clipped plan; the
   report's `out_of_support` flag is informational. Pin: an oversize plan yields the flag and the
   same reward as the clipped plan. Zero-versus-clip is D2.
7. **The floor.** `policy_dev/history/baseline/` is re-swept every offline phase for reference and
   is never a candidate; `LoopConfig.initial_policy` seeds `deployed/iter0001.py` only on a fresh
   workdir. Pins: the candidates of an `offline()` never include the floor; a second `DreamRSI` on
   the same workdir with another `initial_policy` leaves `iter0001.py` unchanged.
8. **The trace ceiling** counts successful cells only, so a timed-out or failed attempt's score
   never raises it. Pin.
9. **Objective names** are the two in `OBJECTIVES` (§3.2). Row names the knob.
10. **Replay equivalence.** `tests/test_replay_equivalence.py` asserts what its comment claims:
    seeds 0 and 1 leave the portfolio's live tree partial (fewer recorded cells than the grid
    holds) while seed 2 and every parallel-refine episode fill it; and each replay's `Episode.plan`
    equals the manifest's `planned_grid`.

## 5. The GAPS §3 table

§3 gains a fourth column, "How to change", filled for all 24 existing rows and the new ones. Values
are one of: a `LoopConfig` field (with the `scripts/run_dream_rsi.py` flag when one exists:
`--versions`, `--workers`, `--grid`, `--hard-max`, `--objective`, `--agent-timeout`,
`--iterations`, `--discovery-agent`, `--policy-agent`); a `python -m see sweep` flag (`--betas`,
`--lam`, `--beta1`, `--beta2`, `--max-rounds`, `--fallback`, `--hard-max`); a callable passed to
`DreamRSI` (`discovery_agent`, `policy_agent`, `directions`) or a `CommandAgent` argument
(`timeout`, `kill_grace`, `env`); or "fixed in `see/<file>.py`" for a mechanism with no knob. The
fact-check reports are the source; every value is checked against the code by the task's reviewer.

§1's two rows are split as ruling 2 says; §1 keeps only what Sec. 3 states.

## 6. Tests

New module `tests/test_ledger.py` (rulings 1, 4, 5, 6, 7, 8): one test per ruling, named for the
claim it protects, hand-computed on the toy world. Extended existing tests: ruling 3 in
`test_objective.py`; ruling 10 in `test_replay_equivalence.py`; the timed-out expectation in
`test_command_agent.py`. Per fix (§3): two tests for 3.1, two for 3.2, two for 3.3, one for 3.4,
in the module that already covers that path. Estimated count 82 → about 96; the last task
measures and pins it.

## 7. Verification

The full gate on the final tree; every fix's tests shown failing on the previous tree (stash the
source file, run, pop); the GAPS table read row by row against the code by the reviewer of task 7;
`python -m see demo` end to end. Suite time may grow by a second (the 110-attempt episode is
scripted, milliseconds per attempt).

## 8. Task order

1. Timed-out attempt (§3.1).
2. Objective validation (§3.2).
3. One `state.json` write per iteration (§3.3).
4. Swallowed fault in the manifest (§3.4).
5. `tests/test_ledger.py` (rulings 1, 4, 5, 6, 7, 8) and ruling 3's extra case.
6. Replay-equivalence assertions (ruling 10).
7. GAPS: the "How to change" column, the new rows, the §1 split (rulings 1, 2, 5, 7, 9 and the four
   fixes' clauses).
8. README and CLAUDE.md wording; CI pin to the measured count.

Process as before: this spec, then the writing-plans skill, then subagent-driven execution with a
task review each, a whole-branch review at the end, and the finishing menu.

## 9. D2: behaviour changes, gated on one real-agent run

Each of these moves a number already in the ledger, and waits for one real-agent run (a small
budget on the Lasso task) to show which departures from the paper matter:

- Enforce a per-round call budget: a product cap in `validate_plan` or a `LoopConfig` budget,
  decided then; the paper's 110 and 640 are the candidates.
- Score an out-of-support replay as zero rather than as the clipped plan.
- Treat an attempt whose program is byte-identical to its resume source as `no_program`, timed out
  or not; and skip the evaluation of a timed-out attempt that touched nothing.
- Whether `probe_batch([])` should end the episode rather than score −∞.

Stays with track C: the C1 spec's C2 and C3 lists. Stays with track G: the GAPS split.

## 10. Open questions

None that block the plan. The demo's hardcoded grids stay (documented in ruling 1); a demo flag
would be track G.
