# The real-agent Lasso run (track D2a): evidence for the D2 behaviour changes

Date: 2026-09-25. Branch: `spec/real-agent-run-d2a`, from `main` at 836889d (tracks C2 and D1
merged). Owner decisions are recorded inline as **Decision**.

## 1. Context

The D1 spec (`2026-09-25-fidelity-ledger-d1-design.md`, section 9) deferred four behaviour
changes to D2 and gated them on one real-agent run: a per-round call budget, scoring an
out-of-support replay as zero rather than as the clipped plan, treating an attempt whose program
is byte-identical to its resume source as `no_program`, and whether `probe_batch([])` should end
the episode rather than score minus infinity. Each moves a number the ledger already publishes,
and none of the reconstruction's evidence so far comes from a real coding agent: every test and
the demo use scripted agents on the toy task.

**Decision (shape).** D2 is cut in two. **D2a (this spec)** produces the run and a report that
answers the four questions with counts. **D2b** designs the changes from that report in its own
cycle. The spec's gate is kept in its intended order: evidence first, then changes.

**Decision (agents).** Discovery on the Gemini CLI (the paper's backbone family, with the
Listing 1 prompt written for it; the account logged in on this machine), policy development on
the Claude CLI (few calls; authenticated here). Both are the existing presets in
`see/live.py:AGENT_PRESETS`.

**Decision (size).** Two iterations, a 4 x 3 fallback grid, 4 workers, 3 policy versions:
about 32 to 46 discovery calls and 4 policy calls, one to two hours wall clock.

**Decision (approach).** Run on this machine through the existing runner, with an Eigen include
hook in the SimpleTES adapter and a report script that turns a workdir into the evidence. Not a
Linux container for the evaluator (a moving part, and timing inside a VM adds noise to a
timing-based score) and not a hand-written report (not reproducible; D2b would rest on a
reading).

## 2. Goals, non-goals, constraints

Goals:

1. The loop runs end to end on the paper's Lasso task with real coding agents, on this machine.
2. A script answers the four D2 questions from the run's workdir with counts and the cases behind
   them, and shows the run was sound.
3. The evidence is committed, license-safe, and the ledger records the host toolchain choice and
   the run's headline numbers.

Non-goals: any change to the loop, the scoring or the policy API (all four D2 items are D2b); a
reproduction of the paper's numbers (different backbone, host and budget); variance (one run);
a Gemini-versus-Claude comparison; a budget knob (D2b).

Constraints carried from tracks A to D1: the gate stays strict (`ruff format --check .`,
`ruff check .`, `pyright` basic with 0 errors, `python tools/extract_listings.py --check`,
`python -m pytest -q` with zero skips, `tools/check_junit.py --expect N`); no `xfail`, `skip`,
`# noqa`, `# type: ignore`; nothing public in `see/policy/api.py` or
`see/policy/observation_signal.py` changes; no workdir directory is renamed; the agent callable's
signature is unchanged; every paper-silent choice gets a GAPS section 3 row with its "How to
change" cell and a pin; one commit per task, plain imperative messages, no attribution trailers.
SimpleTES is AGPL and stays unvendored.

## 3. Host and build

The machine: Intel Core i7-7700K (4 cores, 8 threads), macOS 13.7.8, MacPorts. `g++` resolves
to MacPorts gcc 13.4.0 through `port select gcc mp-gcc13`; the `eigen3` port (3.4.1) is at
`/opt/local/include/eigen3`. An Eigen plus OpenMP test program compiles and runs through plain
`g++` with `-fopenmp -I/opt/local/include/eigen3`. The venv has the `lasso` extras (numpy,
scikit-learn, psutil).

SimpleTES's Lasso evaluator compiles each program with a hardcoded `g++` and looks for an `eigen`
directory inside the task directory before falling back to `-I/usr/include/eigen3`, a Linux path
that does not exist on macOS. The adapter already copies the task directory without the vendored
`eigen` tree (it lacks `Eigen/Core`).

**The hook.** `see.tasks.simpletes_task(simpletes_dir, name, workdir, *, eigen_include=None)`:
when given, the adapter creates `task/src/eigen` as a symlink to that directory (the root that
contains `Eigen/`), which is exactly where the evaluator looks first. `scripts/run_dream_rsi.py`
and `scripts/verify_lasso.py` gain `--eigen-include`. Without it, behaviour is unchanged (the
Linux fallback). No compiler hook: an OpenMP-capable `g++` first on `PATH` is a documented
requirement (`port select` on macOS, `g++` on Linux), and the smoke test proves it before any
budget is spent.

**Smoke test.** `python scripts/verify_lasso.py --simpletes ../SimpleTES --repeats 2
--eigen-include /opt/local/include/eigen3` on this host, before the run: proves the toolchain
end to end (compile, 17 instances, the paper's Listing 3) and measures this host's evaluation
noise for the report. Its JSON output is kept as `evidence/d2a-lasso/host.json`. GAPS section 5's
existing numbers (a Linux Xeon) stay as they are.

**Ledger and pin.** A GAPS section 3 row "Lasso host toolchain": what the evaluator assumes
(`g++`, `task/eigen` or `/usr/include/eigen3`), what was chosen (the symlink hook, MacPorts gcc
via `port select`), how to change (`--eigen-include`; the compiler is whatever `g++` resolves
to). Pin: a fake SimpleTES task directory (an `evaluator.py` that records whether `task/src/eigen`
exists and where it points, an `init_program.py`, a statement file) under `tmp_path`;
`simpletes_task` with and without `eigen_include`; hand-computed, no compiler involved.

## 4. The run

From `reconstruction/` in its venv, after `git clone --depth 1 https://github.com/wq-will/SimpleTES
../SimpleTES` (the clone's commit hash goes into the report):

```
python scripts/run_dream_rsi.py --simpletes ../SimpleTES --task lasso_path \
    --workdir runs/d2a-lasso --discovery-agent gemini --policy-agent claude \
    --iterations 2 --versions 3 --workers 4 --grid 4 3 --hard-max 6 4 \
    --agent-timeout 900 --eigen-include /opt/local/include/eigen3
```

- **Grid and caps.** Fallback 4 x 3 (16 attempts per round). The hard caps are 6 x 4 rather than
  the default 32 x 19: no budget cap exists yet, so the caps are the only bound on what a
  developed policy may plan in iteration 2. Six by four leaves room above the fallback, so a
  policy that plans wider than the recorded tree can still show up, at a worst case of 30 calls
  in that round.
- **What happens.** Iteration 1: the paper's initial policy (parallel refine) drives 16 Gemini
  attempts; offline, the floor is swept, Claude writes two policy versions, each is replay-scored,
  the argmax is deployed. Iteration 2: the deployed policy plans its own grid and drives
  discovery; offline repeats.
- **Timeouts.** 900 s per agent call; SimpleTES's own 600 s evaluator timeout stays.
- **Everything else default.** Objective `pareto`, the paper's beta grid and lambda, no direction
  provider, the real Listing 1 and 2 prompts from `generated/`, evaluations serialized.
- **Launch and failure.** The run is launched from the session in the background with a log
  file, only after the owner says go (it spends budget). An interrupt or crash freezes the partial
  iteration under `runs/`; one restart is allowed (delete exactly the directories the guard names
  and rerun; that iteration's calls are spent again). A second failure ends the experiment with
  what exists: a failing run is evidence too. Quota errors show up as failed attempts in the
  health table; the run is not retried into a broken quota.

## 5. The evidence report

`scripts/report_run.py --workdir <dir> --out <dir>` reads a finished or partial workdir and
writes `report.md` and `report.json` (the same numbers, machine-readable). It touches no loop
code and imports `see.world` only, to load traces. Four sections, one per D2 question:

1. **Per-round call budget.** Per iteration, from the live manifest: the planned grid, whether it
   was rejected for the fallback, the effective grid, probes spent, and the cap product in force.
   Against the paper's 110 and 640: did anything push on a budget, and how far.
2. **Out-of-support replay.** Per policy version, from its sweep report and the executions file
   the sweep writes: episodes clipped, the plan they asked for versus the recorded grid, and
   whether the flagged versions were the deployed ones.
3. **Untouched programs.** Per attempt, the resume source recomputed by the loop's rule (the
   parent's program, else the nearest ancestor's, else the baseline) and compared byte for byte
   with the attempt's program; identical programs crossed with `score.json` (timed out, fail
   class, evaluated) to show what the loop scored them as today.
4. **Empty batches.** From every sweep report's error list and every manifest's error: versions
   scored minus infinity for an empty or illegal batch, and any live batch a policy tried to leave
   empty.

Then a health table per iteration (attempts, successes, fail classes, timeouts, no-program,
evaluator errors, baseline and best score) and per version (valid, reward, Eq. (1) value,
deployed or not); the host facts (machine, compiler, Eigen, SimpleTES commit); and the smoke
test's noise measurement from `host.json` when present.

**Tests.** A scripted-agent workdir built inside the test (toy task, a 2 x 2 grid, one attempt
deliberately left byte-identical to its parent, one policy version that plans wider than the
tree), every count hand-computed, in the pin style of `tests/test_ledger.py`; the resume-source
rule gets its own pin against `see/live.py`'s so the two cannot drift.

## 6. Evidence and licensing

SimpleTES is AGPL and not vendored; every attempt's program is a modified copy of its seed, so
programs do not enter the repository. Committed under `reconstruction/evidence/d2a-lasso/`:
`state.json`, the frozen trace pool (traces and manifests), every policy version's `method.py`
and sweep report (written against the frozen policy API, not derived from SimpleTES), each
attempt's `score.json`, `error.txt` and proposal, `host.json`, and both reports. The full workdir
archive stays on disk, git-ignored, with its sha256 in the report; the report script does the
byte-identity analysis at report time, so the programs are not needed in the repository.

## 7. Success criteria

1. The smoke test has run on this host and its numbers are captured.
2. The run completed two iterations, or its partial state is reported as such after at most one
   restart.
3. `report.md` answers the four questions with counts and cases, and the health table shows the
   run was sound.
4. The evidence subset and both reports are committed; the archive's sha256 is in the report.
5. The ledger and docs are in step: the GAPS section 3 row, a "D2a run" subsection in GAPS
   section 5 with the headline numbers and a pointer to the evidence, the README's Lasso section
   (MacPorts gcc via `port select`, `eigen3`, `--eigen-include`), the CLAUDE.md commands, and the
   CI count pinned to the measured suite.

## 8. Hand-forward to D2b

D2b's brainstorm opens on `report.json`. For each of the four items the decision is "observed:
change it" or "not observed: leave it, and say why", grounded in the counts. The report's health
table also says whether the run itself surfaced anything the audit did not anticipate.

## 9. Task order

1. `--eigen-include` in the adapter, the runner and `verify_lasso.py`; the fake-task pin; the
   README lines.
2. `scripts/report_run.py` with its hand-computed pins.
3. SimpleTES clone and the smoke test (owner's go; CPU only).
4. The run (owner's go; spends budget).
5. Report, evidence commit, GAPS rows, docs, CI pin.

Process as before: this spec, then the writing-plans skill, then subagent-driven execution with a
task review each, a whole-branch review at the end, and the finishing menu.

## 10. Open questions

None that block the plan. Whether GAPS section 5 should carry a second machine row for this
host's Listing 3 timings is left to the report's numbers; the smoke test produces them either way.
