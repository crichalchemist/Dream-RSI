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
`see/live.py:AGENT_PRESETS`. (Amended while planning: both run isolated from the owner's global
CLI configuration, passed as JSON argv; section 11.)

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
--eigen-include /opt/local/include/eigen3 --json evidence/d2a-lasso/host.json` on this host, under
the run's `PATH` and with the evaluator's binary cache cleared, before the run: proves the toolchain
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
python scripts/run_dream_rsi.py --simpletes /Users/controlroom/Dream-RSI/SimpleTES \
    --task lasso_path --workdir /Users/Shared/dream-rsi-runs/d2a-lasso \
    --discovery-agent "$(cat discovery.json)" --policy-agent "$(cat policy.json)" \
    --iterations 2 --versions 3 --workers 4 --grid 4 3 --hard-max 6 4 \
    --agent-timeout 900 --eigen-include /opt/local/include/eigen3
```

(Amended while planning, section 11: the workdir is outside the repository, and the two agents
are isolated JSON argv. The plan's Task 5 builds `discovery.json` and `policy.json`, proves them
with a one-call pre-flight each, and launches the run detached under `nohup`, in a session of
its own.)

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
  iteration under `runs/`; one restart is allowed (move exactly the directories the guard names
  aside, keeping them as evidence, and rerun; that iteration's calls are spent again). A second failure ends the experiment with
  what exists: a failing run is evidence too. Quota errors show up as failed attempts in the
  health table; the run is not retried into a broken quota.

## 5. The evidence report

`scripts/report_run.py --workdir <dir> --out <dir>` reads a finished or partial workdir and
writes `report.md` and `report.json` (the same numbers, machine-readable). It touches no loop
code and imports only `see.world` (to load traces) and `see.live.node_dirname` (the attempt
directory name); the caps and the program's file name come from the runner's launch record
(section 11). Four sections, one per D2 question:

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

**Tests.** A scripted-agent workdir built inside the test (toy task, a 2 x 1 fallback grid, i.e.
two branches of two attempts, one attempt
deliberately left byte-identical to its parent, one policy version that plans wider than the
tree), every count hand-computed, in the pin style of `tests/test_ledger.py`; the resume-source
rule gets its own pin against `see/live.py`'s so the two cannot drift.

## 6. Evidence and licensing

SimpleTES is AGPL and not vendored; every attempt's program is a modified copy of its seed, so
programs do not enter the repository. Committed under `reconstruction/evidence/d2a-lasso/`:
`state.json`, `launches.jsonl`, the frozen trace pool (traces and manifests), every policy
version's `method.py` and sweep output (`beta_sweep.json` and `policy_execution_traces.jsonl`;
written against the frozen policy API, not derived from SimpleTES), each
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
2. The runner's launch record, `launches.jsonl` (section 11), with its pins.
3. `scripts/report_run.py` with its hand-computed pins.
4. SimpleTES clone and the smoke test (owner's go; CPU only; run by the controller).
5. The pre-flight and the run (owner's go; spends budget; run by the controller).
6. Report, evidence scan and commit, GAPS §5, docs.

(Amended while planning: six tasks, section 11.)

Process as before: this spec, then the writing-plans skill, then subagent-driven execution with a
task review each, a whole-branch review at the end, and the finishing menu.

## 10. Open questions

None that block the plan. Whether GAPS section 5 should carry a second machine row for this
host's Listing 3 timings is left to the report's numbers; the smoke test produces them either way.

## 11. Amendments while planning (2026-09-25)

The plan (`docs/superpowers/plans/2026-09-25-real-agent-run-d2a.md`) was written from a spike.
Every code block in it was run in a throwaway worktree first. These changes came out of that
spike, and each overrides the section it names.

1. **Workdir outside the repository and the home directory** (section 4):
   `/Users/Shared/dream-rsi-runs/d2a-lasso`, with the run directory `chmod 700`.
   - The discovery agent has a shell. Inside `reconstruction/runs/` it could read
     `generated/lasso_path_dream_rsi.py` (the paper's final answer), edit tracked files, and load
     this repository's `.gemini/GEMINI.md`.
   - Both CLIs read context files from the cwd's ancestors, and `/Users/controlroom/GEMINI.md`
     exists; `/Users/Shared`'s ancestors hold none.
2. **Isolated agents** (section 1, **Decision (owner, 2026-09-25)**).
   - Why:
     - The owner's Gemini settings disable yolo mode, and they load the superpowers and
       context-mode extensions and hooks.
     - The owner's Claude setup loads a global CLAUDE.md, plugins, and hooks that write into
       the owner's notes.
     - A headless call cannot pass an approval gate. An attempt could then end with its
       program untouched, which is exactly the count D2a measures.
   - Gemini:
     - runs under a dedicated `HOME` holding only its login files, the owner's auth type and
       `billing` block, and trust for the run directory;
     - is pinned to the owner's model setting, `auto-gemini-2.5`;
     - gets `--include-directories <workdir>`, because Listing 1 sends it outside its cwd.
   - Claude:
     - runs with `--setting-sources project,local --strict-mcp-config` and
       `--add-dir <workdir>/trace_pool`;
     - falls back to a dedicated `CLAUDE_CONFIG_DIR` if the pre-flight shows user instructions
       still loading.
   - Both are JSON argv; no code change.
   - The launch environment unsets the controlling session's `CLAUDE*` variables.
   - The pre-flight is one call per agent. It must show that the agent authenticates, reads
     outside its cwd, writes a file, loads no extension, and fires no user hook, and that no
     ancestor of the workdir holds a context file.
   - The launch `exec`s the runner through `setsid`, so ending this session cannot take the run
     with it.
   - The discovery agent's file tools can write anywhere in the workdir. So the task tree
     (`task/`, the baseline included) is hashed after launch and re-checked before the report.
3. **Launch record** (sections 5 and 9, new Task 2). The runner appends one line per launch to
   `<workdir>/launches.jsonl`, before the first iteration. The line holds:
   - the caps and the program's file name, which no manifest carries;
   - each agent's argv and CLI version;
   - the host toolchain: platform, CPU, `g++`, Eigen version and SimpleTES commit, captured at
     run time.

   The report reads the record's last line.
4. **The report's imports** (section 5): `see.live.node_dirname`, beside `see.world`.
5. **The evidence subset** (section 6) gains:
   - `launches.jsonl`;
   - each version's `policy_execution_traces.jsonl`, the per-episode source of the
     out-of-support and empty-batch counts (`beta_sweep.json` keeps three errors and one
     `any()` flag).

   `report_run.py --copy-evidence` copies an allowlist to `evidence/d2a-lasso/workdir/`, and
   withholds any file quoting `CPP_CODE`. `reconstruction/evidence/` is excluded from ruff and
   from the whitespace hooks, so evidence stays as written. Before the commit, a scan counts
   pasted programs, quoted source, home paths, the owner's email address and files over
   500 KB. What to redact is the owner's call.
6. **Restart** (section 4). The guard-named directories are moved aside to
   `/Users/Shared/dream-rsi-runs/d2a-lasso-restart1/`, with `task/`, `launches.jsonl` and
   `state.json` copied beside them, instead of being deleted. The partial iteration is evidence
   too, and it is reported as a workdir of its own.
7. **The report's test run** (section 5): a 2 x 1 fallback grid and 3 x 2 hard caps. The wider
   policy version asks for 3 x 1.
8. **Controller-run tasks** (section 9). The smoke test, the run and the evidence commit are
   run by the session itself, never by a subagent. The first two need the owner's go, and the
   third has the owner's redaction decision in the middle.

## 12. Amendments during the run (2026-09-25)

These override section 1 and amendment 2 for the run that was made. GAPS §5, "D2a run", records
the outcome.

1. **Discovery agent: the Antigravity CLI** (owner decision).
   - Why: the owner's Gemini CLI login had expired (`invalid_grant`). A fresh login was then
     refused with "This client is no longer supported for Gemini Code Assist for individuals".
   - Discovery ran `agy -p {prompt} --model gemini-3.1-pro-high --dangerously-skip-permissions
     --add-dir <workdir>` under a dedicated `HOME` holding only the owner's Antigravity token and
     a minimal settings file.
   - The pre-flight also checked that `agy plugin list` and `agy mcp list` were empty, and that
     no `.agents/rules` sat above the workdir.
   - agy's leak probe asked for a phrase from the owner's global Gemini instructions
     (`~/.gemini/GEMINI.md`), since the Claude probe's phrase is not in that file.
2. **Two failures, quota first.** agy's logs and each attempt's `agent_returncode` show two
   separate failures.
   - The account's individual Gemini 3.1 Pro quota ran out during iteration 1's last round, at
     18:02. Two of that round's agents exited 3, and so did all four of iteration 2's first
     round.
   - Parallel `agy` calls sharing one `HOME` corrupted the token file at the hourly refresh
     (18:18). Iteration 2's second round could not authenticate and exited 1.

   The fix gives each call its own `HOME`, seeded from a read-only base. It addresses the second
   failure only; it was verified with four concurrent calls but never used in a launch.

   D2b needs:
   - a quota that holds two iterations of up to 30 calls;
   - each discovery call's stderr, or its tail, kept beside its exit code. The loop keeps only
     the exit code, which is why the workdir could not tell quota from auth.
3. **Stopped** (owner decision, under the stop rule of section 4). A restart failed its
   pre-flight on the same quota (HTTP 429), so the run ended with 1 of 2 iterations. The
   stopped iteration is reported from its aside directory.
4. **Evidence scan.**
   - The scan also counts any `@` address, not only the owner's git email, because the
     Antigravity account's address differs.
   - It also counts compiler-quoted source lines (`NN | code`), which neither `CPP_CODE` nor
     `#include` catches. One agent-written line was quoted six times, in 5 files. It was
     replaced by `[source line withheld]`, together with its caret line, under the constraint
     that no C++ source enters the repository.
   - The owner chose redaction (b): the home prefix became `~` in the two copied
     `launches.jsonl` files.
   - `report_run.py --copy-evidence` still withholds only whole files quoting `CPP_CODE`.
     D2b should redact these lines at copy time.
5. **Pre-flight hardening.** A Gemini CLI call hung for ten minutes at a browser-login prompt
   behind `| tail`. From then on:
   - each pre-flight agent call runs under a 300 s watchdog, with stdin from `/dev/null`;
   - the pre-flight writes its output to a log file;
   - a Gemini CLI call gets `NO_BROWSER=true`, so a lapsed login exits (41) at once. This became
     moot when agy replaced the Gemini CLI.
