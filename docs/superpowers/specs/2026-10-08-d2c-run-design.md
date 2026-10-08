# The complete real-agent Lasso run (track D2c)

Date: 2026-10-08. Branch `spec/d2c-run` from `main` ff256fb (PR #12 merged). Owner decisions are
recorded inline as **Decision**.

## 1. Context

Track D2a (`2026-09-25-real-agent-run-d2a-design.md`) ran the loop once on the Lasso task with
real coding agents and finished 1 of 2 iterations: the account's individual Gemini 3.1 Pro quota
ran out, and concurrent agent calls sharing one home corrupted the token file (GAPS §5, "D2a
run"). Track D2b (`2026-09-25-d2b-verdicts-and-instrumentation-design.md`) settled the verdicts
D2a's counts supported and built the instrumentation the next run needs: the untouched check, the
stderr tail, evaluation repeats, the live-plan signal, redaction at copy time. It named D2c as "a
complete two-iteration rerun on the owner's `dream-rsi` Google Cloud project with paid quota" that
"revisits the verdicts that one run could overturn", and handed forward (its §11 and §14):

- quota and auth through paid Gemini, with nothing enabled or billed without the owner's go;
- a Linux host with `g++` (OpenMP) and `libeigen3-dev`;
- the choice of k for `--eval-repeats`, and caps sized to the quota;
- pinned agent CLI versions;
- the three "not observed" verdicts, re-read with the live-plan signal in place;
- `report_run --copy-evidence` run on the run host; the Linux temp-path rule's missing leading
  boundary; `agent_stderr` reviewed by eye; the two spread definitions labelled; default-beta
  live plans counted separately; the two timing tests.

The D2c prep (`2026-09-26-d2c-prep-design.md`, PRs #10 and #12) then made the loop safe to run
with a policy that plans numpy counts, moved the live-plan family out of the policy-development
agent's view, fixed the colour flake, and found that the two timing tests do not reproduce under
load (its §8), so they stay as they are. PR #9 added the next live plan, which D2b's review made
D2c's first prerequisite.

Every item above except the run itself is done. This spec is the run.

**Decision (host).** This machine. It is the same Intel Core i7-7700K as D2a's, now on Linux
(Ubuntu, `g++` 15.2 with OpenMP), so evaluation noise stays comparable with D2a. A CPU VM in the
project was not taken: a different CPU would break that comparison, and it adds a VM to set up
and pay for. The discovery agent's shell is contained as in D2a, by a workdir outside the
repository and the home directory and by a dedicated home per agent.

**Decision (discovery agent).** The Gemini CLI, the paper's own, on Vertex AI billed to the
`dream-rsi` project. D2a's Antigravity CLI was a substitute for an OAuth login the Gemini CLI had
refused; Vertex AI does not go through that login. An AI Studio API key was not taken.

**Decision (credential).** A service account with the Vertex AI User role, its JSON key passed as
`GOOGLE_APPLICATION_CREDENTIALS`. No `gcloud` install, no token refresh, and so no per-call token
race of the kind that stopped D2a. Not taken: `gcloud` application-default credentials (adds
`gcloud` to the pinned toolchain) and a Google Cloud API key (a long-lived secret in an
environment variable every agent process can read).

**Decision (shape).** D2a's: 2 iterations, a 4 x 3 fallback grid, 6 x 4 hard caps, 4 workers, 3
policy versions, a 900 s agent timeout. 32 to 60 discovery calls and 4 policy calls. The budget
D2b's verdicts were sized for, and the only shape comparable with D2a.

**Decision (repeats).** `--eval-repeats 3`. D2a's eight re-evaluations of an unchanged program
spread 14%, about the size of its improvement; the median of three costs evaluator time only.

**Decision (process).** One spec, with the no-spend auth spike as its first task. The D2b spec
had called the spike separate; folding it in keeps its stop points (nothing enabled, created or
billed in the project without the owner's go) as explicit stops in the plan and saves a second
spec, plan and PR for an hour of work.

## 2. Goals, non-goals, constraints

Goals:

1. The loop runs both iterations end to end on the paper's Lasso task with real coding agents, on
   this host, with the D2b instrumentation recording what D2a could not: why each agent call
   failed, whether a version would plan beyond the recorded tree live, and how noisy each
   evaluation was.
2. The report and the license-safe evidence are committed under `reconstruction/evidence/d2c-lasso/`,
   and the ledger records the run.
3. The three "not observed" verdicts (per-round budget, zero versus clip, the empty batch) are
   re-read from the report, and each GAPS row says what the complete run showed.
4. The three report items D2b handed forward are fixed before the run produces the report.

Non-goals:

- a reproduction of the paper's numbers (different backbone settings, host and budget);
- any change to the loop, the scoring or the policy API; a "change" verdict in goal 3 is handed
  forward, not made here;
- a Gemini-versus-Claude comparison; variance across runs; KernelBench;
- more than one restart;
- the two timing tests (D2c prep §8: not reproduced in 50 loaded runs each; unchanged).

Constraints carried from tracks A to D2c prep:

- The gate stays strict: `ruff format --check .`, `ruff check .`, `pyright` basic with 0 errors,
  `python tools/extract_listings.py --check`, `python -m pytest -q` with zero skips,
  `tools/check_junit.py --expect N`, N moved in the same commit as the count.
- No `xfail`, `skip`, `# noqa` or `# type: ignore`.
- Nothing public in `see/policy/api.py` or `see/policy/observation_signal.py` changes; Listing 2
  is not edited.
- No workdir directory is renamed. The agent callable's signature is unchanged.
- Every paper-silent choice gets a GAPS §3 row with its "How to change" cell and a hand-computed
  pin.
- One commit per task, plain imperative messages, no attribution trailers.
- SimpleTES is AGPL and stays unvendored; no attempt's program enters the repository.
- `reconstruction/evidence/d2a-lasso/` is not changed.

## 3. Host and build

The machine: Intel Core i7-7700K (4 cores, 8 threads), 60 GB, Ubuntu with `g++` (Ubuntu
15.2.0) 15.2.0; an OpenMP test program compiles and runs with plain `g++ -fopenmp`. Eigen is not
installed: `apt install libeigen3-dev` (candidate 3.4.0-5) puts it at `/usr/include/eigen3`, the
path SimpleTES's evaluator falls back to, so the run passes no `--eigen-include`. The SimpleTES
clone at the repository root (gitignored) is at `47d3413`, D2a's commit, so the task is byte for
byte D2a's. The venv is Python 3.14.8 with the `extract`, `lasso` and `dev` extras.

**Smoke test.** From `reconstruction/` in the venv, with the evaluator's binary cache cleared and
under the run's `PATH`:

```
python scripts/verify_lasso.py --simpletes ../SimpleTES --repeats 2 --json evidence/d2c-lasso/host.json
```

It proves the toolchain end to end (compile, 17 instances, the paper's Listing 3) and measures
this host's evaluation noise for the report. GAPS §5's Xeon and D2a's macOS numbers stay as they
are; the report's noise table reads `host.json`.

**Workdir.** `/srv/dream-rsi-runs/d2c-lasso`, created with sudo and chowned to the owner, mode
700. Outside the repository, so the discovery agent cannot read `generated/` (the paper's final
answer) or edit tracked files; outside the home directory, so neither CLI finds a context file in
an ancestor (`/srv` and `/` hold none; checked 2026-10-08). The agents' dedicated homes and the
service-account key sit beside it under `/srv/dream-rsi-runs/`, never inside the workdir, so no
evidence copy can reach them.

## 4. Agents

Both agents are isolated from the owner's global CLI configuration and passed as JSON argv, as
in D2a (its §11, amendment 2). No code change.

**Discovery: the Gemini CLI on Vertex AI.**

- Installed with npm at a pinned version (`@google/gemini-cli@<version>`; the current release is
  0.63.0 as of 2026-10-08, and the spike records the one installed). The launch record keeps the
  CLI's `--version`.
- Model `gemini-3.1-pro-preview`, the Vertex AI model id as third-party listings give it on
  2026-10-08. The spike's first paid call confirms the id with the CLI's own output; if it is
  refused, the spike stops and the owner picks from the ids the CLI lists.
- Argv, with the environment set through `env` as D2a set `HOME`:

  ```
  ["env", "HOME=/srv/dream-rsi-runs/gemini-home",
   "GOOGLE_GENAI_USE_VERTEXAI=true", "GOOGLE_CLOUD_PROJECT=dream-rsi",
   "GOOGLE_CLOUD_LOCATION=<region>",
   "GOOGLE_APPLICATION_CREDENTIALS=/srv/dream-rsi-runs/dream-rsi-vertex.json",
   "gemini", "--yolo", "-m", "gemini-3.1-pro-preview",
   "--include-directories", "/srv/dream-rsi-runs/d2c-lasso",
   "--prompt", "{prompt}"]
  ```

  `--yolo` is the preset's; `--include-directories` because Listing 1 sends the agent outside
  its cwd. The region is the one the spike finds the model in.
- The dedicated home holds only `.gemini/settings.json`: the Vertex auth type and trust for the
  workdir. No extensions, no hooks, no `GEMINI.md`.
- A key file has no hourly refresh, so D2a's token-file race cannot recur. Each call still gets
  its own home, seeded from the read-only base as D2a's fix did, in case the CLI writes anything
  else there.
- `launches.jsonl` records the argv, so it records the key file's path. That is a path, not the
  key; the file itself is outside the workdir and is never copied.

**Policy development: the Claude CLI**, 2.1.293 as installed, with D2a's flags:

```
["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits",
 "--setting-sources", "project,local", "--strict-mcp-config",
 "--add-dir", "/srv/dream-rsi-runs/d2c-lasso/trace_pool"]
```

It falls back to a dedicated `CLAUDE_CONFIG_DIR` if the pre-flight shows the owner's user
instructions still loading. The launch environment unsets the controlling session's `CLAUDE*`
variables.

## 5. The spike: Task 1, controller-run, no spend

In order, each stop waiting for the owner's go:

1. **Install and pin the Gemini CLI** (no project change, no spend).
2. **Stop: project changes.** Enable the Vertex AI API on `dream-rsi`; create the service account
   with the Vertex AI User role; create and download its JSON key to
   `/srv/dream-rsi-runs/dream-rsi-vertex.json`, mode 600. These are done by the owner in the
   console or by commands the owner runs; nothing in the project is created by the session.
3. **Stop: the first paid call.** One prompt of one line through the argv above, under a 300 s
   watchdog with stdin from `/dev/null`, output to a log file. It must show the model id in the
   CLI's output, a response, and (in the console) the call billed to the project. This is the
   quota and auth check the D2b spec asked for.
4. **Stop: the pre-flight**, one call per agent, same watchdog and log. Each must show that the
   agent authenticates, reads a file outside its cwd, writes a file, loads no extension and fires
   no user hook, and that no ancestor of the workdir holds a context file. The leak probe asks
   for a phrase from the owner's global instructions (`~/.claude/CLAUDE.md`; the Gemini probe
   uses the same file, since `~/.gemini` is empty on this host). A probe that reproduces the
   phrase fails the pre-flight.

The spike's log files and the recorded versions go into the plan's hand-forward notes, not the
repository.

## 6. Code before the run: the three report items, one task

`scripts/report_run.py` only, with its pins in `tests/test_report_run.py`.

- **The temp-path rule's leading boundary.** `redact` rewrites the temp directory and the home
  directory wherever they appear followed by a non-name character. On Linux the temp directory is
  `/tmp`, so `/var/tmp/x` becomes `/var$TMPDIR/x`. The rule gains a leading boundary
  (`(?<![\w-])`), so a directory is rewritten only where it begins a path. Pin: one line holding
  `/tmp/a` and `/var/tmp/b` reads `$TMPDIR/a` and `/var/tmp/b`; the home rule is checked the
  same way.
- **The two spread labels.** The health table's column becomes "Repeat spread (range / median)"
  and the smoke-test table's "Spread (range / min)". The JSON keys do not change. Pin: the
  headings.
- **Default-beta live plans counted separately.** Each version's live-plan signal counts episodes
  that planned beyond support; the out-of-support table gains the count among episodes at the
  sweep's default beta, the only ones Eq. (1) scores, as "n (m at default beta)". Pin: an
  instrumentation file with episodes at two betas, hand-counted; a workdir without one still reads
  "not measured".

GAPS: the "Live-plan signal in replay" row names the default-beta count; the evidence paragraph in
`.claude/CLAUDE.md` and `report.md`'s redaction note need no change (the rule's scope is the
same). `ci.yml` moves to the measured count.

## 7. The run

From `reconstruction/` in the venv, after the smoke test, only after the owner's go (it spends
budget):

```
python scripts/run_dream_rsi.py --simpletes ../SimpleTES --task lasso_path \
    --workdir /srv/dream-rsi-runs/d2c-lasso \
    --discovery-agent "$(cat discovery.json)" --policy-agent "$(cat policy.json)" \
    --iterations 2 --versions 3 --workers 4 --grid 4 3 --hard-max 6 4 \
    --agent-timeout 900 --eval-repeats 3
```

- Launched detached: `setsid` under `nohup`, in a session of its own, with a log file, so ending
  the controlling session cannot take the run with it. `discovery.json` and `policy.json` are
  the argv of §4 and are not committed.
- Everything else default: objective `pareto`, the paper's beta grid and lambda, no direction
  provider, the real Listing 1 and 2 prompts from `generated/`, evaluations serialized; SimpleTES's
  own 600 s evaluator timeout stays.
- The task tree (`task/`, the baseline included) is hashed after launch and re-checked before the
  report, since the discovery agent's file tools can write anywhere in the workdir.
- Expected wall clock: D2a's spec estimated one to two hours for two iterations at k = 1; three
  evaluations per attempt add evaluator time only (tens of seconds per Lasso evaluation).
- **Failure and restart.** An interrupt or crash freezes the partial iteration under `runs/`.
  One restart is allowed: the directories the guard names (`runs/iterNNNN/`,
  `trace_pool/iterNNNN/`, `policy_dev/history/r*_tNN_m*/`, `instrumentation/r*_tNN_m*/`) are
  moved aside to `/srv/dream-rsi-runs/d2c-lasso-restart1/` with `task/`, `launches.jsonl` and
  `state.json` copied beside them, and the run is relaunched; that iteration's calls are spent
  again. A second failure ends the experiment with what exists; a failing run is evidence too. A
  quota or billing error shows up as failed attempts with their stderr tails; the run is not
  retried into it.

## 8. Report, evidence, verdicts

- **Report.** `python scripts/report_run.py --workdir /srv/dream-rsi-runs/d2c-lasso --out
  evidence/d2c-lasso --copy-evidence`, run on this host, since redaction knows only this host's
  home and temp paths. A restart's aside directory is reported as a workdir of its own under
  `evidence/d2c-lasso/restart1/`, as D2a's was.
- **Scan.** Before the commit, D2a's scan: pasted programs, compiler-quoted source lines, home
  paths, any `@` address, files over 500 KB. In addition, every `agent_stderr` value in the copied
  `score.json` files is read by eye: it can carry program text or account names no rule matches,
  and the service account's address embeds the project id. What to redact is the owner's call;
  the result is recorded in the spec's amendments.
- **Ledger.** GAPS §5 gains a "D2c run" paragraph, dated the day of the run, in the form of the D2a one: host,
  CLI versions, model, shape, what completed, the headline counts and the noise measured with
  k = 3. The README status row for real-agent runs names the run. The workdir archive's sha256
  is recorded; the archive stays off the repository.
- **Verdicts.** For each of the three items, the report's section decides "observed: change it"
  or "not observed: leave it, and say why", grounded in the counts, and the GAPS row's verdict
  clause is rewritten from "D2c revisits" to what D2c showed:
  - per-round call budget: the largest plan any version made, live or as its next live plan,
    against the caps;
  - zero versus clip: the live-plan and next-live-plan flags, and whether a flagged version was
    deployed;
  - the empty batch: any version scored −∞ for one, any live batch left empty.

  A "change it" verdict becomes a dated hand-forward item in this spec's amendments, with the
  count behind it; the change itself is its own track.

## 9. Tests

TDD applies to §6 only: each pin fails on the tree before its change, and after GREEN each is
mutation-checked against its most plausible regression (the boundary dropped; a label reverted;
the default-beta count taken over all episodes). Everything else in this spec is controller-run
and leaves its evidence in `evidence/d2c-lasso/`.

The CI pin moves once, in the §6 task, to the count measured with `NO_COLOR=1`.

## 10. Success criteria

1. The gate passes on the final tree with zero skips, and the CI pin equals the measured count.
2. Both iterations completed, or the run ended under §7's rule with its partial iteration
   reported.
3. `evidence/d2c-lasso/` holds `report.md`, `report.json`, `host.json` and the redacted evidence
   subset, with no attempt program, no quoted source line, no home path and no address, and
   `evidence/d2a-lasso/` is unchanged.
4. GAPS §5 has the run paragraph, the three verdict rows no longer say "D2c revisits", and the
   README status row names the run.
5. The Gemini CLI version, the model id, the region and the service account's existence are
   recorded in the launch record and the GAPS paragraph; the key file and the agent argv files
   are not in the repository.

## 11. Task order

1. The spike (§5): controller-run, no spend; four stops for the owner's go.
2. The three report items (§6), with their pins and the CI pin.
3. Eigen, the workdir, and the smoke test (§3): controller-run; `apt` and `/srv` need sudo.
4. The run (§7): controller-run; owner's go; spends budget.
5. Report, scan, evidence commit, GAPS §5, README (§8): controller-run; the owner's redaction
   decision in the middle.
6. The verdicts (§8) in their GAPS rows, and this spec's amendments.

Tasks 2 and 6 go through subagent-driven execution with a task review each; the rest are run by
the session itself, never by a subagent, because each has the owner's go or decision in the
middle. A whole-branch review at the end, then the finishing menu.

## 12. Open questions

None that block the plan. Whether the Gemini CLI's Vertex path accepts `-m gemini-3.1-pro-preview`
or needs another id is settled by the spike's first call, not by this spec.

## 13. Hand-forward

Filled in as the run shows what the next track needs. Standing items from earlier tracks that
D2c does not touch: the trace-pool tamper window during `offline()` and the discovery agent's view
of sibling attempts (GAPS §7).
