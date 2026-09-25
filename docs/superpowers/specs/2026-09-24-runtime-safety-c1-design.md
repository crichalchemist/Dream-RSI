# Runtime safety, part 1 (track C1): interruption and child-process lifetime

Date: 2026-09-24. Branch: `spec/runtime-safety-c1`, from `main` at e56fb5a (track B merged).
Owner decisions are recorded inline as **Decision**.

## 1. Context

The maturity audit's track C ("runtime safety") is the largest block of code change it proposes:
"a multi-day real run survives interruption, bad evaluators, and is auditable". Track B closed four
of its items (the `runs/iterNNNN` guard, evaluator-result validation, the deploy digest, the
recovery docs) and its final review handed forward three more that it reproduced: the archive-name
reuse after a crash in `offline()`, the sweep subprocess outliving a killed parent, and the
never-read `deployed_sha256`.

**Decision (scope).** Track C is cut into three sub-tracks, each its own spec, plan and SDD cycle:

- **C1 (this spec)** — interruption and child-process lifetime.
- **C2** — batch resilience: a bad cell never drops its completed siblings (`as_completed`, a
  per-attempt bookkeeping guard), the `Trace` is built before the pool directory is created, the
  live `plan_grid` call is wrapped, a circuit breaker on an all-`no_program` batch, and the
  evaluator-in-a-subprocess design.
- **C3** — atomic and auditable state: tmp-then-`os.replace` writes with `state.json.bak`, refusal
  to deploy an all-invalid winner, the load-time `deployed_sha256` check, the sweep-vs-archive
  digest comparison, a `trace.json` digest in the manifest, write-protected frozen directories.

What today's code does on an interrupt (`see/loop.py:online`, `see/live.py`): SIGINT lands in the
main thread while `LiveQuestion._execute` waits on `pool.map`; the `with ThreadPoolExecutor` exit
then blocks until every in-flight agent returns (hours for a real agent), nothing is frozen, the
agent CLIs' own children are never killed, `runs/iterNNNN` is left as raw attempt directories, and
`trace_pool/` is untouched. SIGTERM is not handled at all: Python exits at once, so not even the
raw directories are consistent. `CommandAgent` uses `subprocess.run(timeout=...)`, which kills the
CLI process but not what it forked. `_sweep` uses `subprocess.run` with no process group, so a
killed parent leaves the replay subprocess writing into `policy_dev/history/`. `_archive`
increments `state["round"]` in memory and persists it only on success, so a restart after a crash
in `offline()` recreates and overwrites the aborted iteration's archive directories.

## 2. Goals, non-goals, constraints

Goals:

1. Ctrl-C and SIGTERM stop a run promptly: queued attempts never start, running agents and
   everything they forked are killed, and the process exits.
2. What was collected before the interrupt is preserved as a readable trace, without ever entering
   the training pool.
3. An agent timeout kills the agent's whole process group and is recorded.
4. The replay subprocess cannot outlive `_sweep`.
5. A restart after a crash in `offline()` is refused before it can overwrite archives.

Non-goals: resuming an interrupted iteration (track B's decision stands: refuse loudly, delete, do
not merge); everything listed under C2 and C3 above; killing in-process evaluators (C2's
evaluator-subprocess design); Windows (the CI matrix is ubuntu and macos; `os.killpg` and
`start_new_session` are POSIX); surviving SIGKILL of the parent.

Constraints carried from tracks A and B: the gate stays strict (`ruff format --check .`,
`ruff check .`, `pyright` basic with 0 errors, `python tools/extract_listings.py --check`,
`python -m pytest -q` with zero skips, `tools/check_junit.py --expect N`); no `xfail`, `skip`,
`# noqa`, `# type: ignore`; nothing public in `see/policy/api.py` or
`see/policy/observation_signal.py` changes and no new `fail_class` string is introduced; no workdir
directory is renamed; the agent callable's signature `agent(prompt, *, cwd, target) -> dict` is
unchanged; every paper-silent choice gets a `GAPS.md` §3 row; one commit per task, plain imperative
messages, no attribution trailers; fixes land with their tests.

**Decision (validation).** Tests use the toy loop plus *shell stand-ins* for child processes (an
`sh -c` script that forks a `sleep`, with sub-second timeouts); never a real agent CLI or an API.
The suite runtime target is raised from track B's five seconds to about eight.

## 3. Components

Five responsibilities, each in the file where the process already lives. No new module, no
constructor signature changes, no change to the policy contract.

| Where | What changes |
|---|---|
| `see/live.py`, `CommandAgent` | `Popen(start_new_session=True)`; a locked registry of live children; process-group kill on timeout with `timed_out: True` in the result; new `terminate()`; new constructor field `kill_grace: float = 5.0`. |
| `see/live.py`, module level | `kill_process_group(p, grace)`: SIGTERM, wait `grace`, SIGKILL, wait; tolerant of an already-dead group. |
| `see/live.py`, `LiveQuestion` | `_interrupted: threading.Event`; `_execute` submits futures and, on any `BaseException` in the main thread, sets the event, cancels queued futures, calls `agent.terminate()` when the agent has one, and re-raises; `_run_attempt` raises before creating anything when the event is set. `eval/score.json` gains `agent_timed_out`. |
| `see/loop.py`, `DreamRSI.online` | a second `except BaseException` branch freezes the partial tree under `runs/iterNNNN/partial/` and re-raises; the manifest-building code is shared by both paths. The guard also checks the first archive name and names every existing path. |
| `see/loop.py`, `DreamRSI._sweep` | `Popen(start_new_session=True)` + `communicate(timeout=sweep_timeout)`; timeout kills the group and scores the version invalid as today; `finally` kills a still-live group. |
| `see/loop.py`, module level | `archive_name(round, t, m)` used by `_archive` and the guard; `install_signal_handlers()` mapping SIGTERM to `KeyboardInterrupt`. `LoopConfig` gains `kill_grace: float = 5.0` (used by `_sweep`). |
| `see/__main__.py:cmd_demo`, `scripts/run_dream_rsi.py:main` | call `install_signal_handlers()` first. |

## 4. The interrupt path

1. SIGINT, SIGTERM (mapped by §6) or a `BaseException` escaping the policy surfaces in the main
   thread, which is inside `LiveQuestion._execute` waiting on the batch's futures.
2. `_execute` sets `_interrupted`, calls `pool.shutdown(wait=False, cancel_futures=True)` so
   queued attempts never start, and calls `terminate()` on the agent if it has one. A worker that
   had not yet launched its agent sees the event at the top of `_run_attempt` and raises without
   creating its attempt directory; an agent launched in the race window is refused by the closed
   `CommandAgent`. Killed children return promptly to their workers. The `with` block's own
   `shutdown(wait=True)` then completes quickly, bounded only by in-process evaluations already
   running. Cells from the interrupted batch are never recorded: `_execute` writes `self.cells`
   only after every future has returned normally.
3. The exception propagates through `Question.probe_batch` and `policy.solve` into `online()`,
   whose `except BaseException` branch writes `runs/iterNNNN/partial/trace.json`,
   `live_episode.jsonl` and `live_cycle_manifest.json` from the cells completed before the
   interrupt, with the manifest's `error` set to `f"{type(e).__name__}: {e}"` and `"partial": True`,
   then re-raises. The trace id is `iterNNNN-partial`. Those cells form contiguous chains by
   construction (an attempt runs only after its parent was recorded), so the `Trace` validator
   accepts them; an interrupt before any batch freezes an empty trace. `trace_pool/` is never
   touched on this path, and `see.pool.load_pool` only globs `trace_pool/iter*`, so replay scoring
   never sees a truncated tree.
4. The process exits. On restart, `online()`'s guard refuses, naming every directory that exists
   among `runs/iterNNNN`, `trace_pool/iterNNNN` and `policy_dev/history/r{round+1:04d}_tNN_m0`.
   Recovery is the manual delete track B documented; the collected work is now a trace that can be
   inspected or pooled by hand.

**Decision (interrupt semantics).** Freeze into `runs/`, never into the pool; killed cells are
discarded rather than recorded with a marker (no new `fail_class`).

## 5. Child-process lifetime

`CommandAgent.__call__`:

- Spawns `Popen(argv, cwd=cwd, stdout=PIPE, stderr=PIPE, text=True, env=..., start_new_session=True)`
  and registers the child; unregisters in `finally`.
- `communicate(timeout=self.timeout)`. On `TimeoutExpired`: `kill_process_group(p, self.kill_grace)`,
  drain output, return `{"returncode": None, "stdout": "", "stderr": f"agent timed out after {self.timeout}s", "timed_out": True}`.
- A closed agent returns `{"returncode": None, "stdout": "", "stderr": "agent terminated"}` without
  spawning.

`CommandAgent.terminate()`: under the lock, snapshot the live children, kill each group, mark the
agent closed. Idempotent. Toy agents (`see/toy.py`) have no `terminate`; `_execute` uses
`getattr(self.agent, "terminate", None)` and calls it only when callable.

`kill_process_group(p, grace)`: `os.killpg(p.pid, SIGTERM)`, `p.wait(grace)`; if still alive,
`os.killpg(p.pid, SIGKILL)`, `p.wait()`. `ProcessLookupError` means the group is already gone.

`_run_attempt` writes `"agent_timed_out": bool(run.get("timed_out"))` into `eval/score.json` beside
`agent_returncode`. The cell's error text is unchanged (`agent left no program (agent timed out …)`),
so `classify_failure` and the fail classes are untouched.

`DreamRSI._sweep`: `Popen(cmd, cwd=PKG_ROOT, stdout=PIPE, stderr=PIPE, text=True, start_new_session=True)`;
`communicate(timeout=self.c.sweep_timeout)`; on `TimeoutExpired` kill the group with
`self.c.kill_grace` and raise `RuntimeError(f"sweep timed out after {self.c.sweep_timeout}s")`
into the existing `except Exception` branch, which writes the invalid report as today; `finally`:
if `p.poll() is None`, kill the group. An interrupt arriving mid-sweep therefore kills the replay
subprocess and propagates; `offline()` persists nothing, and §6's guard handles the restart.

## 6. Archive guard and SIGTERM parity

- `archive_name(round: int, t: int, m: int) -> str` returns `f"r{round:04d}_t{t:02d}_m{m}"`;
  `_archive` uses it.
- `online()`'s guard candidates are `runs/iterNNNN`, `trace_pool/iterNNNN` and
  `os.path.join(self.dev_history, archive_name(self.state["round"] + 1, t, 0))`. It raises once
  with every existing candidate: singular wording for one path (`{path} exists: iteration {t} was
  interrupted or already ran; delete it, do not merge into it`), plural for several (`{a} and {b}
  exist: …; delete them, do not merge into them`). Track B's tests match the singular form.
- `install_signal_handlers()` installs a SIGTERM handler that raises
  `KeyboardInterrupt(f"signal {signum}")`. Called first thing by `cmd_demo` and by
  `scripts/run_dream_rsi.py:main`; never on import, never by tests (they raise the stand-in
  exception directly).

## 7. Edge cases and limitations

- Interrupt during the baseline evaluation or planning: nothing collected; an empty partial trace
  with `error` set is written.
- Interrupt while an in-process evaluator runs: it cannot be killed; the interrupt returns after
  in-flight evaluations finish (about 20 s for Lasso). Limitation, recorded in GAPS; C2 owns it.
- An agent that ignores SIGTERM dies at SIGKILL after `kill_grace`. A child that calls `setsid`
  itself leaves the group and escapes; documented, not solved.
- A second Ctrl-C during the freeze may leave `partial/` half-written; the guard still refuses.
- `Question.probe_batch` advances its round counters before `_execute`, so a partial manifest's
  `decision_rounds` counts the interrupted round. Acceptable; the manifest says `partial`.
- A terminated `CommandAgent` never spawns again; a new run constructs a new agent.
- SIGKILL of the parent runs none of this; the CLAUDE.md recovery bullet keeps one clause about
  checking for a surviving `see sweep` in that case only.

## 8. Tests (55 → 62 collected; suite about 8 s)

Shell stand-in: `["sh", "-c", "sleep 30 & echo $! > \"$0/child-$$.pid\"; wait", "<dir>", "{prompt}"]`
— forks a `sleep`, records the grandchild's pid, waits. The test reads the pid file and asserts the
grandchild is gone with `os.kill(pid, 0)` raising `ProcessLookupError`, polling briefly for reaping.
Timeouts 0.2–0.5 s, `kill_grace` 0.2 s.

| # | File | Test (name states the claim) |
|---|---|---|
| 1 | `tests/test_command_agent.py` | `test_agent_timeout_kills_the_whole_process_group` — timeout 0.3 s; result `timed_out` True, `returncode` None; grandchild dead. |
| 2 | `tests/test_command_agent.py` | `test_terminate_kills_live_agents_and_refuses_new_calls` — a call sleeping in a thread returns within 2 s after `terminate()`; grandchild dead; a later call returns `agent terminated` and writes no pid file. |
| 3 | `tests/test_loop.py` | `test_interrupt_freezes_the_partial_tree_under_runs_not_the_pool` — `_RecordingAgent(interrupt_on=2)`; after `online(1)` raises, `runs/iter0001/partial/trace.json` loads as a `Trace` with exactly the cell completed before the interrupt, the manifest's `error` names `_Interrupted` and `partial` is true, `live_episode.jsonl` exists, `trace_pool/iter0001` is absent, `state.json` is unchanged. |
| 4 | `tests/test_live_interrupt.py` | `test_interrupt_cancels_queued_attempts_and_kills_running_agents` — `LiveQuestion` with two workers over a `(4, 0)` grid; an agent wrapper whose first call runs the shell stand-in and whose second raises the stand-in exception, delegating `terminate()` to the inner `CommandAgent`; `probe_batch` raises within 2 s, the grandchild is dead, calls 3 and 4 never happened, no attempt directory exists for them, `q.cells` is empty. |
| 5 | `tests/test_loop.py` | `test_sweep_timeout_kills_the_subprocess_group_and_scores_invalid` — a `method.py` that forks a `sleep` on import and records its pid, `sweep_timeout` 0.5 s; `_sweep` returns an invalid report whose error mentions the timeout; grandchild dead. |
| 6 | `tests/test_loop.py` | `test_restart_refuses_when_the_first_archive_dir_exists` — one toy iteration (`versions=1`), delete `runs/iter0001` and `trace_pool/iter0001`, leave `policy_dev/history/r0001_t01_m0`; a fresh `DreamRSI` over the workdir raises naming that directory; with `runs/iter0001` re-created the message names both. |
| 7 | `tests/test_loop.py` | `test_sigterm_takes_the_same_path_as_ctrl_c` — `install_signal_handlers()`, `os.kill(os.getpid(), SIGTERM)` inside `pytest.raises(KeyboardInterrupt)`; the previous handler is restored in `finally`. |

Each new test is shown red before its fix (for 1, 2, 5 the old code leaves the grandchild alive;
for 3 no `partial/` exists; for 4 `probe_batch` blocks for the full sleep or the old code lacks
`terminate`; for 6 the old guard does not raise; for 7 the default SIGTERM disposition kills the
process, so the red run is the helper being absent).

## 9. Ledger and docs

- `GAPS.md` §3: three rows — interrupt semantics (partial frozen under `runs/`, never pooled,
  killed cells discarded); process-group semantics (POSIX only; `setsid` escape; in-process
  evaluators cannot be interrupted; `agent_timed_out` recorded); the archive-name guard (why the
  first archive name, and that the round counter is persisted only on success).
- `.claude/CLAUDE.md`: the interrupted-iteration bullet becomes: refuse on `runs/`, `trace_pool/`
  or the first archive dir; the partial trace is under `runs/iterNNNN/partial/`; delete the three
  before restarting; after a SIGKILL of the parent (only), check for a surviving `see sweep`.
- `reconstruction/README.md`: status table rows for interruption handling and agent timeouts.
- `.github/workflows/ci.yml`: `--expect 62`. The count is quoted nowhere else.

## 10. Verification

1. The full gate from `reconstruction/` on the final tree, 62 passed, no skips, under about 8 s.
2. `python -m see demo --workdir <tmp>` exits 0.
3. Manual interrupt check on the demo: start the demo, send SIGTERM within the first iteration
   (a wrapper script, since the toy loop is fast), observe `runs/iter0001/partial/` with a manifest
   whose `error` names `KeyboardInterrupt`, an empty `trace_pool/`, and a restart that refuses
   naming `runs/iter0001`; after deleting it, the demo completes.
4. Manual orphan check: start the demo, SIGTERM it during `offline`, and confirm no `see sweep`
   process survives (`pgrep -f "see sweep"` prints nothing).

## 11. Sequencing (one commit per task; count after each)

1. `CommandAgent`: session, registry, group kill on timeout, `timed_out`, `agent_timed_out` (56).
2. `CommandAgent.terminate()` and the closed state (57).
3. `LiveQuestion._execute` cancellation and `_interrupted` (58).
4. `DreamRSI.online` partial freeze and shared manifest builder (59).
5. `DreamRSI._sweep` session and kill; `LoopConfig.kill_grace` (60).
6. `archive_name` and the three-candidate guard (61).
7. `install_signal_handlers()` and the two entry points (62).
8. GAPS rows, CLAUDE.md, README, CI pin 62.

## 12. Deferred, surfaced by this design

- C2: `as_completed` with never-dropped siblings; per-attempt bookkeeping guard; `Trace` before the
  pool dir; wrap the live `plan_grid`; all-`no_program` circuit breaker; evaluator subprocess with a
  timeout (which would also make interrupts kill evaluations).
- C3: atomic writes and `state.json.bak`; refuse an all-invalid winner; load-time
  `deployed_sha256` check; sweep-vs-archive digest comparison; `trace.json` digest; write-protect
  frozen directories.
- Elsewhere: resumability (owner decision: not planned); Windows support; static checks on LLM
  policy code; scrubbing the environment passed to agents; relative history symlinks.
