> Provenance: produced 2026-09-24 by an 8-agent spike workflow (schema mapping → three adapter lenses prototyped over a 40-session sample of the owner's own Claude Code transcripts → three judges → synthesis). Prototypes are throwaway and were not kept. Structure and counts only; no transcript content.

# Trace-adapter spike: can Claude Code transcripts become Dream-RSI replay worlds?

Report for the owner. Inputs: the structure pass (schema and counts), three lens prototypes
(tool-run, goal-attempt, evaluator-backed), and three independent judgements. All numbers below
are measured on the owner's own transcripts; only schema, counts, rates and structural
descriptions appear here. No prose, paths, slugs, names or command text from any transcript.

## 1. Answer

**Conditional, and as the data stands today the honest reading is no.** The single most important
reason is structural, not a defect of any lens: across 106,701 records in the 40-session sample the
transcripts contain zero genuine alternative-attempt forks, so every adapter has to fabricate
branches out of one linear trajectory on one shared, mutable workspace, and under the paper's
success predicate the evaluator-derived score collapses to a single value (1.0) among successes,
so the stock ParallelRefine policy hits the trace ceiling in its first round on 27/29 non-degenerate
population traces and replay cannot rank policies. The answer becomes yes only if the P3 plugin
*records* trees (W parallel workspaces from one root, one pinned evaluator, logged termination)
rather than *mines* them from history, and gates every trace on having a replay slope before it
enters the pool.

## 2. What the transcript data actually is

Sample used everywhere below unless stated: transcript paths sorted by basename (stable by size);
10 smallest (222 B to 36.7 KB), 20 nearest the median size (415.9 to 469.4 KB, median 436.5 KB),
10 largest (19.2 to 44.2 MB) = 40 sessions, 106,701 records, 0 bad JSON lines. Nested subagent
files of those sessions were included by the tool-run and evaluator-backed lenses (266 files,
53,220 records in the 5 sessions with the most nested files). The evaluator-backed lens also ran a
population pass over all 255 top-level files.

### Files and record types

- `~/.claude/projects/<slug>/<session-uuid>.jsonl`: 255 top-level sessions in 129 projects.
  Nested: `<slug>/<session-uuid>/subagents/agent-<agentId>.jsonl` (963 hex-named, 43 word-named)
  and `subagents/workflows/wf_<id>/` (agent files plus one workflow log with record types
  `launched`/`started`/`result` and keys `agentId, key, label, phase, result`).
- 21 record `type` values. Graph-bearing: `user`, `assistant`, `system`, `attachment` (carry
  `uuid`, `parentUuid`, `isSidechain`, `timestamp`, `sessionId`, `cwd`, `gitBranch`, `version`,
  `entrypoint` in {cli, sdk-cli}). Non-graph metadata rows keyed by `sessionId` and rewritten
  repeatedly: `last-prompt`, `mode`, `permission-mode`, `ai-title`, `custom-title`, `pr-link`,
  `queue-operation`, `bridge-session`, `relocated`, `worktree-state`, `cost-state`
  (`totalCostUSD, totalDuration, totalToolDuration, totalLinesAdded, totalLinesRemoved`),
  `continued-in`.
- Population by entrypoint: `cli` 136 (7 <50 KB, 23 50-500 KB, 60 0.5-5 MB, 46 >5 MB), `sdk-cli`
  116 (3, 108, 5, 0), other 3. 84 sessions are cli with >=2 prompts and >=10 tool uses; 51 have
  nested subagent dirs (1,028 files). **All 20 median-bucket sample sessions are sdk-cli one-shots
  (1 prompt, no `turn_duration`, no `cost-state`); all 10 large ones are interactive cli.** Size
  stratification therefore under-represents the only stratum with material.

### The three kinds of user record

1. Tool result: `message.content` is a list with exactly one `tool_result{tool_use_id, content,
   is_error?}` plus top-level `toolUseResult` (tool-specific dict) and `sourceToolAssistantUUID`.
2. Human prompt: content is a string or text/image blocks, with `promptId` and optional
   `promptSource` / `origin.kind` in {human, task-notification, peer}.
3. Meta: `isMeta: true`, or `isCompactSummary: true` (post-compaction summary).

### Assistant, system, attachment

- Assistant: one record per API content block (thinking / text / tool_use / ...), `message.id`
  shared across the blocks of one API response, `apiBlockIndex` [inferred: block index],
  `message.stop_reason` in {tool_use, end_turn, stop_sequence, null, max_tokens}, `message.usage`.
- System `subtype`: `stop_hook_summary` (`hookErrors` empty in 831/831), `turn_duration`
  (`durationMs`, `messageCount`), `compact_boundary` (`parentUuid` null, `logicalParentUuid`,
  `compactMetadata.trigger` manual|auto), `local_command`, `informational`, `away_summary`.
- Attachment: 41 `attachment.type` values; attachments are graph nodes and can sit between two
  assistant blocks of one message.

### Tool pairing (verified 13,120/13,120)

`tool_result.tool_use_id == tool_use.id`; the result record's `parentUuid ==
sourceToolAssistantUUID ==` the uuid of the assistant record holding the `tool_use`; exactly one
`tool_result` per user record; 1 tool use without a result in the sample. `sourceToolUseID` is
NOT the pairing key (29 records, all `user` + `isMeta`). `is_error` is present on 9,649/9,649 Bash
results (true on 282) [inferred: non-zero exit]. `toolUseResult` shapes that matter: Bash
`{stdout, stderr, interrupted, gitOperation{commit|push|pr}, backgroundTaskId}`; Edit/Write
`{filePath, structuredPatch, userModified}`; Agent `{agentId, status in
async_launched|forked|completed, isAsync}`; AskUserQuestion `{questions, answers}`.

### The parentUuid graph is a chain, not a tree

- `parentUuid == preceding linked record` in 66,914/71,014 (94.2%); timestamps are not monotonic
  (2,419 inversions) so file order is the reliable linearization.
- Raw records with >1 child: 3,788 (5.32%). After dropping attachment children and merging
  same-`message.id` assistant siblings: 561 (0.79%) = 553 parallel-tool-call ladders (a width-W
  batch laid out A1->{A2, U1}, A2->U2 with U1 a leaf), 4 prompt/prompt forks (all abandoned stubs
  of 1-3 records against 183-257), 4 misc. **True alternative-attempt forks: 0.**
- Multiple roots per file come from `compact_boundary` (136 in 10/40 sessions; `logicalParentUuid`
  resolves in-file 95/136, the other 41 point to another file [inferred: resumed session]) and
  `hook_success` attachments (29).

### Subagents are the only real concurrency

`isSidechain` is false on 71,189/71,189 top-level linked records and true on 53,220/53,220 nested
records; every nested record carries `agentId == filename id`, `sessionId == parent session`
(265/266); the nested file's first record is a root user prompt [inferred: the Agent tool's
prompt]. Parent Agent `toolUseResult.agentId` resolves to a file in 204/204; 62/266 nested files
are referenced by no parent Agent result (35 word-named; likely teammates / workflow agents).
Up to 8 subagents (evaluator-backed measure) or 10 parallel tool_use blocks (tool-run measure) are
alive at once. Nested files have their own roots, compaction boundaries and parallel-call forks.

### Where a score could come from

| signal | field | availability in sample |
|---|---|---|
| pytest summary counts | Bash `toolUseResult.stdout` regex `(\d+) (passed|failed|error)` | 209 results in 7/40 sessions; 45% of 463 test commands; 100% of parseable outcomes are pytest |
| exit status | `tool_result.is_error` | 9,649/9,649 Bash results, true on 282; >=1 in 18/40 sessions |
| build/compile error | regex on stdout+stderr | 75 hits in 10/40; success phrase 36 |
| npm/vitest/jest | 256 commands | 0 `Tests:` summary lines; only PASS/FAIL tokens (144); 14 run in background and return later as task notifications |
| `classify_failure` (see.policy.observation_signal) | Bash stderr | usable in 21/40 but 812/887 land in the default `code` class (rules written for kernel-compile errors) |
| edit -> test cycle | Edit/Write followed by a test command | 7/40 sessions, 3-48 cycles each |
| `cost-state` counters | cumulative per session | 14/40 |
| `turn_duration.durationMs` | per human turn | 744 records in 10/40 |
| Agent completion | `toolUseResult.status`, nested last `stop_reason` (end_turn 222/266) | 10/40 |
| AskUserQuestion answers | human judgement | 8/40 |
| git milestones | `gitOperation` commit 77 / push 106 / pr 45; `pr-link` | 7-10/40 |
| `stop_hook_summary.hookErrors` | | 0/831 non-empty: no signal |

Branching material: subagent files (10/40 sample, 51/255 population, up to 103 per session);
parallel tool_use batches (553); edit->test cycles on the main line (7/40); gitBranch/worktree
changes (4/40 sessions, 43 changes, 86 worktree commands, 13 EnterWorktree); compaction
boundaries; teammate coordination (SendMessage 157, origin.kind peer); session continuation
(`continued-in` 3, 41 out-of-file logicalParentUuids); task notifications (322).

The claude-remember stores (67, 31 MB) are compressed prose narratives with no scores and were
not used by any lens.

## 3. The three lenses side by side

### Mapping

| | tool-run | goal-attempt | evaluator-backed |
|---|---|---|---|
| probe / attempt | one tool_use/tool_result pair of a probe tool {Bash, Edit, Write, MultiEdit, NotebookEdit} | one human turn (turn-defining prompt and its span to the next such prompt) | one evaluator RUN: a Bash (or ctx_execute shell) command whose head segment is a test/build/lint tool, with a result, not background, not denied |
| branch | (scope, target): scope = main file or one nested agent file; target = normalized command text (`--key exact`) or program+subcommand (`--key head`), or edited file path | one human goal = maximal run of consecutive turns judged to be about the same thing (reaction-class, cwd, gitBranch, 20-min gap, token containment rules) | (file, gitBranch): main-line git-branch segments plus every nested subagent file |
| root / baseline | session start; baseline 0.0 | repository state at first prompt; baseline 0.5 (convention) | first main-line RUN before any mutating tool_use or Agent call (`root_eval`, 2/40 sample, 24/255 population) else floor 0.0 |
| score | parsed test counts -> pass fraction; other Bash 1.0, or 0.0 on is_error / strong failure marker; edits 1.0 or 0.0; denied 0.0 unevaluated | weighted mean H 0.6 (class of the NEXT prompt: corrective 0, interrupted 0, mixed 0.25, follow-up 0.5, slash/moved-on 0.75, approval 1) + T 0.25 (tests) + C 0.1 (clean end_turn) + M 0.05 (commit/push/pr) | pass fraction n_valid/n_total from pytest/unittest/jest/vitest/mocha/cargo/go summaries; binary 0/1 from is_error (+ build-error regex) for build/lint; no parse -> evaluated=False |
| fail_class / error | closed labels; `classify_failure` on stdout+stderr in memory | closed labels: corrective/rejected -> correctness, interrupted -> timeout, api_error -> env | synthesized `<evaluator>: F failed, E errors of N`; partial pass = repairable `correctness`; `classify_failure` for env/timeout/resource |
| W (max_parallelism) | max parallel tool_use blocks sharing one `message.id` (up to 10) | forced to 1 | 1 + max overlapping nested-file timestamp intervals (up to 8) [inferred] |
| termination | last probe on its target; tags.term in {last_ok, last_fail, last_denied, no_result} | next prompt starts a new goal or session ends | last RUN on the key; end in {session_end, sidechain_end, env_fail} |

### Coverage (sample of 40 unless noted)

| | tool-run | goal-attempt | evaluator-backed |
|---|---|---|---|
| parsed / Trace validation / round trip | 40/40 / 40/40 / 40/40 | 40/40 / 40/40 / 40/40 (+ ReplayQuestion, `python -m see sweep`) | 40/40 / 40/40 / 40/40 (255/255 population) |
| sessions with >=1 cell | 21/40 (0 small, 11 median with 1-7 probes, 10 large) | 37/40 (27 single-branch) | 7/40 (0 small, 0 median, 7 large) |
| non-degenerate (>=2 branches and a chain >=2) | 10/40 exact key, 15/40 head key | 10/40 | 5/40; population 29/255 (11.4%) |
| cells | 22,919 | 658 | 1,670 (population 4,996) |
| score availability | 99.96% | 94.5% (informative human verdicts 16.7%, test evidence 17.6%) | 94.8% (population 92.4%) |
| failure cells | 5.6% | 7.8% | 17% (population: ok 1,366 of 1,670) |
| chain shape | 95.8% singleton branches (exact); longest chain 151 = one command re-run | chain length median 1, q3 2, max 11 | 148/169 chains >=2; max 30, median 7 in the largest trace |
| where the material sits | the >5 MB interactive cli stratum | the same 10 large cli sessions | large interactive cli sessions with subagents (~30-50 in the population) |

### Score validity (what optimizing the score optimizes)

| | tool-run | goal-attempt | evaluator-backed |
|---|---|---|---|
| distinct successful score values per non-degenerate trace | 1 (all 1.0); 94.4% of cells at ceiling | median ~9.5 (judge 3); 4-12 (judge 2) | 1 (all 1.0); 61% of branches green at attempt 0 |
| ParallelRefine replay | attainment 1.0 on 10/10 at 13.6 probes; beta sweep flat; random reaches ceiling in 1.0-1.4 probes | ceiling in round 1 on only 1/10 traces; median 15 probes to ceiling vs random 22.7 | attainment 1.0 on 29/29; ceiling in round 1 on 27/29; 9/29 have ceiling == baseline |
| what is really measured | "any tool call exited 0", edits flat 1.0 | "the next human prompt was not a short negative reply"; moved-on (0.75) = 365/658 cells conflates acceptance with abandonment | time-to-first-green, a step function; 68% of adjacent counted pairs change n_total (moving denominator); 34% switch evaluator kind |
| gaming (as a live reward) | read-only commands; never re-run a failure; narrow tests; mask exit status | stop early and ask a question; split turns; skip tests; vocabulary attacks | prefer lint/tsc cells; narrow selections; sidechains (roots green 61%); never refine |

### Judge scores (1-5; three judges, in input order)

| dimension | tool-run | goal-attempt | evaluator-backed |
|---|---|---|---|
| fidelity to Trace contract | 2 / 2 / 2 | 1 / 2 / 2 | 3 / 4 / 4 |
| score validity | 1 / 1 / 1 | 2 / 2 / 2 | 2 / 2 / 2 |
| coverage | 3 / 4 / 4 | 3 / 3 / 3 | 2 / 2 / 2 |
| implementation cost (5 = cheapest) | 5 / 5 / 4 | 3 / 4 / 3 | 4 / 4 / 4 |
| privacy safety | 4 / 4 / 4 | 3 / 3 / 3 | 5 / 4 / 5 |
| overall verdict | conditional | conditional | conditional |

All three judges pick **evaluator-backed** as best lens and all three give an overall
"conditional" that reads, in their own words, as "as specified, none produces worlds worth
dreaming over". Points of agreement: the tree half of every mapping is real and cheap; the scalar
half is not yet a score; goal-attempt is the only lens with a replay slope but the slope is the
classifier's own vocabulary; tool-run is the best PostToolUse substrate but semantically empty.
One disagreement worth noting: judge 1 rates goal-attempt's fidelity 1 because a cell's score is
read from the *next* prompt (it discloses the following cell on reveal, and chain depth counts
corrections, inverting the paper's refinement semantics); judges 2 and 3 give it 2 for the same
facts.

## 4. Recommendation

**Lens: evaluator-backed as the substrate, with two things borrowed.** From tool-run take only the
pairing / scope / seq plumbing (it is the same code path) and the width-W-from-`message.id`
measurement. From goal-attempt take only the hard human signals, `toolDenialKind ==
"user-rejected"` and `interruptedMessageId`, as failure labels on the enclosing branch; never the
prose classifier, never the H component.

**Exact score definition** (all evaluator-backed cells; the changes are the judges' conditions):

- `score = n_valid / n_total_ref` where `n_valid` = passed count of a *counted* evaluator run and
  `n_total_ref` is a per-trace fixed denominator: the largest `n_total` observed for that evaluator
  invocation shape in the trace (the "widest suite seen"). A run with a narrower selection therefore
  cannot score higher than the widest green run. `n_total` and `n_valid` are stored on the cell
  exactly as `see.synthetic` does.
- Partial pass (`failed + errors > 0`) keeps its fraction and is a repairable failure:
  `fail_class = correctness`, `error = "<evaluator>: F failed, E errors of N"` (synthesized, never
  transcript text), per the paper's Listing 2 semantics.
- Binary evaluators (lint, tsc, build, npm test without a summary) are **not** scored cells. They
  are recorded as evidence in `tags` (`lint_ok`, `build_ok`) and, when they fail, as
  `fail_class = compile_other` / `env` via `classify_failure`, so that a lint 1.0 can never equal a
  full green.
- Baseline = the counted evaluator run on the root workspace before any mutating tool_use or Agent
  call (`root_eval`). Traces with no such run get `baseline_score = 0.0` and `info.baseline_source
  = "floor"` and are flagged; today that is 231/255.
- Chains break on evaluator-kind switch and on a change of invocation shape; a chain is successive
  runs of the *same* shape on the *same* branch.

**Branch definition**: a branch is a workspace that started from the root state: a nested subagent
file (`isSidechain`, `agentId`) or a worktree/gitBranch segment whose first record follows a
`git worktree` / EnterWorktree from the root commit. Sequential main-line goals and per-command
chains are not alternatives and must not be replayed as such. Record each subagent's launch
position (the file position of its parent Agent tool_use) in tags so a branch whose attempt 0 was
inherited from a mutated main line is distinguishable from a true root start.

**Exclude**: sdk-cli one-shots (116/255, zero material under every lens); background runs
(`run_in_background`, `backgroundTaskId`); denied calls (`toolDenialKind`); runs whose output the
agent redirected or piped away (evaluated=False, never a 0); `--collect-only`; the 35 word-named
teammate files until their parent link is understood; anything whose only score would be
`is_error` on a non-evaluator command.

**Pool gate** (mechanical, run before any policy is trained): a trace enters the dream pool only if
(a) it has >=2 branches that start from the root state, (b) >=2 distinct successful score values,
and (c) ParallelRefine does not reach the ceiling in round 1 and beats a seeded random policy on
probes-to-ceiling. Today 0/29 non-degenerate population traces pass (a) strictly and 0/29 pass (c).

**Conditions under which P3 is worth building.** P3 is worth building only as a *recorder*, not a
miner. Specifically:

1. The plugin itself launches W parallel subagents or worktrees on the same task from the same
   root workspace, each ending with a pinned evaluator invocation (identical command shape, full
   suite, constant n_total within the trace); each workspace's successive runs of that command are
   its chain. This is the only way to obtain sibling branches; the transcripts have none.
2. The baseline run happens on the root before any edit (the `root_eval` path, today 24/255).
3. Branch termination is a logged decision written into tags ({stopped, budget, env}), because
   "moved on" and "session ended" are indistinguishable in transcripts.
4. Cell.error and tags stay closed labels and counts; no command text, error text or prompt text
   in traces; `classify_failure` labels when policies need a failure signal.
5. The pool gate above passes on at least ~20 traces before any dreaming loop runs. If (1) and
   the gate cannot both be met on ~20 traces, the honest outcome is no: the transcripts record one
   behaviour policy's single path per goal, and there is nothing for a meta-exploration policy to
   choose between.

Until then the transcripts are useful only as a coarse label source: evaluator-backed red-to-green
repair chains (213/514 population chains go red then green) and goal-attempt's hard reject /
interrupt fields (~3% of turns), not as replay worlds.

## 5. What P3 would minimally need

**Ingestion hook shape.** Two hooks, both stdlib Python, both emitting closed fields only:

- `PostToolUse` on `Bash` (and on the ctx_execute shell channel if kept): the hook already receives
  `tool_input.command` and `tool_response`, so only the head-of-segment evaluator classifier and
  the summary parsers (~150 lines of the evaluator-backed adapter) run per event. It appends one
  line `{session_id, agent_id?, seq, evaluator, parser, n_valid, n_total, is_error, fail_class,
  invocation_shape_hash, git_branch_index, mutations_since_last_eval}` to a per-session log.
  Unclassified command heads are logged as a count with a hashed head so the regex tail is
  visible without storing text.
- `SessionEnd` (and `SubagentStop`): assemble the per-session log plus its nested agent logs into
  one `Trace` via `see.world.Trace`, apply the pool gate, write the trace and the gate result.
  Optional: a `PreToolUse` on `Agent`/`EnterWorktree` that records launch position and root commit
  so branch-from-root can be asserted rather than inferred.

For the recorder mode (condition 1 in section 4) a third piece is a `/dream-branch` command or
skill that launches W subagents on the same prompt from the same root, runs the pinned evaluator
first (baseline) and pins the evaluator invocation shape in the session log.

**Storage under `CLAUDE_PLUGIN_DATA`.**

```
$CLAUDE_PLUGIN_DATA/
  sessions/<sha1(session-uuid)[:12]>.jsonl     per-event closed-field log (append-only)
  traces/<sha1(session-uuid)[:12]>.json        see.world.Trace via Trace.save (TRACE_FORMAT)
  pool/manifest.json                           trace id -> gate result {branches_from_root,
                                               distinct_success_scores, pr_round1_ceiling,
                                               beats_random, admitted}
  policies/<name>/{policy.json, eval.json}     learned policy + replay metrics
  unclassified_heads.json                      hashed head -> count (classifier upkeep)
```

Trace ids are hashes of the session uuid; no slug, cwd, gitBranch name or command text is stored
anywhere. The gate result is recomputed on every republish of the pool.

**Deployment and measurement.** The learned policy ships only as `SessionStart` guidance about the
tree-shaped decisions it actually controls: how many parallel branches to open, how many red
attempts on one branch before abandoning, when to stop. It is never wired in as a live reward.
Measurement, given the paper's own finding that injected guidance hurt (GAPS.md line 64):

- Hold out unguided sessions. Every future pool keeps a fixed fraction (say one in three) of
  sessions with no guidance injected, because replay favours policies near the behaviour policy
  (GAPS.md section 6) and a closed loop drifts toward whatever the guidance already made Claude do.
- A/B on the same evaluator: guided vs unguided sessions on matched tasks, comparing
  time-to-first-green (attempts and wall time from `turn_duration.durationMs`), final `n_valid` on
  the widest suite, and the fraction of branches abandoned after >=1 red attempt.
- Replay acceptance test before any deployment: `replay_check.py` over the admitted pool; the
  policy must beat seeded random on probes-to-ceiling and the beta sweep must not be flat.
- A kill switch: if guided sessions do not beat unguided on time-to-first-green after N sessions,
  stop injecting.

## 6. Open questions and the cheapest next experiment

Open questions:

1. Is there any redefinition of the scalar on *existing* transcripts that produces a slope? The
   widest-suite `n_valid` variant was checked by judge 1 and is flat in 0/29 traces; effort-to-green
   (attempts until first success) has not been measured as a score and is the last candidate.
2. How many sessions in the population truly have >=2 subagents launched from the same main-line
   state on the same task? The structure pass counted 0 forks in `parentUuid`; the sidechain
   launch-position comparison has not been run.
3. What are the 35 word-named nested files and the 62 unreferenced nested files (teammates, workflow
   agents)? They may be the closest thing to parallel siblings in the existing data.
4. Does rtk output rewriting (3 sessions in the sample, 226 population cells, ~1/3 of them losing
   the pytest summary) need a dedicated parser, or should rtk-prefixed runs be excluded?
5. Can the 168-command unclassified head tail be closed with the hashed-head count log, or does it
   need a different classifier design (e.g. reading `toolUseResult` for the executable, which no
   field currently names)?
6. Is `tool_result.is_error` really exit status != 0? It is inferred; a small instrumented check
   would settle it.

**Cheapest next experiment** (a day, read-only, no new hooks): on the 29 non-degenerate population
traces from the evaluator-backed adapter, (a) rescore with effort-to-green and with the pinned-shape
denominator, (b) re-run `replay_check.py` with ParallelRefine against a seeded random policy, (c)
compute subagent launch positions and count traces with >=2 sidechains launched from the same
main-line position. If (b) shows a slope on >=20 traces and (c) finds any, P3 as a miner has a
case. If not, the next experiment is the recorder: run 5-10 new sessions with `/dream-branch`
launching W=3 subagents on one task with a pinned pytest invocation and a root baseline, and check
whether those traces pass the pool gate. That is the decisive test.

## 7. Appendix

All prototypes are throwaway and live under
`/private/tmp/claude-501/-Users-controlroom-Dream-RSI/097e2e14-0174-4ac0-82cd-4c84a7f306c1/scratchpad/spike/`:

- `structure/notes.md`: schema and graph measurements (the source of section 2).
- `tool-run/adapter.py` (+ `notes.md`, `analyze.py`; outputs `traces/`, `traces_head/`,
  `pool_exact/`, `sweep_exact/`). Run from `/Users/controlroom/Dream-RSI/reconstruction`:
  `python3 .../adapter.py --sample [--key exact|head]`.
- `goal-attempt/adapter.py` (+ `notes.md`, `run_sample.py`, `explore1-3.py`; outputs `traces/`,
  `pool/iterNNNN/trace.json`, `sweep_out/`, `sweep_out.txt`, `cli_check.json`).
- `evaluator-backed/adapter.py` (+ `run_sample.py [--population]`, `analyze_traces.py`,
  `replay_check.py`, `diag_classifier.py`, `notes.md`; outputs `traces/sNN.json`,
  `traces/pNNN.json`).
- `judge/`: the three judgement passes.

Sample: transcript paths under `~/.claude/projects/*/*.jsonl` sorted by basename (stable by
size); 10 smallest (222 B to 36.7 KB), 20 nearest the median (415.9 to 469.4 KB), 10 largest
(19.2 to 44.2 MB); 40 sessions, 106,701 records. Nested files: all subagent files of those
sessions for tool-run and evaluator-backed (the structure pass measured 266 files / 53,220
records from the 5 sessions with the most nested files). The evaluator-backed adapter additionally
ran over all 255 top-level sessions with their nested files. Trace ids in prototype outputs are
sample ordinals or filename hashes.

Privacy check: every generated trace directory was grep-scanned for path, URL, e-mail and
filename fragments and for non-label strings in `Cell.error` and `tags`; none found. The only
non-label string found anywhere was the adapter's own scratchpad path in one `info` block.
