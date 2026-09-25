# D2a run report

Workdir `/Users/Shared/dream-rsi-runs/d2a-lasso-restart1`, 1 launch(es), task `lasso_path`.
Archive `d2a-lasso-restart1.tar.gz`, sha256 `0cc3528026b9ab44683eabbbddde9f00d4221260fe4bc034fdff61a59b324659`.

## 1. Per-round call budget

Caps in force: fallback 4 x 3 = 16 calls, hard max 6 x 4 = 30 calls. The paper: 110 calls per round (Pro), 640 (Flash).

| Iteration | Policy | Planned | Used fallback | Effective | Grid calls | Probes | Rounds |
|---|---|---|---|---|---|---|---|
| 2 (partial) | r0003_t01_m2 | 4 x 4 | False | 4 x 4 | 20 | 4 | 2 |

## 2. Out-of-support replay

0 version(s) replayed on clipped episodes; 0 of them deployed.

| Version | Episodes | Clipped | Asked | Recorded | Deployed |
|---|---|---|---|---|---|

## 3. Untouched programs

4 of 4 attempts left their resume source byte for byte.

| Iteration | Cell | Source | Fail class | Agent timed out | Evaluated | Score | Source score |
|---|---|---|---|---|---|---|---|
| 2 | b0a0 | baseline | ok | False | True | 0.0151303 | 0.0134266 |
| 2 | b1a0 | baseline | ok | False | True | 0.0148444 | 0.0134266 |
| 2 | b2a0 | baseline | ok | False | True | 0.0143934 | 0.0134266 |
| 2 | b3a0 | baseline | ok | False | True | 0.0150375 | 0.0134266 |

## 4. Empty batches

Versions scored minus infinity for an empty batch: none. Live iterations whose policy left a batch empty: none.

| Version | Valid | Cause | Episode errors | Last error line |
|---|---|---|---|---|

## Health

| Iteration | Attempts | Successes | Fail classes | Agent timeouts | No program | Evaluator crashed | Baseline | Best | Error |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 4 | 4 | ok 4 | 0 | 0 | 0 | 0.0134266 | 0.0151303 | KeyboardInterrupt: signal 15 |

| Version | Valid | Reward | Eq. (1) V | Deployed |
|---|---|---|---|---|

## Host

- compiler: g++ (MacPorts gcc13 13.4.0_1+stdlib_flag) 13.4.0
- cpu: Intel(R) Core(TM) i7-7700K CPU @ 4.20GHz
- cpus: 8
- eigen: 3.4.1
- platform: macOS-13.7.8-x86_64-i386-64bit-Mach-O
- python: 3.13.15
- simpletes_commit: 47d3413da1d85dc24341219d47452d2601e56a57
- discovery agent: `["env", "HOME=/Users/Shared/dream-rsi-runs/agy-home", "agy", "-p", "{prompt}", "--model", "gemini-3.1-pro-high", "--dangerously-skip-permissions", "--add-dir", "/Users/Shared/dream-rsi-runs/d2a-lasso"]`, version 1.2.11
- policy agent: `["claude", "-p", "{prompt}", "--permission-mode", "acceptEdits", "--setting-sources", "project,local", "--strict-mcp-config", "--add-dir", "/Users/Shared/dream-rsi-runs/d2a-lasso/trace_pool"]`, version 2.1.283 (Claude Code)
