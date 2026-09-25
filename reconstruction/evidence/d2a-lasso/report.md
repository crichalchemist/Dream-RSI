# D2a run report

Workdir `/Users/Shared/dream-rsi-runs/d2a-lasso`, 1 launch(es), task `lasso_path`.
Archive `d2a-lasso.tar.gz`, sha256 `281b49ad4c5c41129e3bb1cd1fe64625fdb1ec99c3baccffa39ea8ea5db3cb2c`.

## 1. Per-round call budget

Caps in force: fallback 4 x 3 = 16 calls, hard max 6 x 4 = 30 calls. The paper: 110 calls per round (Pro), 640 (Flash).

| Iteration | Policy | Planned | Used fallback | Effective | Grid calls | Probes | Rounds |
|---|---|---|---|---|---|---|---|
| 1 | initial | 4 x 3 | False | 4 x 3 | 16 | 16 | 4 |

## 2. Out-of-support replay

0 version(s) replayed on clipped episodes; 0 of them deployed.

| Version | Episodes | Clipped | Asked | Recorded | Deployed |
|---|---|---|---|---|---|
| r0001_t01_m0 | 12 | 0 | - | - | False |
| r0002_t01_m1 | 12 | 0 | - | - | False |
| r0003_t01_m2 | 12 | 0 | - | - | True |

## 3. Untouched programs

1 of 16 attempts left their resume source byte for byte.

| Iteration | Cell | Source | Fail class | Agent timed out | Evaluated | Score | Source score |
|---|---|---|---|---|---|---|---|
| 1 | b2a3 | parent | compile_other | False | True | 0 | 0 |

## 4. Empty batches

Versions scored minus infinity for an empty batch: none. Live iterations whose policy left a batch empty: none.

| Version | Valid | Cause | Episode errors | Last error line |
|---|---|---|---|---|

## Health

| Iteration | Attempts | Successes | Fail classes | Agent timeouts | No program | Evaluator crashed | Baseline | Best | Error |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 16 | 11 | compile_other 2, ok 11, timeout 3 | 2 | 0 | 0 | 0.0134266 | 0.0167488 |  |

| Version | Valid | Reward | Eq. (1) V | Deployed |
|---|---|---|---|---|
| r0001_t01_m0 | True | 0.475 | 0.0167488 | False |
| r0002_t01_m1 | True | 0.4529 | 0.0167488 | False |
| r0003_t01_m2 | True | 0.532062 | 0.0167488 | True |

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

## Evaluation noise (smoke test)

| Program | Scores | Spread |
|---|---|---|
| dream_rsi | 0.0158657, 0.016035 | 0.0106749 |
| glmnet_port | 0.0134703, 0.0141842 | 0.0530004 |
| simpletes_best | 0.0159032, 0.0150628 | 0.0557886 |
