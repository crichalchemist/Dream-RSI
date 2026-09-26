"""Turn a Dream-RSI workdir into the D2a evidence report: report.md and report.json.

    python scripts/report_run.py --workdir ~/dream-rsi-runs/d2a-lasso --out evidence/d2a-lasso \\
        --host-json evidence/d2a-lasso/host.json --archive ~/dream-rsi-runs/d2a-lasso.tar.gz \\
        --copy-evidence

Read-only over the workdir. It answers, with counts, the four questions track D1 deferred to D2
(docs/superpowers/specs/2026-09-25-real-agent-run-d2a-design.md, section 5): what each round
spent against its caps, which policy versions were scored on replay episodes clipped to a
recorded tree, which attempts left their resume source byte for byte, and which versions or live
batches ended on an empty or illegal batch. The caps and the program's file name come from the
last line of the runner's launches.jsonl. Programs are compared as they are on disk after the
run; an agent runs with the whole tree as its cwd, so a later agent could have edited an earlier
attempt's program.
"""

import argparse
import collections
import glob
import hashlib
import json
import os
import shutil

from see.live import node_dirname
from see.world import Trace

PAPER_CALLS = {"pro": 110, "flash": 640}  # Sec. 4: 10 x 11 and 32 x 20 calls per round
# The license-safe subset of a workdir (spec section 6): no attempt program is ever listed.
EVIDENCE = (
    "state.json",
    "launches.jsonl",
    "trace_pool/iter*/trace.json",
    "trace_pool/iter*/live_cycle_manifest.json",
    "trace_pool/iter*/live_episode.jsonl",
    "runs/iter*/partial/trace.json",
    "runs/iter*/partial/live_cycle_manifest.json",
    "runs/iter*/partial/live_episode.jsonl",
    "runs/iter*/tree/attempt_*/eval/score.json",
    "runs/iter*/tree/attempt_*/error.txt",
    "runs/iter*/tree/attempt_*/proposal.md",
    "policy_dev/history/r*/method.py",
    "policy_dev/history/r*/proposal_results/beta_sweep.json",
    "policy_dev/history/r*/proposal_results/policy_execution_traces.jsonl",
)
PROGRAM_MARKER = (
    "CPP_CODE"  # opens every SimpleTES Lasso program (AGPL); a file quoting it stays out
)


def _json(path: str):
    with open(path) as f:
        return json.load(f)


def _jsonl(path: str) -> list:
    if not os.path.exists(path):  # a sweep that crashed or timed out wrote only beta_sweep.json
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def calls(grid) -> int:
    """Attempts a (branch_count, refine_count) grid admits; R counts refinements after the root."""
    return grid[0] * (grid[1] + 1)


def resume_source(tree_dir: str, baseline_dir: str, program: str, branch: int, attempt: int):
    """see.live.LiveQuestion._resume_from, read back after the run: (path, source attempt).

    The parent's saved program; past an attempt that left none, the nearest ancestor's; else the
    baseline's, with source attempt None.
    """
    for a in range(attempt - 1, -1, -1):
        path = os.path.join(tree_dir, node_dirname(branch, a), program)
        if os.path.exists(path):
            return path, a
    return os.path.join(baseline_dir, program), None


def cause(errors: list) -> str | None:
    """Why a version or a live batch failed: the empty batch D2 asks about, another illegal one,
    or anything else."""
    text = "\n".join(errors)
    if "empty batch" in text:
        return "empty_batch"
    if "IllegalBatch" in text:
        return "illegal_batch"
    return "other" if errors else None


def _last_line(error: str) -> str:
    lines = error.strip().splitlines()
    return lines[-1] if lines else ""


def analyse_iteration(workdir: str, run_dir: str, frozen: str, program: str) -> tuple:
    """(iteration row, untouched attempts, trace) for one iteration's frozen or partial trace."""
    trace = Trace.load(os.path.join(frozen, "trace.json"))
    manifest = _json(os.path.join(frozen, "live_cycle_manifest.json"))
    tree = os.path.join(run_dir, "tree")
    baseline_dir = os.path.join(workdir, "task", "baseline")
    t = manifest["iteration"]
    timeouts = crashed = failures = unchecked = 0
    untouched = []
    for c in sorted(trace.cells.values(), key=lambda c: c.seq):
        node = os.path.join(tree, node_dirname(c.branch, c.attempt))
        score_path = os.path.join(node, "eval", "score.json")
        score = _json(score_path) if os.path.exists(score_path) else {}
        timeouts += bool(score.get("agent_timed_out"))
        crashed += bool(score.get("evaluator_crashed"))
        returncode = score.get("agent_returncode")
        failures += returncode is not None and returncode != 0
        mine = os.path.join(node, program)
        source, source_attempt = resume_source(tree, baseline_dir, program, c.branch, c.attempt)
        if not os.path.exists(mine):
            # a cell the agent left a program for (fail_class != "no_program") but whose file is
            # gone from disk (e.g. a report run on a program-stripped copy) cannot be checked
            unchecked += c.fail_class != "no_program"
            continue
        if _bytes(mine) != _bytes(source):
            continue
        if source_attempt is None:
            kind, source_score = "baseline", trace.baseline_score
        else:
            parent = trace.cell(c.branch, source_attempt)
            kind = "parent" if source_attempt == c.attempt - 1 else "ancestor"
            source_score = parent.score if parent else None
        untouched.append(
            {
                "iteration": t,
                "cell": c.id,
                "source": kind,
                "fail_class": c.fail_class,
                "agent_timed_out": bool(score.get("agent_timed_out")),
                "agent_returncode": returncode,
                "evaluated": c.evaluated,
                "score": c.score,
                "source_score": source_score,
            }
        )
    planned = manifest.get("planned_grid")
    grid = manifest["effective_grid"]
    fail_classes = collections.Counter(c.fail_class for c in trace.cells.values())
    row = {
        "iteration": t,
        "partial": bool(manifest.get("partial")),
        "policy_round": manifest.get("policy_round"),
        "planned_grid": [planned["branch_count"], planned["refine_count"]] if planned else None,
        "used_fallback": manifest.get("used_fallback"),
        "effective_grid": [grid["branch_count"], grid["refine_count"]],
        "grid_calls": calls((grid["branch_count"], grid["refine_count"])),
        "probes": manifest["probes"],
        "decision_rounds": manifest["decision_rounds"],
        "attempts": len(trace),
        "successes": sum(c.success for c in trace.cells.values()),
        "fail_classes": dict(sorted(fail_classes.items())),
        "agent_timeouts": timeouts,
        "agent_failures": failures,
        "no_program": fail_classes.get("no_program", 0),
        "evaluator_crashed": crashed,
        "untouched_unchecked": unchecked,
        "baseline_score": trace.baseline_score,
        "best_score": manifest.get("best_score"),
        "error": manifest.get("error"),
        "error_cause": cause([manifest["error"]]) if manifest.get("error") else None,
    }
    return row, untouched, trace


def analyse_version(rdir: str, grids: dict, fallback, selected: set) -> dict:
    """One policy version: its sweep's validity and scores, and its clipped replay episodes."""
    name = os.path.basename(rdir)
    _, t, m = name.split("_")
    results = os.path.join(rdir, "proposal_results")
    report_path = os.path.join(results, "beta_sweep.json")
    report = _json(report_path) if os.path.exists(report_path) else {}
    episodes = _jsonl(os.path.join(results, "policy_execution_traces.jsonl"))
    clipped = [e for e in episodes if e["out_of_support"]]
    asked = {
        (e["plan"]["branch_count"], e["plan"]["refine_count"]) if e["plan"] else tuple(fallback)
        for e in clipped
    }
    recorded = {tuple(grids[e["trace_id"]]) for e in clipped if e["trace_id"] in grids}
    # every episode's error, not beta_sweep.json's first three; the report's own when none ran
    errors = [e["error"] for e in episodes if e["error"]] or list(report.get("errors", []))
    return {
        "version": name,
        "iteration": int(t[1:]),
        "m": int(m[1:]),
        "valid": bool(report.get("valid", False)),
        "reward": report.get("pareto", {}).get("reward"),
        "eq1": report.get("eq1", {}).get("V"),
        "episodes": len(episodes),
        "clipped": len(clipped),
        "asked": [list(a) for a in sorted(asked)],
        "recorded": [list(g) for g in sorted(recorded)],
        "episode_errors": sum(1 for e in episodes if e["error"]),
        "cause": cause(errors),
        "first_error": _last_line(errors[0]) if errors else None,
        "deployed": name in selected,
    }


def noise(host_json: str) -> dict:
    """The smoke test's repeats per program; spread is (max - min) / min of combined_score."""
    out = {}
    for name, runs in _json(host_json)["results"].items():
        scores = [r["combined_score"] for r in runs]
        out[name] = {
            "scores": scores,
            "geo_mean_ms": [r["geo_mean_sol_ms"] for r in runs],
            "spread": (max(scores) - min(scores)) / min(scores) if min(scores) > 0 else None,
        }
    return out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def copy_evidence(workdir: str, dest: str) -> dict:
    """Copy the EVIDENCE subset under ``dest``, withholding any file that quotes a program."""
    copied, withheld = 0, []
    workdir_real = os.path.realpath(workdir)
    for pattern in EVIDENCE:
        for path in sorted(glob.glob(os.path.join(glob.escape(workdir), pattern))):
            rel = os.path.relpath(path, workdir)
            real = os.path.realpath(path)
            outside = os.path.relpath(real, workdir_real).startswith(os.pardir)
            if os.path.islink(path) or outside:
                withheld.append(rel)
                continue
            with open(path, errors="replace") as f:
                if PROGRAM_MARKER in f.read():
                    withheld.append(rel)
                    continue
            target = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copyfile(path, target)
            copied += 1
    return {"copied": copied, "withheld": withheld}


def _home_relative(path: str) -> str:
    """The report is published: show a path under the home directory as ~/..."""
    home = os.path.expanduser("~")
    return "~" + path[len(home) :] if path == home or path.startswith(home + os.sep) else path


def build_report(workdir: str, host_json: str | None = None, archive: str | None = None) -> dict:
    workdir = os.path.abspath(workdir)
    launches = _jsonl(os.path.join(workdir, "launches.jsonl"))
    if not launches:
        raise FileNotFoundError(f"{workdir}/launches.jsonl: not a workdir the runner launched")
    launch = launches[-1]
    program = launch["task"]["eval_program"]
    fallback, hard_max = launch["config"]["fallback_grid"], launch["config"]["hard_max_grid"]
    state_path = os.path.join(workdir, "state.json")
    state = _json(state_path) if os.path.exists(state_path) else {"log": []}
    selected = {entry["selected"] for entry in state["log"] if "selected" in entry}
    rows, untouched, grids, unfrozen = [], [], {}, []
    for run_dir in sorted(glob.glob(os.path.join(glob.escape(workdir), "runs", "iter*"))):
        name = os.path.basename(run_dir)
        frozen = os.path.join(workdir, "trace_pool", name)
        if not os.path.exists(os.path.join(frozen, "trace.json")):
            frozen = os.path.join(run_dir, "partial")
        if not os.path.exists(os.path.join(frozen, "trace.json")):
            unfrozen.append(name)
            continue
        row, cases, trace = analyse_iteration(workdir, run_dir, frozen, program)
        rows.append(row)
        untouched += cases
        grids[trace.trace_id] = list(trace.grid)
    history = os.path.join(glob.escape(workdir), "policy_dev", "history", "r*")
    versions = [analyse_version(r, grids, fallback, selected) for r in sorted(glob.glob(history))]
    flagged = [v for v in versions if v["clipped"]]
    return {
        "workdir": _home_relative(workdir),
        "launches": len(launches),
        "task": launch["task"],
        "agents": launch.get("agents"),
        "host": launch.get("host"),
        "caps": {
            "fallback_grid": fallback,
            "fallback_calls": calls(fallback),
            "hard_max_grid": hard_max,
            "hard_max_calls": calls(hard_max),
            "paper_calls": PAPER_CALLS,
        },
        "archive": {"file": os.path.basename(archive), "sha256": sha256_of(archive)}
        if archive
        else None,
        "iterations": rows,
        "unfrozen_iterations": unfrozen,
        "versions": versions,
        "out_of_support": {
            "flagged": [v["version"] for v in flagged],
            "flagged_deployed": [v["version"] for v in flagged if v["deployed"]],
        },
        "untouched": {
            "attempts": sum(r["attempts"] for r in rows),
            "cases": untouched,
            "unchecked": sum(r["untouched_unchecked"] for r in rows),
        },
        "empty_batches": {
            "versions": [v["version"] for v in versions if v["cause"] == "empty_batch"],
            "live": [r["iteration"] for r in rows if r["error_cause"] == "empty_batch"],
        },
        "noise": noise(host_json) if host_json else None,
    }


def _grid(g) -> str:
    return f"{g[0]} x {g[1]}" if g else "none"


def _num(x) -> str:
    return "n/a" if x is None else f"{x:.6g}" if isinstance(x, float) else str(x)


def _cell(text) -> str:
    """Free text placed in a markdown table cell: newlines collapsed to spaces, ``|`` escaped."""
    return str(text).replace("\n", " ").replace("|", "\\|")


def markdown(r: dict) -> str:
    caps = r["caps"]
    out = [
        "# D2a run report",
        "",
        f"Workdir `{r['workdir']}`, {r['launches']} launch(es), task `{r['task'].get('name')}`.",
    ]
    if r["archive"]:
        out.append(f"Archive `{r['archive']['file']}`, sha256 `{r['archive']['sha256']}`.")
    out += [
        "",
        "## 1. Per-round call budget",
        "",
        f"Caps in force: fallback {_grid(caps['fallback_grid'])} = {caps['fallback_calls']} calls, "
        f"hard max {_grid(caps['hard_max_grid'])} = {caps['hard_max_calls']} calls. The paper: "
        f"{caps['paper_calls']['pro']} calls per round (Pro), {caps['paper_calls']['flash']} "
        "(Flash).",
        "",
        "| Iteration | Policy | Planned | Used fallback | Effective | Grid calls | Probes "
        "| Rounds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in r["iterations"]:
        out.append(
            f"| {i['iteration']}{' (partial)' if i['partial'] else ''} | {i['policy_round']} | "
            f"{_grid(i['planned_grid'])} | {i['used_fallback']} | {_grid(i['effective_grid'])} | "
            f"{i['grid_calls']} | {i['probes']} | {i['decision_rounds']} |"
        )
    oos = r["out_of_support"]
    out += [
        "",
        "## 2. Out-of-support replay",
        "",
        f"{len(oos['flagged'])} version(s) replayed on clipped episodes; "
        f"{len(oos['flagged_deployed'])} of them deployed.",
        "",
        "| Version | Episodes | Clipped | Asked | Recorded | Deployed |",
        "|---|---|---|---|---|---|",
    ]
    for v in r["versions"]:
        asked = ", ".join(_grid(a) for a in v["asked"]) or "-"
        recorded = ", ".join(_grid(g) for g in v["recorded"]) or "-"
        out.append(
            f"| {v['version']} | {v['episodes']} | {v['clipped']} | {asked} | {recorded} | "
            f"{v['deployed']} |"
        )
    u = r["untouched"]
    checked = u["attempts"] - u["unchecked"]
    coverage = f"Untouched check covers {checked} of {u['attempts']} attempts"
    coverage += f"; {u['unchecked']} had no program file on disk." if u["unchecked"] else "."
    out += [
        "",
        "## 3. Untouched programs",
        "",
        f"{len(u['cases'])} of {checked} attempts left their resume source byte for byte. "
        + coverage,
        "",
        "| Iteration | Cell | Source | Fail class | Agent timed out | Agent returncode | "
        "Evaluated | Score | Source score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in u["cases"]:
        out.append(
            f"| {c['iteration']} | {c['cell']} | {c['source']} | {c['fail_class']} | "
            f"{c['agent_timed_out']} | {_num(c['agent_returncode'])} | {c['evaluated']} | "
            f"{_num(c['score'])} | {_num(c['source_score'])} |"
        )
    e = r["empty_batches"]
    out += [
        "",
        "## 4. Empty batches",
        "",
        f"Versions scored minus infinity for an empty batch: {', '.join(e['versions']) or 'none'}. "
        f"Live iterations whose policy left a batch empty: "
        f"{', '.join(map(str, e['live'])) or 'none'}.",
        "",
        "| Version | Valid | Cause | Episode errors | Last error line |",
        "|---|---|---|---|---|",
    ]
    for v in r["versions"]:
        if v["cause"]:
            out.append(
                f"| {v['version']} | {v['valid']} | {v['cause']} | {v['episode_errors']} | "
                f"`{_cell(v['first_error'])}` |"
            )
    out += [
        "",
        "## Health",
        "",
        "| Iteration | Attempts | Successes | Fail classes | Agent timeouts | Agent failures | "
        "No program | Evaluator crashed | Baseline | Best | Error |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i in r["iterations"]:
        classes = ", ".join(f"{k} {n}" for k, n in i["fail_classes"].items())
        out.append(
            f"| {i['iteration']} | {i['attempts']} | {i['successes']} | {classes} | "
            f"{i['agent_timeouts']} | {i['agent_failures']} | {i['no_program']} | "
            f"{i['evaluator_crashed']} | {_num(i['baseline_score'])} | {_num(i['best_score'])} | "
            f"{_cell(i['error'] or '')} |"
        )
    out += ["", "| Version | Valid | Reward | Eq. (1) V | Deployed |", "|---|---|---|---|---|"]
    for v in r["versions"]:
        out.append(
            f"| {v['version']} | {v['valid']} | {_num(v['reward'])} | {_num(v['eq1'])} | "
            f"{v['deployed']} |"
        )
    if r["unfrozen_iterations"]:
        out += ["", f"Iterations with no trace: {', '.join(r['unfrozen_iterations'])}."]
    out += ["", "## Host", ""]
    for k, v in sorted((r["host"] or {}).items()):
        out.append(f"- {k}: {v}")
    for role, a in sorted((r["agents"] or {}).items()):
        out.append(f"- {role} agent: `{json.dumps(a.get('argv'))}`, version {a.get('version')}")
    if r["noise"]:
        out += [
            "",
            "## Evaluation noise (smoke test)",
            "",
            "| Program | Scores | Spread |",
            "|---|---|---|",
        ]
        for name, n in r["noise"].items():
            scores = ", ".join(_num(s) for s in n["scores"])
            out.append(f"| {name} | {scores} | {_num(n['spread'])} |")
    return "\n".join(out) + "\n"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host-json", help="the smoke test's verify_lasso.py --json output")
    ap.add_argument("--archive", help="the workdir archive whose sha256 the report records")
    ap.add_argument(
        "--copy-evidence",
        action="store_true",
        help="also copy the license-safe subset of the workdir to <out>/workdir/",
    )
    a = ap.parse_args(argv)
    report = build_report(a.workdir, a.host_json, a.archive)
    os.makedirs(a.out, exist_ok=True)
    if a.copy_evidence:
        report["evidence"] = copy_evidence(
            os.path.abspath(a.workdir), os.path.join(a.out, "workdir")
        )
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write(markdown(report))
    print(os.path.join(a.out, "report.md"))
    return report


if __name__ == "__main__":
    main()
