import json
import os

import pytest

from see.live import LiveQuestion
from see.loader import load_policy
from see.loop import DreamRSI, LoopConfig
from see.objective import run_episode
from see.pool import context_factory, load_pool
from see.toy import ScriptedDiscoveryAgent, ScriptedPolicyAgent, make_task

pytestmark = pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "generated", "policy_improvement_prompt.md"
        )
    ),
    reason="run tools/extract_listings.py first (needs the paper's prompts)",
)


@pytest.fixture(scope="module")
def finished_loop(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("loop"))
    cfg = LoopConfig(
        workdir=work,
        iterations=3,
        versions=3,
        max_parallelism=4,
        fallback_grid=(4, 5),
        hard_max_grid=(8, 8),
        betas=(0.0, 0.5, 1.0),
    )
    loop = DreamRSI(
        cfg,
        make_task(work),
        ScriptedDiscoveryAgent(seed=7),
        ScriptedPolicyAgent(betas=(0.3, 0.9), broken_every=2),
    )
    loop.run()
    return loop


def test_loop_writes_the_layout_listing_2_refers_to(finished_loop):
    w = finished_loop.w
    for t in (1, 2, 3):
        for name in ("trace.json", "live_cycle_manifest.json", "live_episode.jsonl"):
            assert os.path.exists(os.path.join(w, "trace_pool", f"iter{t:04d}", name))
    assert os.readlink(os.path.join(w, "trace_pool", "_current")) == "iter0003"
    base = os.path.join(w, "policy_dev", "history", "baseline", "proposal_results")
    assert os.path.exists(os.path.join(base, "beta_sweep.json"))
    rounds = sorted(d for d in os.listdir(os.path.join(w, "policy_dev", "history")) if d[0] == "r")
    assert len(rounds) == 9  # 3 iterations x M = 3 versions
    # one replay episode per (trace, beta) plus one per trace at the default beta
    first = os.path.join(
        w, "policy_dev", "history", rounds[0], "proposal_results", "policy_execution_traces.jsonl"
    )
    with open(first) as f:
        assert sum(1 for _ in f) == 1 * (3 + 1)


def test_broken_versions_score_minus_infinity_and_are_never_deployed(finished_loop):
    for entry in finished_loop.state["log"]:
        scores = {c["round"]: c for c in entry["offline"]}
        broken = [c for c in scores.values() if not c["valid"]]
        assert broken, "the scripted agent breaks every second revision"
        assert all(c["score"] == float("-inf") for c in broken)
        chosen = scores[entry["selected"]]
        assert chosen["valid"]
        assert chosen["score"] == max(c["score"] for c in scores.values())
        m0 = next(c for c in scores.values() if c["m"] == 0)
        assert chosen["score"] >= m0["score"]  # Sec. 3: never worse than pi_t on H_t


def test_on_policy_replay_reproduces_the_live_episode(finished_loop):
    """Replaying the deployed policy on its own tree must repeat its live batches."""
    w, c = finished_loop.w, finished_loop.c
    pool = load_pool(os.path.join(w, "trace_pool"))
    context_for = context_factory(pool, c.fallback_grid, c.hard_max_grid)
    for t, (trace, manifest) in enumerate(pool, start=1):
        cls = load_policy(os.path.join(w, "deployed", f"iter{t:04d}.py"))
        replay = run_episode(cls(None), trace, context_for(trace), record=True)
        with open(os.path.join(w, "trace_pool", f"iter{t:04d}", "live_episode.jsonl")) as f:
            live = [json.loads(line) for line in f]
        assert [s["batch"] for s in replay.log] == [s["batch"] for s in live]
        assert replay.probes == manifest["probes"] and replay.best == manifest["best_score"]


def test_plan_contexts_only_see_earlier_cycles(finished_loop):
    pool = load_pool(os.path.join(finished_loop.w, "trace_pool"))
    context_for = context_factory(pool, (4, 5), (8, 8))
    for i, (trace, _) in enumerate(pool):
        assert [m["iteration"] for m in context_for(trace).history] == list(range(1, i + 1))


def test_agent_crash_is_a_failed_attempt_not_a_failed_episode(tmp_path):
    task = make_task(str(tmp_path))

    def crashes_on_roots(prompt, *, cwd, target):
        if target.endswith("_a000"):
            raise RuntimeError("agent process died")
        return {"returncode": 0}  # leaves the resumed program untouched

    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        task,
        crashes_on_roots,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=1,
    )
    obs = q.probe_batch(q.legal_roots())
    assert [o.fail_class for o in obs] == ["no_program", "no_program"]
    assert not obs[0].evaluated
    assert (tree / "attempt_b000_a000" / "error.txt").exists()
    assert q.legal_actions() == ["b0a1", "b1a1"]  # the episode goes on
    # children of a crashed attempt resume from the nearest program: here the baseline
    obs = q.probe_batch(q.legal_actions())
    assert [o.fail_class for o in obs] == ["ok", "ok"] and obs[0].score == 1.0
    assert obs[0].delta_vs_parent is None  # the parent failed
