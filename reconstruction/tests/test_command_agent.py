"""Child-process lifetime through CommandAgent and LiveQuestion.

Every test forks a real child with a shell stand-in for an agent CLI (never a real agent or an
API) and asserts that a timeout, terminate() or an interrupt kills the whole process group.
"""

import json
import time

from see.live import CommandAgent, LiveQuestion
from see.toy import make_task

# Stands in for an agent CLI that forks work and waits for it. ``$0`` is the pid-file directory
# (``{dir}`` below); ``{prompt}`` is substituted by CommandAgent and ignored by the script. The
# pid file is renamed into place so a poller never reads it half-written.
STAND_IN = [
    "sh",
    "-c",
    'sleep 30 & echo $! > "$0/tmp-$$" && mv "$0/tmp-$$" "$0/child-$$.pid"; wait',
    "{dir}",
    "{prompt}",
]


def stand_in(pid_dir) -> list:
    return [str(pid_dir) if a == "{dir}" else a for a in STAND_IN]


def grandchild_pids(pid_dir) -> list:
    """The pids of every ``sleep`` the stand-in forked, one per call so far."""
    return [int(p.read_text()) for p in sorted(pid_dir.glob("child-*.pid"))]


def test_agent_timeout_kills_the_whole_process_group(tmp_path, stub_prompts, process_gone):
    """A timed-out agent and everything it forked are dead when the call returns; through
    LiveQuestion the attempt is scored as the program the agent left, with agent_timed_out set."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=0.3, kill_grace=0.2)
    started = time.time()
    result = agent("ignored prompt", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 3.0  # the child slept for 30 s; the group was killed
    assert result == {
        "returncode": None,
        "stdout": "",
        "stderr": "agent timed out after 0.3s",
        "timed_out": True,
    }
    [pid] = grandchild_pids(pid_dir)
    assert process_gone(pid)
    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        agent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=1,
        branch_count=1,
        refine_count=0,
    )
    [obs] = q.probe_batch(q.legal_roots())
    # the stand-in never touched the program, so the parent's copy (x=1.0) is what gets scored
    assert (obs.cell_id, obs.score, obs.evaluated, obs.fail_class) == ("b0a0", 1.0, True, "ok")
    with open(tree / "attempt_b000_a000" / "eval" / "score.json") as f:
        score = json.load(f)
    assert score["agent_timed_out"] is True and score["agent_returncode"] is None
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)
