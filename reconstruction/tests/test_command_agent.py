"""Child-process lifetime through CommandAgent and LiveQuestion.

Every test forks a real child with a shell stand-in for an agent CLI (never a real agent or an
API) and asserts that a timeout, terminate() or an interrupt kills the whole process group.
"""

import json
import os
import signal
import sys
import threading
import time

import pytest

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


class _Interrupted(BaseException):
    """Stands in for KeyboardInterrupt, which pytest intercepts itself."""


def test_interrupt_during_an_agent_call_kills_its_process_group(tmp_path, process_gone):
    """An interrupt raised in the calling thread (Ctrl-C during the policy-agent call) does not
    leave the agent CLI and its children running: the group dies before the interrupt propagates."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)

    def interrupt(signum, frame):
        raise _Interrupted("SIGALRM while the agent was running")

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 0.3)  # the stand-in has forked its child by then
    started = time.time()
    try:
        with pytest.raises(_Interrupted):
            agent("p", cwd=str(tmp_path), target=str(tmp_path))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 2.0  # not the 3 s timeout
    [pid] = grandchild_pids(pid_dir)
    assert process_gone(pid)


# An agent that prints half a multibyte character, then forks a child that leaves the process
# group (setsid) but keeps the inherited pipes, then hangs.
ESCAPEE = (
    "import subprocess, sys, time\n"
    "sys.stdout.buffer.write(b'\\xe2\\x82')\n"
    "sys.stdout.flush()\n"
    "subprocess.Popen([sys.executable, '-c', 'import os, time; os.setsid(); time.sleep(3)'])\n"
    "time.sleep(30)\n"
)

# The escapee stand-in whose escapee touches a marker file (the agent's argv[2]) once it has left
# the process group, so a test can act only after the escape has really happened.
ESCAPEE_THEN_MARK = (
    "import subprocess, sys, time\n"
    "sys.stdout.buffer.write(b'\\xe2\\x82')\n"
    "sys.stdout.flush()\n"
    "subprocess.Popen([sys.executable, '-c', 'import os, pathlib, sys, time; os.setsid(); "
    "pathlib.Path(sys.argv[1]).touch(); time.sleep(3)', sys.argv[2]])\n"
    "time.sleep(30)\n"
)


def test_timeout_returns_even_when_an_escapee_holds_the_pipes(tmp_path):
    """A descendant outside the group is not killed, by design; it must not make the timed-out
    call wait for it, and the bytes it left half-written must not raise."""
    agent = CommandAgent([sys.executable, "-c", ESCAPEE, "{prompt}"], timeout=0.3, kill_grace=0.2)
    started = time.time()
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 2.0  # the escapee sleeps 3 s
    assert result["timed_out"] is True and result["stdout"] == ""


def test_terminate_kills_live_agents_and_refuses_new_calls(tmp_path, process_gone):
    """terminate() ends a call in flight and kills what it forked; later calls spawn nothing."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)
    results: list = []
    in_flight = threading.Thread(
        target=lambda: results.append(agent("p", cwd=str(tmp_path), target=str(tmp_path))),
        daemon=True,
    )
    in_flight.start()
    deadline = time.time() + 2.0
    while not grandchild_pids(pid_dir) and time.time() < deadline:
        time.sleep(0.02)  # until the stand-in has forked its child
    [pid] = grandchild_pids(pid_dir)
    started = time.time()
    agent.terminate()
    in_flight.join(timeout=2.0)
    assert not in_flight.is_alive() and time.time() - started < 2.0  # not the 3 s timeout
    assert results == [{"returncode": None, "stdout": "", "stderr": "agent terminated"}]
    assert process_gone(pid)
    assert agent("p", cwd=str(tmp_path), target=str(tmp_path)) == {
        "returncode": None,
        "stdout": "",
        "stderr": "agent terminated",
    }
    assert grandchild_pids(pid_dir) == [pid]  # the refused call forked nothing
    agent.terminate()  # idempotent with nothing live


# The stand-in with SIGTERM ignored; the sleep it forks inherits that (an ignored signal stays
# ignored across exec), so only the SIGKILL after kill_grace ends either of them.
TERM_PROOF_STAND_IN = [
    "sh",
    "-c",
    'trap "" TERM; sleep 30 & echo $! > "$0/tmp-$$" && mv "$0/tmp-$$" "$0/child-$$.pid"; wait',
    "{dir}",
    "{prompt}",
]


def test_terminate_gives_every_live_group_one_shared_grace_period(tmp_path, process_gone):
    """Three agents that ignore SIGTERM are SIGKILLed after one kill_grace, not one each: the
    cost of an interrupt does not grow with the batch width."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    argv = [str(pid_dir) if a == "{dir}" else a for a in TERM_PROOF_STAND_IN]
    agent = CommandAgent(argv, timeout=10.0, kill_grace=0.5)
    calls = [
        threading.Thread(
            target=lambda: agent("p", cwd=str(tmp_path), target=str(tmp_path)), daemon=True
        )
        for _ in range(3)
    ]
    for call in calls:
        call.start()
    deadline = time.time() + 5.0
    while len(grandchild_pids(pid_dir)) < 3 and time.time() < deadline:
        time.sleep(0.02)  # until every stand-in has forked its child
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 3
    started = time.time()
    agent.terminate()
    elapsed = time.time() - started
    assert 0.5 <= elapsed < 1.2  # SIGTERM was ignored, and one grace period covered all three
    for call in calls:
        call.join(timeout=2.0)
    assert not any(call.is_alive() for call in calls)
    assert all(process_gone(pid) for pid in pids)


def test_interrupt_kills_running_agents_and_records_nothing(tmp_path, stub_prompts, process_gone):
    """Every attempt in a batch is running when an interrupt lands in the main thread (as SIGINT
    does; probe_batch caps a batch at max_parallelism): both agents' children are killed, the
    killed attempts are not evaluated, and nothing from the batch is recorded."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)
    tree = tmp_path / "tree"
    tree.mkdir()
    q = LiveQuestion(
        make_task(str(tmp_path)),
        agent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=0,
    )

    def interrupt(signum, frame):
        raise _Interrupted("SIGALRM while two agents were running")

    def once_both_are_running():  # the signal lands only after both stand-ins have forked
        deadline = time.time() + 5.0
        while len(grandchild_pids(pid_dir)) < 2 and time.time() < deadline:
            time.sleep(0.02)
        os.kill(os.getpid(), signal.SIGALRM)

    previous = signal.signal(signal.SIGALRM, interrupt)
    threading.Thread(target=once_both_are_running, daemon=True).start()
    started = time.time()
    try:
        with pytest.raises(_Interrupted):
            q.probe_batch(q.legal_roots())
    finally:
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 2.0  # not the 3 s timeout
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)
    assert q.cells == {}  # nothing from the interrupted batch is kept
    assert sorted(os.listdir(tree)) == ["attempt_b000_a000", "attempt_b001_a000"]
    assert not list(tree.rglob("score.json"))  # killed attempts are not evaluated


def test_terminate_returns_even_when_an_escapee_holds_the_pipes(tmp_path):
    """terminate() ends a call within about a second even when a descendant outside the group
    still holds the agent's pipes: the call reports terminated instead of waiting for EOF."""
    marker = tmp_path / "escapee-forked"
    agent = CommandAgent(
        [sys.executable, "-c", ESCAPEE_THEN_MARK, "{prompt}", str(marker)],
        timeout=10.0,
        kill_grace=0.2,
    )
    results: list = []
    in_flight = threading.Thread(
        target=lambda: results.append(agent("p", cwd=str(tmp_path), target=str(tmp_path))),
        daemon=True,
    )
    in_flight.start()
    deadline = time.time() + 5.0
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.02)
    assert marker.exists()
    started = time.time()
    agent.terminate()
    in_flight.join(timeout=4.0)
    assert not in_flight.is_alive() and time.time() - started < 2.0  # the escapee sleeps 3 s
    assert results == [{"returncode": None, "stdout": "", "stderr": "agent terminated"}]


def test_a_half_written_multibyte_character_does_not_crash_the_call(tmp_path):
    """An agent killed or exiting between the bytes of one character must not raise out of the
    call: its output is diagnostic text, decoded with replacement."""
    agent = CommandAgent(["sh", "-c", "printf '\\342\\202'; exit 3", "{prompt}"], timeout=3.0)
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert result == {"returncode": 3, "stdout": "�", "stderr": ""}
