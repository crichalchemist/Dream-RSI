"""Child-process lifetime through CommandAgent and LiveQuestion.

Every test forks a real child with a shell stand-in for an agent CLI (never a real agent or an
API) and asserts that a timeout, terminate() or an interrupt kills the whole process group.
"""

import concurrent.futures
import json
import os
import signal
import sys
import threading
import time

import pytest

from see.live import CommandAgent, LiveQuestion
from see.toy import PROGRAM, make_task

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
    LiveQuestion the attempt is scored as the program the agent left but recorded as the
    `timeout` failure it is, never as a valid non-improving attempt."""
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
    # the stand-in never touched the program, so the parent's copy (x=1.0) is what gets scored,
    # but the attempt is a timeout failure: not a success, and it never raises the ceiling
    assert (obs.cell_id, obs.score, obs.evaluated, obs.fail_class) == ("b0a0", 1.0, True, "timeout")
    assert obs.error == "agent timed out after 0.3s"
    assert obs.delta_vs_parent is None and obs.delta_vs_baseline is None  # not a success
    node = tree / "attempt_b000_a000"
    with open(node / "eval" / "score.json") as f:
        score = json.load(f)
    assert score["agent_timed_out"] is True and score["agent_returncode"] is None
    assert score["fail_class"] == "timeout" and score["combined_score"] == 1.0
    assert (node / "error.txt").read_text() == "agent timed out after 0.3s"
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)


def test_a_timed_out_agent_that_left_a_broken_program_keeps_the_evaluators_verdict(tmp_path):
    """When the evaluator itself rejects what a timed-out agent left, its error and class win:
    the program is broken either way, and score.json's agent_timed_out keeps the cause."""
    tree = tmp_path / "tree"
    tree.mkdir()
    target = tree / "attempt_b000_a000" / PROGRAM
    writes_then_hangs = ["sh", "-c", 'printf "{broken" > "$0"; sleep 30', str(target), "{prompt}"]
    agent = CommandAgent(writes_then_hangs, timeout=0.3, kill_grace=0.2)
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
    assert obs.fail_class == "code" and obs.evaluated and obs.score == 0.0
    assert obs.error is not None and obs.error.startswith("ValueError: unreadable solution")
    with open(tree / "attempt_b000_a000" / "eval" / "score.json") as f:
        assert json.load(f)["agent_timed_out"] is True


class _Interrupted(BaseException):
    """Stands in for KeyboardInterrupt, which pytest intercepts itself."""


class _AlarmOnce:
    """Send SIGALRM to this process from a helper thread once ``ready()`` holds, so the signal
    lands while the stand-ins are known to be running rather than after a fixed timer that races
    shell startup; ``cancel()`` makes sure no late alarm lands after the handler is restored."""

    def __init__(self, ready):
        self._ready, self._lock, self._done = ready, threading.Lock(), False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        deadline = time.time() + 5.0
        while not self._ready() and time.time() < deadline:
            time.sleep(0.02)
        with self._lock:
            if not self._done:
                os.kill(os.getpid(), signal.SIGALRM)

    def cancel(self):
        with self._lock:
            self._done = True


def test_interrupt_during_an_agent_call_kills_its_process_group(tmp_path, process_gone):
    """An interrupt raised in the calling thread (Ctrl-C during the policy-agent call) does not
    leave the agent CLI and its children running: the group dies before the interrupt propagates."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    agent = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)

    def interrupt(signum, frame):
        raise _Interrupted("SIGALRM while the agent was running")

    previous = signal.signal(signal.SIGALRM, interrupt)
    alarm = _AlarmOnce(lambda: bool(grandchild_pids(pid_dir)))  # once the stand-in has forked
    started = time.time()
    try:
        with pytest.raises(_Interrupted):
            agent("p", cwd=str(tmp_path), target=str(tmp_path))
    finally:
        alarm.cancel()
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 2.0  # not the 3 s timeout
    [pid] = grandchild_pids(pid_dir)
    assert process_gone(pid)


# An agent that prints half a multibyte character, then forks a child that leaves the process
# group (setsid) but keeps the inherited pipes, then hangs. The escapee touches a marker file (the
# agent's argv[2]) once it has left the group, so a test can prove the escape really happened.
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
    marker = tmp_path / "escapee-forked"
    agent = CommandAgent(
        [sys.executable, "-c", ESCAPEE_THEN_MARK, "{prompt}", str(marker)],
        timeout=0.3,
        kill_grace=0.2,
    )
    started = time.time()
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 2.0  # the escapee sleeps 3 s
    assert result["timed_out"] is True and result["stdout"] == ""
    assert marker.exists()  # an escapee really was holding the pipes while the call returned


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

    previous = signal.signal(signal.SIGALRM, interrupt)
    alarm = _AlarmOnce(lambda: len(grandchild_pids(pid_dir)) >= 2)  # once both have forked
    started = time.time()
    try:
        with pytest.raises(_Interrupted):
            q.probe_batch(q.legal_roots())
    finally:
        alarm.cancel()
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 2.0  # not the 3 s timeout
    pids = grandchild_pids(pid_dir)
    assert len(pids) == 2 and all(process_gone(p) for p in pids)
    assert q.cells == {}  # nothing from the interrupted batch is kept
    assert sorted(os.listdir(tree)) == ["attempt_b000_a000", "attempt_b001_a000"]
    assert not list(tree.rglob("score.json"))  # killed attempts are not evaluated


def test_a_worker_fault_kills_the_batchs_running_agents_but_keeps_the_agent_usable(
    tmp_path, stub_prompts, process_gone
):
    """An exception a worker raises after its agent returned (here a host fault: the attempt's
    eval/ path exists as a file) ends the episode; the sibling agent still running is killed
    instead of running to its timeout, nothing from the batch is recorded, and the agent stays
    open, because offline() and the next iteration use the same one."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    inner = CommandAgent(stand_in(pid_dir), timeout=3.0, kill_grace=0.2)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "attempt_b000_a000").mkdir()
    (tree / "attempt_b000_a000" / "eval").write_text("a file where the eval directory goes\n")

    class ReturnsAtOnceOnBranch0:
        kill_running, terminate = inner.kill_running, inner.terminate

        def __call__(self, prompt, *, cwd, target):
            if target.endswith("attempt_b000_a000"):
                deadline = time.time() + 5.0
                while not grandchild_pids(pid_dir) and time.time() < deadline:
                    time.sleep(0.02)  # until branch 1's stand-in has forked
                return {"returncode": 0}
            return inner(prompt, cwd=cwd, target=target)

    q = LiveQuestion(
        make_task(str(tmp_path)),
        ReturnsAtOnceOnBranch0(),
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=0,
    )
    started = time.time()
    with pytest.raises(FileExistsError):
        q.probe_batch(q.legal_roots())
    assert time.time() - started < 2.0  # killed, not the 3 s timeout
    [pid] = grandchild_pids(pid_dir)
    assert process_gone(pid)
    assert q.cells == {}
    later = threading.Thread(  # the agent is not closed: a later call still spawns
        target=lambda: inner("p", cwd=str(tmp_path), target=str(tmp_path)), daemon=True
    )
    later.start()
    deadline = time.time() + 5.0
    while len(grandchild_pids(pid_dir)) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert len(grandchild_pids(pid_dir)) == 2
    inner.terminate()
    later.join(timeout=2.0)
    assert not later.is_alive()


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
    # one slice at most (the escapee sleeps 3 s); a second slice would be a regression
    assert not in_flight.is_alive() and time.time() - started < 1.5
    assert results == [{"returncode": None, "stdout": "", "stderr": "agent terminated"}]


# A CLI that forks a setsid escapee holding its pipes (marker as above), prints, and exits 5.
EXITS_BEHIND_AN_ESCAPEE = (
    "import subprocess, sys\n"
    "subprocess.Popen([sys.executable, '-c', 'import os, pathlib, sys, time; os.setsid(); "
    "pathlib.Path(sys.argv[1]).touch(); time.sleep(5)', sys.argv[2]])\n"
    "print('cli done')\n"
    "sys.exit(5)\n"
)


def test_a_finished_agent_whose_escapee_holds_the_pipes_keeps_its_exit_status(tmp_path):
    """A descendant outside the group is not killed, by design; once the CLI has exited it must not
    hold the call until the timeout either: within about two seconds the call returns the CLI's
    real exit status, with the output the escapee holds dropped, not a timed_out report."""
    marker = tmp_path / "escapee-forked"
    agent = CommandAgent(
        [sys.executable, "-c", EXITS_BEHIND_AN_ESCAPEE, "{prompt}", str(marker)],
        timeout=10.0,
        kill_grace=0.2,
    )
    started = time.time()
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 3.0  # neither the 10 s timeout nor the escapee's 5 s
    assert result == {
        "returncode": 5,
        "stdout": "",
        "stderr": "agent exited, but a process outside its group holds its output; output dropped",
    }
    assert marker.exists()


# A CLI whose group holds only a zombie by the time the CLI exits: X forks Y, Y exits at once, X
# leaves the group with setsid and never reaps Y in time. On Darwin, signalling such a group fails
# with EPERM rather than ESRCH.
ZOMBIE_LEFT_IN_GROUP = (
    "import os, sys, time\n"
    "devnull = os.open(os.devnull, os.O_RDWR)\n"
    "if os.fork() == 0:\n"
    "    os.dup2(devnull, 1)\n"
    "    os.dup2(devnull, 2)\n"
    "    if os.fork() == 0:\n"
    "        os._exit(0)\n"
    "    os.setsid()\n"
    "    time.sleep(3)\n"
    "    os._exit(0)\n"
    "print('cli done')\n"
    "sys.exit(0)\n"
)


def test_a_zombie_left_in_the_group_does_not_fail_a_finished_call(tmp_path):
    """Sweeping the group after the CLI exited must not turn a successful attempt into a crashed
    one because the only member left is a zombie nothing can signal."""
    agent = CommandAgent([sys.executable, "-c", ZOMBIE_LEFT_IN_GROUP, "{prompt}"], timeout=5.0)
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert result == {"returncode": 0, "stdout": "cli done\n", "stderr": ""}


class _CountsTerminations:
    """An agent stub whose attempts return at once; records terminate() calls."""

    def __init__(self):
        self.terminated = 0

    def __call__(self, prompt, *, cwd, target):
        return {"returncode": 0}

    def terminate(self):
        self.terminated += 1


def _two_root_question(tmp_path, agent) -> LiveQuestion:
    tree = tmp_path / "tree"
    tree.mkdir()
    return LiveQuestion(
        make_task(str(tmp_path)),
        agent,
        str(tree),
        str(tmp_path / "hist"),
        1.0,
        max_parallelism=2,
        branch_count=2,
        refine_count=0,
    )


def test_an_interrupt_between_two_submits_still_terminates_the_agents(
    tmp_path, stub_prompts, monkeypatch
):
    """The submit loop runs inside the try: an interrupt landing between two submits cancels the
    batch and terminates the agent instead of escaping with the first attempt in flight."""
    agent = _CountsTerminations()
    q = _two_root_question(tmp_path, agent)
    real_submit, submits = concurrent.futures.ThreadPoolExecutor.submit, []

    def interrupts_the_second_submit(pool, fn, *args):
        submits.append(args)
        if len(submits) == 2:
            raise _Interrupted("SIGINT between two submits")
        return real_submit(pool, fn, *args)

    monkeypatch.setattr(
        concurrent.futures.ThreadPoolExecutor, "submit", interrupts_the_second_submit
    )
    with pytest.raises(_Interrupted):
        q.probe_batch(q.legal_roots())
    assert agent.terminated == 1 and q.cells == {}


def test_an_interrupt_during_the_workers_join_still_terminates_the_agents(
    tmp_path, stub_prompts, monkeypatch
):
    """An interrupt that lands in the executor's final join, after every attempt completed,
    terminates the agent before propagating instead of leaving it to the interpreter's exit."""
    agent = _CountsTerminations()
    q = _two_root_question(tmp_path, agent)
    real_shutdown = concurrent.futures.ThreadPoolExecutor.shutdown

    def interrupts_the_join(pool, wait=True, *, cancel_futures=False):
        if wait:
            raise _Interrupted("SIGINT during the join")
        return real_shutdown(pool, wait, cancel_futures=cancel_futures)

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "shutdown", interrupts_the_join)
    with pytest.raises(_Interrupted):
        q.probe_batch(q.legal_roots())
    assert agent.terminated == 1 and q.cells == {}


def test_a_finished_agent_whose_child_holds_the_pipes_returns_its_real_result(
    tmp_path, process_gone
):
    """A CLI that exits while a process it forked still holds its stdout neither makes the call
    wait for that process nor turns into a timeout: the rest of the group is killed within a
    second and the CLI's own exit status and output come back."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    holds_the_pipes = [
        "sh",
        "-c",
        'sleep 30 & echo $! > "$0/child.pid"; echo done; exit 7',
        str(pid_dir),
        "{prompt}",
    ]
    agent = CommandAgent(holds_the_pipes, timeout=5.0, kill_grace=0.2)
    started = time.time()
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 3.0  # neither the 5 s timeout nor the 30 s sleep
    assert result == {"returncode": 7, "stdout": "done\n", "stderr": ""}
    assert process_gone(int((pid_dir / "child.pid").read_text()))


def test_members_left_behind_by_a_finished_agent_are_killed(tmp_path, process_gone):
    """A process the CLI forked and left running with the pipes closed (a server it started)
    does not outlive the call: the group is swept once the CLI has exited normally."""
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    leaves_a_daemon = [
        "sh",
        "-c",
        'sleep 30 >/dev/null 2>&1 & echo $! > "$0/child.pid"; exit 0',
        str(pid_dir),
        "{prompt}",
    ]
    agent = CommandAgent(leaves_a_daemon, timeout=5.0, kill_grace=0.2)
    started = time.time()
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert time.time() - started < 1.0  # nothing held the pipes: EOF at once
    assert result == {"returncode": 0, "stdout": "", "stderr": ""}
    assert process_gone(int((pid_dir / "child.pid").read_text()))


def test_a_half_written_multibyte_character_does_not_crash_the_call(tmp_path):
    """An agent killed or exiting between the bytes of one character must not raise out of the
    call: its output is diagnostic text, decoded with replacement."""
    agent = CommandAgent(["sh", "-c", "printf '\\342\\202'; exit 3", "{prompt}"], timeout=3.0)
    result = agent("p", cwd=str(tmp_path), target=str(tmp_path))
    assert result == {"returncode": 3, "stdout": "�", "stderr": ""}
