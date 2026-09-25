"""Shared test fixtures for reconstruction/tests."""

import os
import time

import pytest

from see import prompts

EXPLORATION_PROMPT_STUB = (
    "$node_dir $history_dir $baseline_dir $eval_program $problem_file $direction_guidance\n"
)
POLICY_IMPROVEMENT_PROMPT_STUB = "{method_file} {history_dir} {trace_pool}\n"


@pytest.fixture(scope="module")
def stub_prompts(tmp_path_factory):
    """Point see.prompts.GENERATED at stub Listing 1/2 prompts so loop tests never skip.

    Writes each of exploration_prompt's six $-placeholders and
    policy_improvement_prompt's three brace placeholders exactly once, then
    monkeypatches see.prompts.GENERATED (the only place that name is read,
    see/prompts.py:15) to the stub directory. The scripted toy agents
    (see.toy.ScriptedDiscoveryAgent, see.toy.ScriptedPolicyAgent) never read
    the prompt text they are handed, so the stub content itself is never
    asserted on — only its existence and placeholder syntax matter.

    Module-scoped (not function-scoped tmp_path) so a module-scoped consumer
    like tests/test_loop.py's finished_loop fixture can request it; the
    built-in monkeypatch fixture is function-scoped only, so the patch is
    applied and undone through pytest.MonkeyPatch.context() by hand instead.
    """
    directory = tmp_path_factory.mktemp("generated")
    (directory / "exploration_prompt.md").write_text(EXPLORATION_PROMPT_STUB)
    (directory / "policy_improvement_prompt.md").write_text(POLICY_IMPROVEMENT_PROMPT_STUB)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prompts, "GENERATED", str(directory))
        yield


@pytest.fixture
def process_gone():
    """``process_gone(pid)``: True once no process with that pid exists, polling up to 2 s.

    The shell stand-ins in tests/test_command_agent.py and the hanging sweep in
    tests/test_loop.py fork a ``sleep``; the tests assert that grandchild is dead.
    """

    def gone(pid: int, seconds: float = 2.0) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.02)
        return False

    return gone
