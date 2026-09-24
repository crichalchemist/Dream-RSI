"""Policy files are loaded by path; a non-Python path must fail with a clear error."""

import pytest

from see.loader import load_policy
from see.policy.api import LLMDesignedMethod

POLICY = """
from see.policy.api import LLMDesignedMethod

NAME = "Probe"


class Probe(LLMDesignedMethod):
    pass
"""


def test_non_python_policy_path_fails_loudly(tmp_path):
    bad = tmp_path / "policy.txt"
    bad.write_text(POLICY)
    with pytest.raises(ValueError, match=r"policy\.txt"):
        load_policy(str(bad))


def test_load_policy_returns_the_named_class(tmp_path):
    good = tmp_path / "method.py"
    good.write_text(POLICY)
    cls = load_policy(str(good))
    assert cls.__name__ == "Probe" and issubclass(cls, LLMDesignedMethod)
