"""Load a policy file (``method.py``) the way the loop and the evaluator do."""
import hashlib
import importlib.util
import os
import sys

from see.policy.api import LLMDesignedMethod


def load_policy(path: str):
    """Return the class named by the module-level ``NAME`` in ``path``."""
    path = os.path.abspath(path)
    key = hashlib.sha1(path.encode()).hexdigest()[:10]
    spec = importlib.util.spec_from_file_location(f"see_policy_{key}", path)
    mod = importlib.util.module_from_spec(spec)
    saved, sys.dont_write_bytecode = sys.dont_write_bytecode, True  # keep archives clean
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
    name = getattr(mod, "NAME", None)
    cls = getattr(mod, name, None) if isinstance(name, str) else None
    if not (isinstance(cls, type) and issubclass(cls, LLMDesignedMethod)):
        raise TypeError(f"{path}: NAME must name an LLMDesignedMethod subclass, got {name!r}")
    return cls


def overrides_plan_grid(cls) -> bool:
    """Listing 2 requires every proposed policy to override the plan_grid stub."""
    return cls.plan_grid is not LLMDesignedMethod.plan_grid
