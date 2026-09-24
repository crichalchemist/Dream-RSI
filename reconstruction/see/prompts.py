"""The paper's two prompts (Listings 1 and 2), instantiated for the runtime.

The prompt text is regenerated from the PDF by tools/extract_listings.py
into generated/ (it is the paper's text, so it is not committed).
"""
import os
import string

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED = os.path.join(ROOT, "generated")


def _read(name: str) -> str:
    path = os.path.join(GENERATED, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run `python tools/extract_listings.py` first")
    with open(path) as f:
        return f.read()


def exploration_prompt(*, node_dir, history_dir, baseline_dir, eval_program, problem_file,
                       direction_guidance="") -> str:
    """Listing 1; its placeholders are $-style."""
    return string.Template(_read("exploration_prompt.md")).safe_substitute(
        node_dir=node_dir, history_dir=history_dir, baseline_dir=baseline_dir,
        eval_program=eval_program, problem_file=problem_file,
        direction_guidance=direction_guidance)


def policy_improvement_prompt(*, method_file, history_dir, trace_pool) -> str:
    """Listing 2; its only brace placeholders are these three."""
    text = _read("policy_improvement_prompt.md")
    for key, value in (("method_file", method_file), ("history_dir", history_dir),
                       ("trace_pool", trace_pool)):
        text = text.replace("{" + key + "}", value)
    return text
