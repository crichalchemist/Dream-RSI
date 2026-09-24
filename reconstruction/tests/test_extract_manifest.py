"""The committed digest manifest catches any drift in the paper's regenerated listings."""

import os

from see.loader import load_module_from_path

TOOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "extract_listings.py"
)
el = load_module_from_path("extract_listings_under_test", TOOL)
TEXTS = ["one\n", "two\n", "three\n"]


def test_manifest_round_trips_digests(tmp_path):
    path = str(tmp_path / "generated.sha256")
    el.write_manifest(TEXTS, path)
    assert el.read_manifest(path) == el.digests(TEXTS)
    assert open(path).read().splitlines()[0].endswith("  exploration_prompt.md")


def test_check_reports_nothing_when_text_matches_the_manifest(tmp_path):
    path = str(tmp_path / "generated.sha256")
    el.write_manifest(TEXTS, path)
    assert el.check_manifest(TEXTS, path) == []


def test_check_names_the_listing_whose_text_changed(tmp_path):
    path = str(tmp_path / "generated.sha256")
    el.write_manifest(TEXTS, path)
    tampered = [TEXTS[0], TEXTS[1], TEXTS[2] + " "]
    assert el.check_manifest(tampered, path) == ["lasso_path_dream_rsi.py"]
