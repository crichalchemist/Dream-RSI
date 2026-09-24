"""CI must go red on a skipped test, a failed test, or a test count that quietly changed."""

import os

from see.loader import load_module_from_path

TOOL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "check_junit.py"
)
cj = load_module_from_path("check_junit_under_test", TOOL)
XML = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
    '<testsuite name="pytest" errors="0" failures="{failures}" skipped="{skipped}" tests="{tests}">'
    "</testsuite></testsuites>"
)


def _report(tmp_path, **counts):
    path = tmp_path / "report.xml"
    path.write_text(XML.format(**counts))
    return str(path)


def test_a_skipped_test_fails_the_gate(tmp_path):
    report = _report(tmp_path, tests=38, skipped=5, failures=0)
    assert cj.main([report, "--expect", "38"]) == 1


def test_a_failed_test_fails_the_gate(tmp_path):
    report = _report(tmp_path, tests=38, skipped=0, failures=1)
    assert cj.main([report, "--expect", "38"]) == 1


def test_a_clean_report_with_the_expected_count_passes(tmp_path):
    report = _report(tmp_path, tests=38, skipped=0, failures=0)
    assert cj.main([report, "--expect", "38"]) == 0


def test_an_unexpected_test_count_fails_the_gate(tmp_path):
    report = _report(tmp_path, tests=37, skipped=0, failures=0)
    assert cj.main([report, "--expect", "38"]) == 1
