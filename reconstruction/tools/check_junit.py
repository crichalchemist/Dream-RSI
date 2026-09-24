"""Fail when a pytest JUnit report has skips, errors, failures, or the wrong number of tests.

    python tools/check_junit.py report.xml --expect 47

CI runs this after pytest so a test that quietly skips (for example because
generated/ is missing) can never turn the build green.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

COUNTS = ("tests", "skipped", "errors", "failures")


def summarize(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return {k: sum(int(s.get(k, 0)) for s in suites) for k in COUNTS}


def problems(summary, expect):
    found = [f"{k}={summary[k]}" for k in ("skipped", "errors", "failures") if summary[k]]
    if summary["tests"] != expect:
        found.append(f"tests={summary['tests']} (expected {expect})")
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("report")
    ap.add_argument("--expect", type=int, required=True, help="exact number of tests expected")
    args = ap.parse_args(argv)
    found = problems(summarize(args.report), args.expect)
    print("junit: " + (", ".join(found) if found else f"{args.expect} tests, no skips"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
