"""Regenerate the paper's three code listings from papers/Dream-RSI.pdf.

    Listing 1 -> generated/exploration_prompt.md
    Listing 2 -> generated/policy_improvement_prompt.md
    Listing 3 -> generated/lasso_path_dream_rsi.py   (Appendix C solver)

The listings are the paper's text (c) 2026 Google, so they are regenerated
locally from the PDF already in this repository instead of being committed.

How the recovery works: listing line numbers are set in XCharter-Roman 5.98pt
and code in LMMono8-Regular 8.77pt on a fixed character grid, so each glyph's
column is round((x - X0) / CW), which restores indentation and alignment
exactly. A row that carries code but no line number is a soft wrap of the
previous numbered line and is joined back onto it. The join inserts a space,
except before closing punctuation, where the wrap split a token. Every wrap in
the C++ listing falls on a token boundary, so the space is inert there.

Glyph normalisation: LaTeX listings renders ` as U+2018 and ' as U+2019, and
the font ligates "--" into U+2013, so "---" in the source appears as U+2013
followed by "-". These are mapped back to ASCII. The two symbol-font glyphs in
Listing 2 (U+00D7 in "branch x attempt", U+2192 in "parent -> child") are kept.

Usage: python tools/extract_listings.py [--pdf PATH] [--out DIR]
"""

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PDF = os.path.join(os.path.dirname(ROOT), "papers", "Dream-RSI.pdf")
DEFAULT_OUT = os.path.join(ROOT, "generated")

X0, CW = 76.34, 4.6575  # x origin of code column 0 and monospace advance (pt)
NUM_FONT, NUM_SIZE, CODE_FONT = "XCharter-Roman", 5.98, "LMMono8-Regular"
FIRST_LISTING_PAGE = 17  # 0-based; Appendix B starts on page 18
NO_SPACE_BEFORE = tuple("‘’?,.;:)]")
GLYPHS = {"‘": "`", "’": "'", "“": '"', "”": '"', "–": "--", "−": "-", "˜": "~", "∼": "~"}
OUTPUTS = ["exploration_prompt.md", "policy_improvement_prompt.md", "lasso_path_dream_rsi.py"]
EXPECTED_LINES = [28, 273, 847]


def _rows(page):
    rows = collections.defaultdict(list)
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                rows[round(span["origin"][1], 1)].append(span)
    return [rows[y] for y in sorted(rows)]


def _render(spans):
    grid = {}
    for span in spans:
        for ch in span["chars"]:
            col = round((ch["origin"][0] - X0) / CW)
            if col < 0:
                if ch["c"].strip():
                    raise ValueError(f"glyph left of the code column: {ch['c']!r}")
                continue
            grid[col] = ch["c"]
            # a symbol-font glyph can span two columns; don't pad it with a space
            width = round((ch["bbox"][2] - ch["bbox"][0]) / CW)
            for extra in range(col + 1, col + width):
                grid.setdefault(extra, "")
    if not grid:
        return ""
    return "".join(grid.get(i, " ") for i in range(max(grid) + 1)).rstrip()


def extract(pdf_path):
    import pymupdf

    doc = pymupdf.open(pdf_path)
    listings, current, last = [], None, None
    for page_index in range(FIRST_LISTING_PAGE, doc.page_count):
        for row in _rows(doc[page_index]):
            nums = [
                s
                for s in row
                if s["font"] == NUM_FONT and round(s["size"], 2) == NUM_SIZE and s["origin"][0] < X0
            ]
            code = []
            if any(s["font"] == CODE_FONT for s in row):
                # keep symbol-font glyphs (x, ->) set inline in the listing
                code = [s for s in row if s not in nums]
            if nums:
                number = int("".join(c["c"] for c in nums[0]["chars"]).strip())
                if number == 1:
                    current = {}
                    listings.append(current)
                if number in current:
                    raise ValueError(f"line {number} seen twice (page {page_index + 1})")
                current[number] = _render(code)
                last = number
            elif code and current is not None:
                cont = _render(code).strip()
                sep = "" if cont.startswith(NO_SPACE_BEFORE) else " "
                current[last] = current[last] + sep + cont
    texts = []
    for listing in listings:
        top = max(listing)
        missing = [i for i in range(1, top + 1) if i not in listing]
        if missing:
            raise ValueError(f"listing is missing lines {missing}")
        text = "\n".join(listing[i] for i in range(1, top + 1)) + "\n"
        for glyph, ascii_ in GLYPHS.items():
            text = text.replace(glyph, ascii_)
        texts.append(text)
    return texts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    texts = extract(args.pdf)
    if [t.count("\n") for t in texts] != EXPECTED_LINES:
        sys.exit(
            f"unexpected listing sizes {[t.count(chr(10)) for t in texts]}; "
            f"expected {EXPECTED_LINES} (different PDF build?)"
        )
    os.makedirs(args.out, exist_ok=True)
    for name, text in zip(OUTPUTS, texts):
        with open(os.path.join(args.out, name), "w") as f:
            f.write(text)
    non_ascii = sorted({c for t in texts for c in t if ord(c) > 126})
    print(
        json.dumps(
            {
                "out": args.out,
                "files": OUTPUTS,
                "lines": EXPECTED_LINES,
                "non_ascii_left": non_ascii,
            }
        )
    )


if __name__ == "__main__":
    main()
