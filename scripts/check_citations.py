"""Check the citation bookkeeping of the repository.

  * every key cited in README.md, docs/*.md, supplementary/*.tex, the notebook builder and the source files
    (as \cite{key}, [key], or `key` in a "source" field) exists in docs/references.bib;
  * every entry of the bibliography has a "STATUS:" note (READ-FULL / READ-ABSTRACT / LISTING / FROM-RECORD);
  * every entry of the bibliography is cited somewhere (no orphan references);
  * references that could not be verified at all are not in the bibliography or in the text.
Usage: python scripts/check_citations.py   (exit status 1 if any problem is found)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FORBIDDEN = ("Chua", "Garcia-Santiago", "Garcia Santiago")     # could not be verified in this work


def parse_bib(text: str) -> dict:
    entries = {}
    for m in re.finditer(r"@\w+\{([^,\s]+),(.*?)\n\}", text, flags=re.S):
        entries[m.group(1)] = m.group(2)
    return entries


def check(root: Path) -> list[str]:
    problems = []
    bib_path = root / "docs" / "references.bib"
    bib = parse_bib(bib_path.read_text())
    if not bib:
        return ["no entries parsed from docs/references.bib"]
    for k, body in bib.items():
        if "STATUS:" not in body:
            problems.append(f"bib entry {k} has no STATUS note")
    files = [root / "README.md", *sorted((root / "docs").glob("*.md")), *sorted((root / "supplementary").glob("*.tex")),
             *sorted((root / "src" / "vcsel_bo").glob("*.py")),
             *[f for f in sorted((root / "scripts").glob("*.py")) if f.name not in ("check_citations.py", "make_references_md.py")], root / "CITATION.cff"]
    cited = set()
    for f in files:
        if not f.exists():
            continue
        t = f.read_text()
        for m in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", t):
            for k in m.group(1).split(","):
                cited.add(k.strip())
                if k.strip() not in bib:
                    problems.append(f"{f.relative_to(root)}: \\cite{{{k.strip()}}} not in bibliography")
        for k in bib:
            # a bare key in prose / docstrings counts as a citation, except in the two files that merely list everything
            if f.name in ("REFERENCES.md", "AUDIT.md"):
                continue
            if re.search(rf"\[{re.escape(k)}(?:[;,\]])|\b{re.escape(k)}\b", t) and f.suffix != ".tex":
                cited.add(k)
        for word in FORBIDDEN:
            if f.name in ("REFERENCES.md", "AUDIT.md"):
                continue                                                # the audit trail is allowed to name what was removed
            for line in t.splitlines():
                if word in line and not re.search(r"removed|dropped|could not be verified|unverif", line):
                    problems.append(f"{f.relative_to(root)}: mentions unverifiable reference '{word}' outside a removal note")
    for k in bib:
        if k not in cited:
            problems.append(f"bib entry {k} is never cited")
    return problems


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    probs = check(root)
    for p in probs:
        print("PROBLEM:", p)
    print("citation check:", "OK" if not probs else f"{len(probs)} problem(s)")
    sys.exit(1 if probs else 0)
