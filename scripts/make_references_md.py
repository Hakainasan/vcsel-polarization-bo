"""Write docs/REFERENCES.md (human-readable reference log) from docs/references.bib, so that the verification status shown
to a reader is exactly the one stored with each bibliography entry.  Usage: python scripts/make_references_md.py"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bib = (ROOT / "docs" / "references.bib").read_text()


def field(body: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*\{{(.*?)\}}\s*(?:,|\n\}}|$)", body, flags=re.S | re.M)
    if not m:
        m = re.search(rf"\b{name}\s*=\s*\{{(.*)\}}", body)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def clean(s: str) -> str:
    s = s.replace("{\\\"o}", "ö").replace("{\\'e}", "é").replace("{\\'\\i}", "í").replace("{\\\"", "").replace("\\&", "&")
    return re.sub(r"[{}]", "", s)


rows = []
for m in re.finditer(r"@(\w+)\{([^,\s]+),(.*?)\n\}", bib, flags=re.S):
    kind, key, body = m.groups()
    names = []
    for a in clean(field(body, "author")).split(" and "):
        a = a.strip()
        if a == "others":
            names.append("others")
            continue
        parts = [x.strip() for x in a.split(",")]
        last, first = (parts[0], parts[-1]) if len(parts) > 1 else (a.split()[-1], " ".join(a.split()[:-1]))
        init = " ".join(w[0] + "." for w in re.split(r"[\s-]+", first) if w)
        names.append(f"{init} {last}".strip())
    authors = ", ".join(names) if (len(names) <= 4 and "others" not in names) else f"{names[0]} et al."
    title = clean(field(body, "title"))
    venue = clean(field(body, "journal") or field(body, "booktitle") or field(body, "howpublished") or "")
    vol = field(body, "volume")
    pages = field(body, "pages").replace("--", "-")
    year = field(body, "year")
    note = clean(field(body, "note"))
    st = re.search(r"STATUS:\s*([A-Z-]+)", note)
    status = st.group(1) if st else "?"
    where = ", ".join(x for x in (venue, vol and f"vol. {vol}", pages and f"p. {pages}", year) if x)
    rows.append(f"| `{key}` | {authors}, \"{title}\", {where} | **{status}** | {note.strip()} |")

text = f"""# Reference verification log (generated from `docs/references.bib` by `scripts/make_references_md.py`)

Status legend
* **READ-FULL**: the full text (or, where noted, the main text) was opened and read in this work.
* **READ-ABSTRACT**: title, authors and abstract were read on the publisher page; the full text was not.
* **LISTING**: title, venue and year were confirmed on publisher / index / repository pages; the paper itself was not opened.
* **FROM-RECORD**: identifiers taken from an earlier record of this work and not re-confirmed.

| key | reference | status | note |
|---|---|---|---|
{chr(10).join(rows)}

## What was removed and why
* "Chua et al." (anisotropic oxide apertures, Appl. Phys. Lett., late 1990s) and "Garcia-Santiago, Burger, Rockstuhl, Schneider, J. Lightwave Technol. 39 (2021)" appeared in the first version of this project.
  Neither could be confirmed in this work (no title, volume or page could be found) and both were removed from the code, the text and the bibliography.
  A test (`tests/test_citations.py`) fails if either name reappears outside a removal note.
* Two quotations of the first version ("seldom switch polarization", "maintained stable x-polarization") could not be found in a second extraction pass of the sources and were removed from the text.
  An independent reference check later located the wording "seldom switch polarization" in the Fig. 2(a) caption of the CLEO paper (not re-verified by me), so that quotation was probably right; the anchors use the wording that was found in the second pass.
  Any quotation you reuse should be checked against the PDF: the quoted fragments were taken from arXiv HTML text, not compared character by character with the PDFs.
* A statement that the spin-flip constants of the model are those of a table of the APL 126 paper could not be re-verified in the last pass; the model documents them as typical magnitudes and uses the constants of the metasurface paper as a robustness check.
* The polarization-splitting range attributed to Choquette et al. was corrected to what the paper states (below 1.3 GHz up to 9.9 GHz in individual devices).

## Sources that would improve the study but were not read in full
* Yuan et al., J. Lightwave Technol. 43, 6819 (2025) (abstract only): contains the statistics of hundreds of fabricated tailored-aperture VCSELs; with the per-design numbers the qualitative anchors of the model could become quantitative.
* Yuan et al., Appl. Phys. Lett. 127, 221102 (2025) (listing only): the polarization-switchable arrays for Ising computing.

## Novelty check
Searches for Bayesian-optimisation, surrogate-based or portfolio design of VCSEL oxide-aperture *shapes* for polarization switching found the experimental papers above and, for machine learning on VCSELs, a neural-network surrogate for emission properties (`belonovskii2025`); none treats this problem.
That is absence of evidence in a limited search, **not** proof of novelty. The closest paper (`yuan2025jlt`) was read as an abstract only; whether its full text contains a calibrated model of aperture shape and switching is unknown, so no novelty claim is made.
The arXiv identifiers of the 2026 preprints should be confirmed on arxiv.org before submission.

## Independent check of the reference list
An independent check of the 28 entries (existence, authors, venue, pages) found every entry to exist. Corrections made afterwards: JMLR pages of `desautels2014` (4053-4103), first name in `belonovskii2025` (Aleksei), type and venue of `yuan2025mesa` (2025 IEEE Photonics Conference, from the check; the venue is not printed on the arXiv page that was read here), arXiv version noted for `yuan2025apl126`, pages and issue of `martinregalado1997` and `willemsen2000`, diacritics in `molesky2018`.
Not corrected because the source is missing: the full author list of `yuan2025apl127` (the check found seven authors; this project lists the first author and "others").
The numbers taken from the metasurface paper (`chen2026meta`: flip ratios 0.0025 and 0.07, "28 times", 3-bit accuracy 62 % to 30 %) come from the PDF you supplied and were not re-verified in the last pass; check them against the PDF before quoting.
"""
(ROOT / "docs" / "REFERENCES.md").write_text(text)
print(f"wrote docs/REFERENCES.md with {len(rows)} entries")
