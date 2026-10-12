"""j51 - the reverse-direction number scan, pointed at the conference paper.

The journal got j47: every numeral in the .tex must be derivable from a result
file, asserted by a check, or whitelisted with a reason. The conference paper
never got the equivalent. Its "69 of 69 numbers reconciled" came from
j10_audit_numbers.py, which says of itself, accurately, that it is "not a
checker: it is a worksheet generator" -- a human read the worksheet once and
signed it off. Nothing has re-verified it since, and the paper has been edited
since (the sanchez2020 bibliography fix, and the near-miss scoping fix).

This runs j47's scan against paper/crossexplain_ieee.tex. It reuses j47's
scraper rather than reimplementing it, so the two documents are held to the
same definition of "a numeral in claim position".

The conference paper's numbers predate the corpus merge by design: it reports
the 16-design corpus and the checkpoints as released then. So the derivable set
here is the CONFERENCE result files first, with the journal's files available
too, since several conference numbers were re-derived during the journal work.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
ROOT = JROOT.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

# Import j47 by path: it is a sibling script, not a package module.
_spec = importlib.util.spec_from_file_location(
    "j47", Path(__file__).resolve().parent / "j47_orphan_numbers.py")
j47 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j47)

CONF = ROOT / "paper" / "crossexplain_ieee.tex"
CONF_SINGLE = ROOT / "paper" / "crossexplain_ieee_single.tex"
CONF_RESULTS = ROOT / "results"

# Numerals that are real claims but derivable from no result file. Same rule as
# j47's whitelist: each needs a reason, and a result number belongs in the
# derivation, not here.
WHITELIST = {
    "0": "zero, and the decision threshold's neighbourhood",
    "1": "counts of one, first, enumerations",
    "2": "two-flip-flop synchronizer; two clock domains",
    "3": "three-flip-flop synchronizer; enumerations",
    "4": "four Yosys optimization levels O0-O3",
    "5": "five hazard classes; five seeds; five layers",
    "6": "six pages, six feature groups",
    "0.5": "the decision threshold",
    "10": "ten handcrafted features",
    "32": "float32",
    "64": "hidden width",
    "2002": "citation year (Cummings, async FIFO)",
    "2006": "citation year (Litterick)",
    "2008": "citation year (Cummings, CDC)",
    "2011": "citation year (Ginosar)",
    "2017": "citation year (Kipf, Hamilton)",
    "2018": "citation year (Velickovic)",
    "2020": "citation year (Sanchez-Lengeling)",
    "2021": "citation year (Yasaei)",
    "2022": "citation year (Alrahis, Zhuang)",
    "2023": "citation year (Lashen, Yuan, Real Intent)",
    "2026": "access dates on the commercial-tool citations",
    "0.0.13": "sv2v version string",
    "2005": "Verilog-2005",
}


def derivable() -> set[str]:
    """Every string form of every number in either result tree."""
    out: set[str] = set()

    def add(v):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return
        for d in range(11):
            out.add(f"{abs(f):.{d}f}")
        for d in range(3):
            out.add(f"{abs(f) * 100:.{d}f}")
        out.add(str(int(round(f))))
        out.add(str(int(f)))
        if f != 0.0:
            for sig in range(1, 6):
                out.add(f"{abs(f):.{sig}e}".split("e")[0])

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        else:
            add(o)

    for tree in (CONF_RESULTS, J.RESULTS):
        for p in sorted(tree.glob("*.json")):
            try:
                walk(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
    return out


def main() -> int:
    J.banner("j51  conference-paper numerals against the conference results")

    ok = derivable()
    print(f"  derivable string forms: {len(ok)}")

    orphans = []
    total = 0
    for path in (CONF, CONF_SINGLE):
        if not path.exists():
            continue
        for line, tok, ctx in j47.claim_numbers(path):
            total += 1
            if tok in ok or tok in WHITELIST:
                continue
            alts = {tok.lstrip("0") or "0",
                    tok.rstrip("0").rstrip(".") or "0"}
            if alts & ok:
                continue
            orphans.append((path.name, line, tok, ctx))

    print(f"  scanned {total} numerals in claim position across "
          f"the two conference variants\n")
    if orphans:
        print(f"  {len(orphans)} not derivable and not whitelisted:\n")
        for name, line, tok, ctx in orphans:
            print(f"    {name}:{line}  {tok!r}")
            print(f"        ...{ctx}...")
    else:
        print("  no orphan numbers")

    # The two conference files are byte-identical, and the README documents
    # only the first. "_single" implies a single-column variant that was never
    # actually made: both carry \documentclass[conference]. Whatever the
    # history, every fix has had to be applied twice by hand, which is exactly
    # how the sanchez2020 correction nearly went into only one of them. While
    # both files exist, hold them identical and say so loudly when they drift.
    import hashlib
    same = None
    digests = {}
    if CONF.exists() and CONF_SINGLE.exists():
        for p in (CONF, CONF_SINGLE):
            digests[p.name] = hashlib.md5(p.read_bytes()).hexdigest()[:12]
        same = len(set(digests.values())) == 1
        print()
        if same:
            print(f"  the two conference variants are byte-identical "
                  f"({digests[CONF.name]})")
            print("  -- note: the README documents only "
                  f"{CONF.name}, and both files declare")
            print("     \\documentclass[conference], so '_single' is a "
                  "duplicate, not a layout variant")
        else:
            print("  DRIFT: the two conference variants differ")
            for n, d in digests.items():
                print(f"    {n}  {d}")

    J.save("j51_conference_numbers", dict(
        n_numerals=total, n_orphans=len(orphans),
        variants_identical=same, variant_digests=digests,
        orphans=[dict(file=f, line=l, value=t, context=c)
                 for f, l, t, c in orphans]))
    return 1 if (orphans or same is False) else 0


if __name__ == "__main__":
    raise SystemExit(main())
