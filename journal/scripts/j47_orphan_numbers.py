"""j47 - find numbers in the paper that nothing derives.

j34 runs in the wrong direction. Every one of its checks compares a value read
out of a result JSON against a literal typed into j34 by hand. If a number in
the .tex drifts away from the literal in j34, j34 stays green. That is not a
hypothetical: the caption of Fig. 3 claimed the two models disagree on 97% of
real crossings for an entire corpus revision after the true figure became 87%,
and all 148 checks passed throughout.

This script runs the other direction. It extracts every numeral that appears in
the rendered prose of the paper and requires each one to be *accounted for* --
either derivable from a result JSON, or present in an explicit whitelist with a
stated reason. A number that is neither is reported as an orphan.

The whitelist is the honest part of this script. It is not a place to silence
failures; each entry names why the number is not derivable (it is a citation
year, a hardware fact, a count of sections, a figure geometry). Adding a result
number to the whitelist instead of deriving it would defeat the check, so the
whitelist is kept short enough to read.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

SEC = JROOT / "paper" / "sections"
MAIN = JROOT / "paper" / "crossexplain_journal.tex"


# ------------------------------------------------------------------ derived
# Result files that read the third-party confirmation corpus and were NOT
# regenerated when it grew from 31 crossings to 39. Their numbers were correct
# when written and are wrong for any claim the paper makes about the corpus
# today. Section VIII's "0.806" -- the single-task false-positive rate, 25/31 --
# survived every check because three of these files still contain it, so the
# orphan scan saw a legitimate-looking derivation. A numeral supported ONLY by
# these is reported separately rather than silently accepted.
STALE_CORPUS_SOURCES = {
    "j08_grouping", "j17_p2_train", "j18_idiom_fix",
    "j19_diagnostic", "j21_tighten",
}

# Numerals whose only support is a stale source but which are deliberately
# historical: the paper reports them as facts about the conference pipeline at
# the time, not as current corpus measurements. Each is here because we looked.
HISTORICAL: dict[str, str] = {}


def derivable_strings(include_stale: bool = False) -> set[str]:
    """Every string form of every number reachable from the result JSONs.

    A claim in the paper is 'accounted for' if it appears here. We generate the
    plain rounding at 0-4 decimals, the percentage form, and the integer form,
    because the prose quotes the same underlying quantity in all of those.
    """
    out: set[str] = set()

    def add(v: float) -> None:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            return
        for d in range(11):
            out.add(f"{f:.{d}f}")
            out.add(f"{abs(f):.{d}f}")
        # the same quantity quoted as a percentage
        for d in range(3):
            out.add(f"{f * 100:.{d}f}")
            out.add(f"{abs(f) * 100:.{d}f}")
        # ratios and factors are often quoted rounded to a whole number
        out.add(str(int(round(f))))
        out.add(str(int(f)))
        # Small and large magnitudes are quoted in scientific notation, where
        # only the mantissa reaches the page: 8.67e-7 is typeset as
        # "$8.67\times10^{-7}$". Register the mantissa at 1-5 significant
        # figures so those are derived rather than whitelisted.
        if f != 0.0:
            for sig in range(1, 6):
                m = f"{abs(f):.{sig}e}"
                out.add(m.split("e")[0])

    def walk(o) -> None:
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        else:
            add(o)

    for p in sorted(J.RESULTS.glob("*.json")):
        if p.stem in STALE_CORPUS_SOURCES and not include_stale:
            continue
        try:
            walk(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue

    # Values j34 asserts. Some of its checks compare the paper against a mean
    # or a spread computed over stored values rather than against a stored
    # value, so those numbers appear in no JSON. They are verified -- just by
    # computation rather than by lookup -- and this is how j47 learns that.
    j34 = J.RESULTS / "j34_audit_journal.json"
    if j34.exists():
        for v in json.loads(j34.read_text(encoding="utf-8")).get("asserted", []):
            add(v)

    # Counts of things the paper states about its own corpus and scripts. These
    # are derived here rather than whitelisted, so they cannot drift silently.
    try:
        conf = J.confirmation_set()
        add(len(conf))
        add(len({s["design"] for s in conf}))
    except Exception:
        pass
    return out


# -------------------------------------------------------------- tex scraping
# Constructs whose numbers are typography, not claims.
STRIP = [
    (r"(?m)(?<!\\)%.*$", " "),                    # comments
    # The bibliography is years, volumes and page ranges, none of which are
    # claims about our results. j36_citations.py audits it against the
    # published record, which is the check that belongs there.
    (r"(?s)\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " "),
    # TikZ coordinates and style bodies are geometry. Node *labels* are pulled
    # out before this runs, so claims inside a figure still reach the scan.
    (r"(?s)\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " "),
    (r"\\label\{[^}]*\}", " "),
    (r"\\(?:ref|eqref|cite|citep|includegraphics)\s*(?:\[[^\]]*\])?\{[^}]*\}", " "),
    (r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " "),   # handled apart
    (r"\\[a-zA-Z@]+\s*=\s*[-\d.]+\s*(?:mm|cm|pt|em|ex|in|\\[a-zA-Z]+)", " "),
    (r"[-\d.]+\s*(?:mm|cm|pt|em|ex|in)\b", " "),               # lengths
    (r"[-\d.]+\\(?:columnwidth|textwidth|linewidth|baselineskip)", " "),
    (r"\\begin\{tabular\}\{[^}]*\}", " "),                     # column specs
    (r"\\(?:cmidrule|midrule|toprule|bottomrule)(?:\([^)]*\))?(?:\{[^}]*\})?", " "),
    (r"\\multicolumn\{\d+\}", " "),
    (r"\\(?:hspace|vspace|setlength|addtolength|scalebox|resizebox)\*?\{[^}]*\}", " "),
    (r"\\IEEEauthorrefmark\{\d+\}", " "),
    (r"\\(?:section|subsection|subsubsection)\b", " "),
]

# TikZ bodies are geometry. We keep only the text inside node braces, which is
# where a figure states a claim -- the 97% that started this script lived in a
# caption, and other numbers live in node labels.
TIKZ = re.compile(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", re.S)
NODE_TEXT = re.compile(r"\)\s*(?:\[[^\]]*\]\s*)?\{(.*?)\};", re.S)

# A comma groups thousands ("64,845") only when it is followed by exactly
# three digits. Treating any comma that way glued the two halves of a
# confidence interval written without a space -- "[0.394,0.857]" was read as
# the single numeral "0.3940.857" and reported as an orphan. The journal
# writes "[0.500, 0.872]" with a space, so this only surfaced when the same
# scan was pointed at the conference paper.
NUM = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")

# Section XI reports the numbers that drifted, so it necessarily contains the
# wrong values in quotation. A checker that fires on the confession teaches the
# reader to ignore the checker, so the confession is fenced off by name in the
# source rather than special-cased in here. The fence is deliberately noisy to
# write and trivial to grep for, and it is only ever placed around text that
# quotes a value this paper has already corrected.
QUOTED = re.compile(
    r"(?s)%\s*audit:quote-begin.*?%\s*audit:quote-end")


def unquoted(raw: str) -> str:
    return QUOTED.sub(" ", raw)


def claim_numbers(path: Path) -> list[tuple[int, str, str]]:
    """Return (line, number, context) for every numeral that is a claim."""
    raw = path.read_text(encoding="utf-8")
    # "64{,}845" is one number, not two: LaTeX's thin-space digit grouping has
    # to go before anything counts numerals.
    raw = unquoted(raw).replace("{,}", ",").replace("\\,", "")

    # pull node labels out of tikz first, then drop the picture wholesale
    kept = []
    for body in TIKZ.findall(raw):
        for lab in NODE_TEXT.findall(body):
            kept.append(lab)
    text = raw
    for pat, rep in STRIP:
        text = re.sub(pat, rep, text)
    text = text + "\n" + "\n".join(kept)

    # line numbers for the part that came from the file itself
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in NUM.finditer(line):
            tok = m.group(1).replace(",", "")          # 64,845 -> 64845
            lo = max(0, m.start() - 46)
            out.append((i, tok, line[lo:m.end() + 46].strip()))
    return out


# ------------------------------------------------------------------ allowed
# Numbers that are real claims but are not derivable from a result JSON. Each
# needs a reason. Anything that could be derived should be derived instead.
WHITELIST: dict[str, str] = {
    # --- structural / typographic constants of the document itself
    "0": "zero, used rhetorically and as a threshold",
    "1": "counts of one, enumerations, first",
    "2": "two-flip-flop synchronizer; enumerations",
    "3": "three-stage synchronizer; enumerations; O3",
    "4": "four optimization levels O0-O3; enumerations",
    "5": "five hazard classes; five seeds",
    "0.5": "the decision threshold",
    "32": "float32",
    "64": "float64, hidden width",
    # --- external facts, each attributable to a cited source or to the tools
    "2008": "citation year (Cummings)",
    "2011": "citation year (Ginosar, Torralba)",
    "2006": "citation year (Litterick)",
    "2019": "citation year (Recht)",
    "2021": "citation year (Liao)",
    "2026": "citation year (Akhtar)",
    "2024": "citation year (Shanmugavelu)",
    "2002": "citation year (Higham)",
    "0.0.13": "sv2v version string",
    "2005": "Verilog-2005, the sv2v output standard",
    "3": "BSD-3 licence of sv2v",
}


# --------------------------------------------------------------- bound claims
# The orphan scan above is necessary but not sufficient. A number can be wrong
# in context and still be "accounted for", because some other result happens to
# have that value: the 97% that motivated this script would have passed the
# orphan scan, since 0.974 appears throughout the paper and rounds to 97.
#
# These patterns bind a number to the quantity it is claiming to be. The
# captured group must equal the derived value, wherever in the paper it occurs
# and however the line happens to wrap.
def bound_claims() -> list[tuple[str, str, float, float]]:
    """(name, regex with one numeric group, expected value, tolerance)."""
    j23 = json.loads((J.RESULTS / "j23_pooling_order.json").read_text())
    j37 = json.loads(
        (J.RESULTS / "j37_audit_corpus_history.json").read_text())
    disagree = 1.0 - j23["models"]["legacy"]["agree_with_reference"]
    leg = j23["models"]["legacy"]["conf_fpr"]
    ref = j23["models"]["reference"]["conf_fpr"]
    return [
        ("share of real crossings the two orderings disagree on",
         r"disagree on (\d+(?:\.\d+)?)\\?% of real", disagree * 100, 0.6),
        # The same quantity is stated four times in four different sentence
        # shapes. Binding only one of them would leave the other three free to
        # drift, which is the failure this whole script exists to stop.
        ("the null change's effect on real-world FPR (postfix form)",
         r"false-positive rate by \$?(\d+\.\d+)", abs(leg - ref), 6e-4),
        ("the null change's effect on real-world FPR (prefix form)",
         r"differing by \$?(\d+\.\d+)\$? in real-world false-positive",
         abs(leg - ref), 6e-4),
        ("the null change's effect on real-world FPR (confirmation set)",
         r"differ by \$?(\d+\.\d+)\$? on the", abs(leg - ref), 6e-4),
        # Size of the confirmation set. Two occurrences of this still read
        # "31" after the corpus grew to 39, in Section VIII and Section X,
        # and passed both j34 and the orphan scan -- 31 is a real number
        # elsewhere in the paper, so nothing flagged it out of position.
        ("size of the third-party confirmation set",
         r"(?:the|On the) (\d+)\s+third-party crossings",
         float(j23["n_conf"]), 0.5),
        ("number of designs the confirmation set covers",
         r"design-clustered over (\d+) designs",
         float(j23["n_conf_designs"]), 0.5),
        # Fig. 2's node label and its caption both still said "10 of 16"
        # after the corpus reached 21 designs and the body text had been
        # corrected to 13. Both numbers are real elsewhere in the paper, so
        # only a binding catches it.
        ("designs a name-based clock rule would discard",
         r"name-based rule would drop \\textbf\{(\d+) of \d+\}",
         float(j37["n_name_excluded"]), 0.5),
        ("designs accepted into the corpus (figure node)",
         r"name-based rule would drop \\textbf\{\d+ of (\d+)\}",
         float(j37["n_accepted"]), 0.5),
        # Fig. 2's funnel head read "25 tried" feeding "21 accepted" after a
        # second pass had added the candidates five of those 21 came from.
        # 25 could not produce 21; the head is both passes.
        ("candidates tried, funnel head",
         r"candidate cores\\\\ \\textbf\{(\d+) tried\}",
         float(j37["n_tried_total"]), 0.5),
    ]


def check_bound(files: list[Path]) -> list[str]:
    flat = re.sub(r"\s+", " ", "\n".join(
        unquoted(p.read_text(encoding="utf-8")) for p in files))
    bad = []
    for name, pat, want, tol in bound_claims():
        hits = re.findall(pat, flat)
        if not hits:
            bad.append(f"no occurrence of the claim: {name}")
            continue
        for h in hits:
            if abs(float(h) - want) > tol:
                bad.append(f"{name}: paper says {h}, data gives "
                           f"{want:.4g}")
        print(f"  [{'ok  ' if not bad else 'FAIL'}] bound: {name}"
              f"  ({len(hits)} occurrence(s), want {want:.4g})")
    return bad


def main() -> int:
    J.banner("j47  numbers in the paper that nothing derives")

    ok = derivable_strings()
    # The same set, but admitting the pre-merge corpus files. Anything that is
    # in `stale_ok` and not in `ok` is a numeral the paper can only justify
    # from a measurement taken on the 31-crossing corpus.
    stale_ok = derivable_strings(include_stale=True)
    files = sorted(SEC.glob("*.tex")) + [MAIN]

    orphans: list[tuple[str, int, str, str]] = []
    stale: list[tuple[str, int, str, str]] = []
    total = 0
    for p in files:
        for line, tok, ctx in claim_numbers(p):
            total += 1
            if tok in ok or tok in WHITELIST:
                continue
            # a number written 0.871 is derivable if 0.871 is; also allow the
            # leading-zero-less and trailing-zero variants the prose may use
            alts = {tok.lstrip("0") or "0", tok.rstrip("0").rstrip(".") or "0",
                    "0" + tok if tok.startswith(".") else tok}
            if alts & ok:
                continue
            # Supported only by a pre-merge corpus measurement: not an orphan,
            # but not a current number either. Reported apart so the two cases
            # are never confused.
            if (tok in stale_ok or alts & stale_ok) and tok not in HISTORICAL:
                stale.append((p.name, line, tok, ctx))
                continue
            if tok in HISTORICAL:
                continue
            orphans.append((p.name, line, tok, ctx))

    print(f"  scanned {len(files)} files, {total} numerals in claim position")
    print(f"  derivable string forms available: {len(ok)}\n")
    bad = check_bound(files)
    print()
    if orphans:
        print(f"\n  {len(orphans)} ORPHAN(S) -- not derivable, not whitelisted:\n")
        for name, line, tok, ctx in orphans:
            print(f"    {name}:{line}  {tok!r}")
            print(f"        ...{ctx}...")
    else:
        print("\n  no orphan numbers")

    if stale:
        print(f"\n  {len(stale)} numeral(s) supported ONLY by a corpus "
              f"measurement taken before the merge:\n")
        for name, line, tok, ctx in stale:
            print(f"    {name}:{line}  {tok!r}")
            print(f"        ...{ctx}...")
    else:
        print(f"  no numeral rests only on the {len(STALE_CORPUS_SOURCES)} "
              f"pre-merge corpus files")

    if bad:
        print("  BOUND-CLAIM FAILURES:")
        for b in bad:
            print(f"    {b}")

    J.save("j47_orphan_numbers", dict(
        n_files=len(files), n_numerals=total, n_orphans=len(orphans),
        n_bound=len(bound_claims()), bound_failures=bad,
        n_stale_provenance=len(stale),
        stale_provenance=[dict(file=f, line=l, value=t, context=c)
                          for f, l, t, c in stale],
        orphans=[dict(file=f, line=l, value=t, context=c)
                 for f, l, t, c in orphans]))
    return 1 if (orphans or bad or stale) else 0


if __name__ == "__main__":
    raise SystemExit(main())
