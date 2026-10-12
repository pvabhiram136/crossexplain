"""j35 - structural checks on the journal LaTeX source.

pdflatex reports undefined references and citations, but says nothing about
orphan bibliography entries, duplicate labels, or \\input files that exist on
disk but are never pulled in. Those are the failure modes that survive a clean
build and reach a reviewer.

Checks:
  orphan bibitems      defined in the bibliography, never cited
  missing citations    cited, no bibitem  (pdflatex catches these too)
  dangling refs        \\ref or \\eqref with no matching \\label
  duplicate labels     the same label defined twice
  orphan section files sections/*.tex never \\input by the main document
  unverified citations flagged in the source as needing a check
"""
import re
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

PAPER = JROOT / "paper"
MAIN = PAPER / "crossexplain_journal.tex"
SECTIONS = PAPER / "sections"

# Citations still to be checked against the published record. All 21 entries
# in both papers were verified on 23 Aug 2026; see j36_citations.py for what
# was checked against what source. Add a key here if an entry is edited and
# needs re-checking.

UNVERIFIED = []


def main():
    J.banner("j35  structural check of the journal LaTeX")
    main_src = MAIN.read_text(encoding="utf-8")
    sec_files = sorted(SECTIONS.glob("*.tex"))
    body = main_src + "".join(p.read_text(encoding="utf-8") for p in sec_files)

    problems = []

    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", main_src))
    cited = set()
    for m in re.findall(r"\\cite\{([^}]+)\}", body):
        cited |= {c.strip() for c in m.split(",")}
    orphan = sorted(defined - cited)
    missing = sorted(cited - defined)
    print(f"  bibliography: {len(defined)} entries, {len(cited)} cited")
    if orphan:
        problems.append(f"orphan bibitems: {orphan}")
    if missing:
        problems.append(f"cited but undefined: {missing}")

    labels = re.findall(r"\\label\{([^}]+)\}", body)
    label_set = set(labels)
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
    dangling = sorted(refs - label_set)
    dup = sorted({l for l in labels if labels.count(l) > 1})
    print(f"  labels: {len(label_set)} defined, {len(refs)} referenced")
    if dangling:
        problems.append(f"dangling refs: {dangling}")
    if dup:
        problems.append(f"duplicate labels: {dup}")

    # \input can appear in the main document or inside a section file, so
    # scan the whole body rather than just the main document.
    inputs = set(re.findall(r"\\input\{sections/([^}]+)\}", body))
    on_disk = {p.stem for p in sec_files}
    never = sorted(on_disk - {i.replace(".tex", "") for i in inputs})
    print(f"  sections: {len(on_disk)} files on disk, {len(inputs)} inputs")
    if never:
        problems.append(f"section files never \\input: {never}")

    # ---- flattened-text regression guards -------------------------------
    # Every check above is line-oriented, and LaTeX wraps prose at 79 columns,
    # so a claim split across a line break is invisible to grep. That is
    # exactly how a miscounted method list survived a full read-through in
    # Section I after being fixed in the abstract, Section VII and Section
    # VIII. Flatten whitespace before matching.
    flat = re.sub(r"\s+", " ", body)
    guards = [
        (r"Random Forest over ten aggregate statistics, and a few hundred "
         r"lines of hand-written CDC rules with no learned parameters all "
         r"reach the same score",
         "lists the Random Forest among methods reaching 1.0000; it "
         "reaches 0.9963"),
        # Section XI quotes this phrase when confessing the error, so match
        # only unquoted uses. A guard that fires on the confession would
        # train the reader to ignore it.
        (r"(?<!``)statistically indistinguishable(?!'')",
         "claims a significance test the paper explicitly declines to run"),
        # Matched only the Section IV wording at first, so the identical
        # overstatement in Section V survived. Match the quantifier itself.
        (r"(?<!``)almost no (third-party|real) core",
         "the clk naming convention holds for 6 of 16, not 'almost no'"),
        (r"The method that tops the benchmark",
         "six methods are tied at the top; there is no unique top"),
    ]
    for pattern, why in guards:
        if re.search(pattern, flat):
            problems.append(f"regression: {why}")
    # This line used to print a hardcoded 4 while the list beside it grew.
    # A checker that misreports its own scope is the subject of Section XI-D.
    print(f"  flattened-text guards: {len(guards)} patterns checked")

    # Unfilled placeholders. The artifact URL is the one every comparable
    # paper in this area states (TrojanSAINT and GNN-RE both announce their
    # release in the abstract) and the one this paper would look worst for
    # omitting. It must not ship as a placeholder.
    placeholders = [tok for tok in ("ARTIFACT-URL-PENDING", "TODO", "XXX",
                                    "FIXME", "\\todo{")
                    if tok in body]
    print(f"  unfilled placeholders: {len(placeholders)}")
    for tok in placeholders:
        problems.append(
            f"placeholder still in the LaTeX: {tok!r} -- fill it before "
            f"submission")

    # Stripped-backslash guard. This shell's heredocs eat backslashes, and the
    # classic residue is a control sequence with its backslash replaced by the
    # tab that "\t" became: "\texttt{...}" arrives as TAB + "exttt{...}", and
    # "\thanks" as TAB + "hanks". LaTeX compiles both silently, as literal
    # text. It has happened four times in this project, twice reaching a built
    # PDF. No .tex file here indents with tabs, so any tab is the signature.
    tabbed = []
    for p in sec_files + [MAIN]:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if "\t" in line:
                tabbed.append(f"{p.name}:{i}: {line.strip()[:60]}")
    print(f"  tab characters (stripped-backslash signature): {len(tabbed)}")
    for t in tabbed:
        problems.append(f"tab in LaTeX source, likely a lost backslash: {t}")

    still = [c for c in UNVERIFIED if c in defined]
    print(f"  citations flagged as unverified: {len(still)}")
    for c in still:
        print(f"    - {c}")

    J.banner("Result")
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
    else:
        print("  no structural problems")
    if still:
        print(f"\n  {len(still)} citations still need checking against the "
              f"published record\n  before submission. They are marked in the "
              f".tex and listed in this script.")
    J.save("j35_check_tex", dict(
        n_bibitems=len(defined), n_cited=len(cited), orphan=orphan,
        missing=missing, dangling=dangling, duplicate_labels=dup,
        sections_never_input=never, unverified_citations=still,
        problems=problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
