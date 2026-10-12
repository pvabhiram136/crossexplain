"""j50 - measure every table against the column it has to fit in.

A tabular inside a float sets its own natural width. If that width exceeds the
column, LaTeX does not complain: nothing is overfull, because the box is
exactly as wide as it wants to be. It simply prints on top of whatever sits
beside it. This has now happened three times in this paper -- Table IV, then
Table III printing its Origin column across the body text of Section VI, then
two more found by this script -- and every one of them was invisible to the
LaTeX log.

So we measure. For each generated table the tabular is extracted, typeset
inside an \\sbox with the paper's real preamble and table font, and compared
against \\columnwidth for a table and \\textwidth for a table*.

Two details cost an hour each and are worth recording:

  - TeX absorbs the space that terminates a control word, so
    "\\the\\wd\\tmpbox COL" reaches the log as "...ptCOL". The separators below
    are \\space for that reason.
  - the .log wraps at 79 columns, and a wrap can fall mid-token, so the log is
    flattened before matching. This is the same wrapping that defeated the
    line-oriented guards in j35.

The margin is reported, not just pass/fail: a table at 97% of the column is not
a bug today and becomes one as soon as a number gains a digit.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

SEC = JROOT / "paper" / "sections"
MAIN = JROOT / "paper" / "crossexplain_journal.tex"
CONF = JROOT.parent / "paper" / "crossexplain_ieee.tex"

TABULAR = re.compile(r"(?s)(\\begin\{tabular\}.*?\\end\{tabular\})")
STARRED = re.compile(r"\\begin\{table\*\}")
# A float and the tabular inside it, so a table written inline in a single
# document (the conference paper) can be attributed to its own float.
FLOAT = re.compile(
    r"(?s)\\begin\{(table\*?)\}.*?(\\begin\{tabular\}.*?\\end\{tabular\})")
TIGHT = 97.0          # percent of the available width we call "tight"


def collect(main: Path, sec: Path | None):
    """(name, starred, tabular) for every table belonging to one document.

    The journal keeps each table in its own generated file; the conference
    paper writes them inline. Both shapes are measured, because the failure
    mode does not care which file the tabular lives in.
    """
    out = []
    if sec is not None and sec.is_dir():
        for p in sorted(sec.glob("tab_*.tex")):
            src = p.read_text(encoding="utf-8")
            m = TABULAR.search(src)
            if m:
                out.append((p.name, bool(STARRED.search(src)), m.group(1)))
        return out
    src = main.read_text(encoding="utf-8")
    for i, m in enumerate(FLOAT.finditer(src), 1):
        label = re.search(r"\\label\{([^}]+)\}",
                          src[m.start():m.end()])
        name = label.group(1) if label else f"{main.stem}#{i}"
        out.append((name, m.group(1) == "table*", m.group(2)))
    return out


def measure(title: str, main: Path, sec: Path | None, tabfont: str):
    tables = collect(main, sec)
    starred_of = {n: s for n, s, _ in tables}

    body = [r"\begin{document}", r"\newsavebox\tmpbox"]
    for name, _, tab in tables:
        body.append(r"\sbox\tmpbox{\tabfont " + tab + "}")
        body.append(r"\typeout{J50 " + name +
                    r" WIDTH \the\wd\tmpbox\space COL \the\columnwidth"
                    r"\space TEXT \the\textwidth}")
    body += [r"\typeout{J50 DONE}", r"\end{document}"]

    pre = main.read_text(encoding="utf-8").split(r"\begin{document}")[0]
    # The conference paper has no \tabfont; supply whatever that
    # document uses to size a table so the measurement matches the page.
    if "\\newcommand{\\tabfont}" not in pre:
        pre += "\\newcommand{\\tabfont}{" + tabfont + "}\n"
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "measure.tex"
        f.write_text(pre + "\n".join(body), encoding="utf-8")
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "measure.tex"],
            cwd=td, capture_output=True, text=True)
        log = Path(td) / "measure.log"
        out = (log.read_text(encoding="utf-8", errors="replace")
               if log.exists() else proc.stdout)

    flat = re.sub(r"\s+", " ", out)
    if "J50 DONE" not in flat:
        print("  the measurement run did not complete:")
        for line in out.splitlines():
            if line.startswith("!"):
                print(f"    {line}")
        return 1

    rows, problems = [], []
    for m in re.finditer(
            r"J50 (\S+) WIDTH ([\d.]+)pt\s*COL ([\d.]+)pt\s*TEXT ([\d.]+)pt",
            flat):
        name = m.group(1)
        w, col, txt = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        avail = txt if starred_of.get(name) else col
        pct = 100.0 * w / avail
        rows.append((name, starred_of.get(name, False), w, avail, pct))
        if w > avail:
            problems.append(
                f"{name} is {w:.1f}pt wide in a {avail:.1f}pt "
                f"{'text block' if starred_of.get(name) else 'column'} "
                f"({pct:.0f}%): it will print over the text beside it")

    print(f"  {title}: measured {len(rows)} of {len(tables)} tables\n")
    for name, starred, w, avail, pct in sorted(rows, key=lambda r: -r[4]):
        flag = ("OVERFLOW" if pct > 100 else
                "tight" if pct > TIGHT else "ok")
        print(f"  [{flag:>8s}] {name:<26s} {w:7.1f}pt /{avail:7.1f}pt "
              f"{'table*' if starred else '      '} {pct:5.1f}%")

    return dict(
        title=title,
        n_tables=len(rows), n_overflow=len(problems), problems=problems,
        n_tight=sum(1 for r in rows if TIGHT < r[4] <= 100),
        tables=[dict(name=n, starred=st, width_pt=round(w, 1),
                     avail_pt=round(a, 1), pct=round(q, 1))
                for n, st, w, a, q in rows])


def main() -> int:
    J.banner("j50  table widths against the column they must fit")
    if not shutil.which("pdflatex"):
        print("  pdflatex not available; skipping")
        return 0

    # Both documents. The conference paper writes its tables inline and had
    # never been measured; the failure mode is a property of LaTeX, not of
    # this project's journal draft.
    results = [
        measure("journal", MAIN, SEC, r"\footnotesize"),
        measure("conference", CONF, None, r"\footnotesize"),
    ]

    J.banner("Result")
    bad = [r for r in results if r["n_overflow"]]
    for r in results:
        print(f"  {r['title']:<12s} {r['n_tables']} tables, "
              f"{r['n_overflow']} overflowing, {r['n_tight']} tight")
    for r in results:
        for p_ in r["problems"]:
            print(f"  PROBLEM [{r['title']}]: {p_}")
    if not bad:
        print("  every table in both documents fits the column it sits in")

    J.save("j50_table_widths", dict(
        documents=results,
        n_overflow=sum(r["n_overflow"] for r in results)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
