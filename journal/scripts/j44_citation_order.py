"""j44 - renumber the bibliography into order of first citation.

IEEE style requires references numbered in the order they are first cited.
`thebibliography` numbers by the order the \\bibitem entries are written, so
appending new references to the end of the list -- which is what happened when
the benchmark-validity and floating-point citations were added -- produces a
paper that cites [22] on page 2 and [20] on page 3.

Nothing warns about this. pdflatex is happy, and the only way to notice is to
read the rendered references against the text.

This reads the document in \\input order, records the order of first citation,
reorders the \\bibitem blocks to match, and reports anything cited but not
defined or defined but never cited.
"""
import re
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

PAPER = JROOT / "paper"
MAIN = PAPER / "crossexplain_journal.tex"


def expand(text, depth=0):
    """Inline \\input{sections/...} so citations are seen in reading order."""
    if depth > 4:
        return text
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\\input\{sections/([^}]+)\}", line)
        if m:
            f = PAPER / "sections" / m.group(1)
            f = f if f.suffix else f.with_suffix(".tex")
            if f.exists():
                out.append(expand(f.read_text(encoding="utf-8"), depth + 1))
                continue
        out.append(line)
    return "\n".join(out)


def main():
    J.banner("j44  bibliography order")
    src = MAIN.read_text(encoding="utf-8")

    bib_start = src.index(r"\begin{thebibliography}")
    bib_end = src.index(r"\end{thebibliography}")
    body = expand(src[:bib_start])
    bib = src[bib_start:bib_end]

    # order of first citation
    seen = []
    for m in re.findall(r"\\cite\{([^}]+)\}", body):
        for k in (c.strip() for c in m.split(",")):
            if k not in seen:
                seen.append(k)

    # split the bibliography into (key, block) pairs
    parts = re.split(r"(\\bibitem\{[^}]+\})", bib)
    head = parts[0]
    entries = {}
    order_now = []
    for i in range(1, len(parts), 2):
        key = re.match(r"\\bibitem\{([^}]+)\}", parts[i]).group(1)
        entries[key] = parts[i] + parts[i + 1].rstrip() + "\n"
        order_now.append(key)

    missing = [k for k in seen if k not in entries]
    orphan = [k for k in order_now if k not in seen]
    print(f"  {len(order_now)} bibitems, {len(seen)} cited")
    if missing:
        print(f"  MISSING bibitem for: {missing}")
    if orphan:
        print(f"  ORPHAN (never cited): {orphan}")

    wrong = [(i + 1, k) for i, k in enumerate(order_now)
             if i < len(seen) and seen[i] != k]
    if not wrong:
        print("  already in order of first citation; nothing to do")
        J.save("j44_citation_order", dict(n=len(order_now), reordered=False,
                                          missing=missing, orphan=orphan))
        return 0

    print(f"  {len(wrong)} entries are out of order; renumbering")
    for i, k in enumerate(seen):
        was = order_now.index(k) + 1
        if was != i + 1:
            print(f"    [{was:2d}] -> [{i + 1:2d}]  {k}")

    new_bib = head + "".join(entries[k] for k in seen)
    new_bib += "".join(entries[k] for k in order_now if k not in seen)
    MAIN.write_text(src[:bib_start] + new_bib + src[bib_end:],
                    encoding="utf-8")
    print(f"  -> rewrote the bibliography of {MAIN.name}")

    J.save("j44_citation_order", dict(n=len(order_now), reordered=True,
                                      order=seen, missing=missing,
                                      orphan=orphan))
    return 0


if __name__ == "__main__":
    sys.exit(main())
