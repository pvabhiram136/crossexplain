"""j52 - assemble the public artifact tree.

Section XI promises an artifact. This builds exactly what that promise covers,
and nothing else, into journal/artifact/release/.

Three decisions worth stating, because each one is a judgement rather than a
mechanical copy:

  1. THIRD-PARTY RTL IS NOT REDISTRIBUTED. journal/corpus/ is 663 MB of cloned
     upstream repositories. Four of them (opentitan, axi, common_cells,
     basejump_stl) ship a LICENSE; others in the tree (i2c, sdram_controller,
     simple_spi) carry none we could find. Redistributing all of that is a
     licensing problem with no upside, so the release ships a fetch script that
     clones each upstream at a PINNED COMMIT instead. Anyone can rebuild the
     corpus; nobody has to take our word for what the sources were.

  2. THE DERIVED NETLISTS ARE SHIPPED, for the accepted designs only. They are
     our output, not upstream's code, and they are what the graphs are built
     from -- without them the results cannot be reproduced without installing
     sv2v and Yosys.

  3. ABANDONED-PHASE DATA IS EXCLUDED. corpus/p2designs, p2probe, sizeidiom
     and idiomdesigns total ~370 MB and belong to the P2 phase the paper
     reports as abandoned. Shipping them would treble the artifact to support
     nothing the paper claims.

The manifest is printed with sizes so the result can be checked against what
Section XI says is released.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
ROOT = JROOT.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

OUT = JROOT / "artifact" / "release"

# (source, destination, what it is). Directories are copied wholesale.
MANIFEST = [
    (ROOT / "crossexplain", "crossexplain", "generator, model, training, eval"),
    (JROOT / "scripts", "journal/scripts", "every j-script, including the checkers"),
    (JROOT / "jlib.py", "journal/jlib.py", "clustered bootstrap, loaders"),
    (JROOT / "readouts.py", "journal/readouts.py", "readout variants (pooling order)"),
    (JROOT / "grouping.py", "journal/grouping.py", "name-blind crossing grouping"),
    (JROOT / "corpus_tools.py", "journal/corpus_tools.py", "clocks, sv2v, blackbox guard"),
    (JROOT / "strong_matcher.py", "journal/strong_matcher.py", "domain-aware rule engine"),
    (JROOT / "wrappers.py", "journal/wrappers.py", "two-clock wrappers"),
    (JROOT / "corpus" / "patches", "journal/corpus/patches", "documented source patches"),
    (ROOT / "data" / "dataset", "data/dataset", "the five benchmark splits"),
    (ROOT / "results", "results", "released checkpoints and conference results"),
    (JROOT / "results", "journal/results", "every result JSON the paper cites"),
    (ROOT / "requirements.txt", "requirements.txt", "pinned dependencies"),
]

SKIP_SUFFIX = {".pyc", ".log"}
SKIP_DIRS = {"__pycache__", "j30_pre_j46", "j30_pre_40crossing", "figures"}


def ignore(_dir, names):
    return [n for n in names
            if n in SKIP_DIRS or Path(n).suffix in SKIP_SUFFIX]


def size_of(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    J.banner("j52  assemble the public artifact")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    rows, total = [], 0
    for src, rel, what in MANIFEST:
        dst = OUT / rel
        if not src.exists():
            print(f"  [MISSING] {src}")
            continue
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        n = size_of(dst)
        total += n
        rows.append(dict(path=rel, bytes=n, what=what))
        print(f"  {n/1e6:7.1f} MB  {rel:<28s} {what}")

    # The netlists of the accepted designs only, named from the triage record.
    triage = json.loads(
        (J.RESULTS / "j12_corpus_triage.json").read_text(encoding="utf-8"))
    # "Usable" is not "accepted": the triage marks 23 usable, one of which
    # (eth_axis_async) is a vendored copy the cell-multiset de-duplicator
    # rejects. Shipping it would contradict the corpus table, which reports 22.
    accepted = sorted({r["name"] for r in triage
                       if r.get("usable") and not r.get("duplicate_of")})
    nl_src, nl_dst = JROOT / "corpus" / "netlists", OUT / "journal/corpus/netlists"
    nl_dst.mkdir(parents=True, exist_ok=True)
    kept = 0
    for name in accepted:
        for f in nl_src.glob(f"{name}.*.json"):
            shutil.copy2(f, nl_dst / f.name)
            kept += 1
    n = size_of(nl_dst)
    total += n
    rows.append(dict(path="journal/corpus/netlists", bytes=n,
                     what=f"{kept} netlists, {len(accepted)} accepted designs"))
    print(f"  {n/1e6:7.1f} MB  {'journal/corpus/netlists':<28s} "
          f"{kept} files for {len(accepted)} accepted designs")

    J.banner("Result")
    print(f"  artifact total: {total/1e6:.1f} MB in {len(rows)} entries")
    print(f"  -> {OUT}")
    print("\n  NOT included, deliberately: cloned upstream RTL (licensing;")
    print("  fetch_corpus.py clones it at pinned commits) and the abandoned")
    print("  P2-phase data (~370 MB supporting nothing the paper claims).")

    J.save("j52_build_artifact",
           dict(total_bytes=total, n_entries=len(rows), entries=rows,
                accepted_designs=accepted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
