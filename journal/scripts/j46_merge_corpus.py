"""j46 - merge the j40 acceptances into the corpus of record.

This is the step j40 deliberately refused to take. It is separated because it
invalidates every number in a finished paper: the corpus grows, the confirmation
set grows, every design-clustered interval moves, and the 60 dispersion models
must be re-evaluated against the new target.

Order matters, and it is:

  1. this script        -- j12_corpus_triage.json gains the new records
  2. j25, j32           -- corpus table, transfer table
  3. j30                -- 60 models re-evaluated (retrains; ~16 min)
  4. j31, j41, j42, j43 -- figures
  5. j34, j37           -- audits, which then report every stale number
  6. fix the prose the audits flag, rebuild, re-read

A backup of the triage file is written first. Re-running is safe: records
already present are skipped.
"""
import json
import shutil
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

TRIAGE = J.RESULTS / "j12_corpus_triage.json"


def main():
    J.banner("j46  merge j40 acceptances into the corpus of record")
    exp = json.loads((J.RESULTS / "j40_expand_corpus.json").read_text())
    new = [r for r in exp["results"] if r.get("usable")]
    if not new:
        print("  nothing to merge")
        return 0

    triage = json.loads(TRIAGE.read_text())
    have = {r["name"] for r in triage}
    todo = [r for r in new if r["name"] not in have]
    print(f"  {len(new)} accepted by j40, {len(new) - len(todo)} already "
          f"merged, {len(todo)} to add")
    if not todo:
        print("  corpus already up to date")
        return 0

    before = sum(1 for r in triage
                 if r.get("usable") and "duplicate_of" not in r)

    backup = TRIAGE.with_suffix(".json.pre_j46")
    if not backup.exists():
        shutil.copy2(TRIAGE, backup)
        print(f"  backup -> {backup.name}")

    for r in todo:
        # match the schema j12 writes, so every downstream consumer works
        rec = dict(name=r["name"], repo=r["repo"], top=r["top"],
                   fetch="cached", n_files=0, status="elaborated",
                   n_clocks=r["n_clocks"], n_cells=r["n_cells"],
                   n_ff=r["n_ff"], clock_names=r["clock_names"],
                   usable=True, secs=0.0, cell_hash=r["cell_hash"],
                   sv2v=r.get("sv2v", False), added_by="j40/j46")
        triage.append(rec)
        print(f"  + {rec['name']:32s} {rec['n_ff']:4d} FFs  "
              f"{rec['n_cells']:5d} cells")

    TRIAGE.write_text(json.dumps(triage, indent=1), encoding="utf-8")
    after = sum(1 for r in triage
                if r.get("usable") and "duplicate_of" not in r)

    conf = J.confirmation_set()
    J.banner("Result")
    print(f"  accepted designs: {before} -> {after}")
    print(f"  confirmation set: {len(conf)} crossings from "
          f"{len({s['design'] for s in conf})} designs")
    print("\n  Every number in the paper that mentions the corpus is now "
          "stale.\n  Run, in order: j25, j32, j30, j31, j41, j42, j43, then "
          "j34 and j37.")

    J.save("j46_merge_corpus", dict(
        added=[r["name"] for r in todo],
        designs_before=before, designs_after=after,
        n_crossings=len(conf),
        n_conf_designs=len({s["design"] for s in conf})))


if __name__ == "__main__":
    sys.exit(main() or 0)
