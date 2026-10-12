"""j45 - find untried multi-clock modules in the repositories already cloned.

The corpus is the binding constraint on every positive claim in the paper, and
j40 established that expanding it is cheap once a candidate is identified. The
expensive part is identifying candidates, which has so far been done by hand.

This scans the RTL already on disk for module definitions whose port list
mentions two or more distinct clock-like signals, and reports the ones that are
not already in the triage. It is a shortlist generator, not a triage: every hit
still has to survive sv2v, synthesis, the blackbox guard and de-duplication
before it can enter the corpus.

Clock-like is deliberately loose here -- any port matching clk/clock/_clk_/aclk
and friends. Over-generating is fine at this stage; j40 rejects what does not
hold up. Under-generating is what we are trying to fix.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

CORPUS = J.JROOT / "corpus"
# Directories that are not third-party RTL we could draw from.
SKIP = {"netlists", "wrapped", "sv2v", "p2designs", "p2probe", "idiomdesigns",
        "idiomprobe", "sizeidiom"}

MODULE = re.compile(r"^\s*module\s+(\w+)", re.M)

# Only INPUT ports count. A first pass matched any token containing "clk" or
# "clock" anywhere in the header, which found 621 candidates that were almost
# entirely noise: "clock" inside comments, output clocks (clk_o) on clock-tree
# cells, and clock-enable strobes. Requiring a declared input port removes all
# three classes.
INPUT_DECL = re.compile(
    r"\binput\b[^;,()]*?\b(\w*(?:clk|clock)\w*)\b", re.I)

# Clock-tree primitives: two clock inputs, but they are a mux or a gate, not a
# domain crossing. Their whole purpose is to have several clocks on the ports.
CELL_NAME = re.compile(
    r"(clock|clk)_(and|or|xor|mux|buf|buffer|gat|gating|inv|inverter|div)",
    re.I)

# Clock enables and strobes are not clocks.
NOT_A_CLOCK = re.compile(r"(clk_?en|en_?clk|clkena|gate|_period|_cnt$)", re.I)


def clock_inputs(header):
    """Distinct clock-like INPUT ports declared in a module header."""
    names = set()
    for n in INPUT_DECL.findall(header):
        low = n.lower()
        if NOT_A_CLOCK.search(low):
            continue
        names.add(low)
    return names


def main():
    J.banner("j45  shortlist of untried multi-clock modules")
    triage = json.loads((J.RESULTS / "j12_corpus_triage.json").read_text())
    tried = {r["top"] for r in triage if r.get("top")}
    tried |= {r["name"] for r in triage}
    j40 = J.RESULTS / "j40_expand_corpus.json"
    if j40.exists():
        for r in json.loads(j40.read_text())["results"]:
            tried.add(r["name"])
    print(f"  {len(tried)} module names already tried")

    repos = [d for d in sorted(CORPUS.iterdir())
             if d.is_dir() and d.name not in SKIP]
    print(f"  scanning {len(repos)} repositories on disk")

    hits = defaultdict(list)
    n_files = 0
    for repo in repos:
        for f in list(repo.rglob("*.v")) + list(repo.rglob("*.sv")):
            if "/tb/" in f.as_posix() or f.name.startswith(("tb_", "test")):
                continue
            n_files += 1
            try:
                txt = f.read_text(encoding="utf-8", errors="replace")
            except Exception:                                      # noqa: BLE001
                continue
            for m in MODULE.finditer(txt):
                name = m.group(1)
                if name in tried or CELL_NAME.search(name):
                    continue
                # module header = text up to the first ');' after the name,
                # with comments stripped so "clock" in prose does not count
                tail = txt[m.start():m.start() + 6000]
                end = tail.find(");")
                header = tail[:end if end > 0 else 2000]
                header = re.sub(r"//[^\n]*", "", header)
                header = re.sub(r"/\*.*?\*/", "", header, flags=re.S)
                cps = clock_inputs(header)
                if len(cps) >= 2:
                    hits[repo.name].append(
                        dict(module=name, file=str(f.relative_to(CORPUS)),
                             clocks=sorted(cps)))

    print(f"  {n_files} RTL files scanned\n")
    total = 0
    for repo, ms in sorted(hits.items(), key=lambda kv: -len(kv[1])):
        print(f"  {repo}  ({len(ms)} candidates)")
        for h in ms[:8]:
            print(f"      {h['module']:34s} {','.join(h['clocks'][:3])}")
        if len(ms) > 8:
            print(f"      ... and {len(ms) - 8} more")
        total += len(ms)

    J.banner("Result")
    print(f"  {total} untried multi-clock modules across "
          f"{len(hits)} repositories")
    print("  These are CANDIDATES only. Each must still pass sv2v, synthesis,")
    print("  the blackbox guard and cell-multiset de-duplication in j40.")
    J.save("j45_find_candidates",
           dict(n_files=n_files, n_candidates=total,
                by_repo={k: v for k, v in hits.items()}))


if __name__ == "__main__":
    main()
