"""j38 - close the last two claims that were only ever verified by hand.

Two numbers in the draft came from one-off checks whose output was never saved:

  Section V   the structural clock rule agrees with the name-based rule on 400
              generated netlists, with zero disagreements
  Section X   relabeling a crossing subgraph by a random permutation changes the
              detection logit by at most 1.3e-08 and changes no decision

Both are re-derived here and written to JSON so j34 can stop calling them
MANUAL. The clock-rule check now runs over every generated netlist available
rather than a sample of 400, which is a stronger claim if it holds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jlib as J                                                   # noqa: E402
from corpus_tools import find_clock_ports                          # noqa: E402
from readouts import ReadoutNet                                    # noqa: E402
from j24_dispersion import relabel_all                             # noqa: E402

sys.path.insert(0, str(J.ROOT))
from crossexplain.models import collate                            # noqa: E402
from crossexplain.train import deterministic_run                   # noqa: E402


def clock_rule_check():
    """Structural clock discovery against the name-based rule it replaced."""
    J.banner("Structural vs name-based clock discovery")
    # data/designs holds netlists AND per-design metadata sidecars; only the
    # former have a "modules" key. Filtering on the filename would silently
    # include the sidecars, which is how an earlier version of this script
    # reported zero disagreements out of zero comparisons.
    files = sorted((J.ROOT / "data" / "designs").rglob("*.json"))
    agree = disagree = skipped = not_netlist = 0
    errors = []
    examples = []
    for f in files:
        try:
            nl = json.loads(f.read_text())
        except Exception as exc:                                   # noqa: BLE001
            skipped += 1
            if len(errors) < 5:
                errors.append(f"{f.name}: {exc}")
            continue
        if "modules" not in nl:
            not_netlist += 1
            continue
        try:
            # returns (clock_ports, clock_bits, top_name, top_module)
            structural, _, _, top = find_clock_ports(nl)
        except Exception as exc:                                   # noqa: BLE001
            skipped += 1
            if len(errors) < 5:
                errors.append(f"{f.name}: {exc}")
            continue
        named = {p for p in top.get("ports", {}) if p.startswith("clk")}
        if set(structural) == named:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append(dict(file=f.name,
                                     structural=sorted(structural),
                                     name_based=sorted(named)))
    print(f"  {len(files)} JSON files: {agree + disagree} netlists compared, "
          f"{not_netlist} metadata sidecars, {skipped} unreadable")
    print(f"  agree {agree}, disagree {disagree}")
    for e in errors:
        print(f"    error: {e}")
    for e in examples:
        print(f"    {e['file']}: structural={e['structural']} "
              f"name={e['name_based']}")
    return dict(n_files=len(files), n_checked=agree + disagree,
                n_not_netlist=not_netlist, n_skipped=skipped,
                agree=agree, disagree=disagree, errors=errors,
                examples=examples)


def relabel_check(n_probe=256):
    """Node relabeling is a graph isomorphism, so it must be a forward no-op."""
    J.banner("Node relabeling is a forward-pass no-op")
    samples = [s for s in J.load("train") if s["label_safe"] != -1][:n_probe]
    with deterministic_run(0, threads=1):
        m = ReadoutNet(readout="mean_seed", arch="gcn", hidden=64, layers=5)
    m.eval()

    def logits(ss):
        out = []
        for i in range(0, len(ss), 64):
            b = collate(ss[i:i + 64])
            with torch.no_grad():
                d, _, _ = m(b["x"], b["ei"], b["batch"], b["seed_mask"],
                            b["n_graphs"])
            out.append(d)
        return torch.cat(out)

    base = logits(samples)
    worst, flips = 0.0, 0
    for seed in range(1, 6):
        p = logits(relabel_all(samples, seed))
        worst = max(worst, float((base - p).abs().max()))
        flips += int((~(base > 0).eq(p > 0)).sum())
    print(f"  {len(samples)} samples x 5 permutations")
    print(f"  max |logit difference| : {worst:.3e}")
    print(f"  decisions changed      : {flips}")
    return dict(n_samples=len(samples), n_perms=5, max_abs_diff=worst,
                decisions_changed=flips)


def main():
    J.banner("j38  closing the last two manual claims")
    clock = clock_rule_check()
    rel = relabel_check()

    J.banner("Result")
    # A check that compared nothing is not a check that passed. Requiring a
    # minimum sample size is the guard that would have caught the earlier
    # "0 disagreements out of 0 comparisons" result.
    ok_clock = clock["disagree"] == 0 and clock["n_checked"] >= 400
    ok_rel = (rel["decisions_changed"] == 0 and rel["max_abs_diff"] < 1e-6
              and rel["n_samples"] >= 64)
    if clock["n_checked"] < 400:
        print(f"  WARNING: only {clock['n_checked']} netlists compared; "
              f"the claim in Section V is for 400 or more.")
    print(f"  clock rule: {clock['agree']}/{clock['n_checked']} agree, "
          f"{clock['disagree']} disagreements -> "
          f"{'CONFIRMED' if ok_clock else 'DISAGREEMENTS FOUND'}")
    print(f"  relabeling: max diff {rel['max_abs_diff']:.3e}, "
          f"{rel['decisions_changed']} decisions changed -> "
          f"{'CONFIRMED' if ok_rel else 'NOT A NO-OP'}")
    if ok_clock and clock["n_checked"] > 400:
        print(f"\n  Section V may now say {clock['n_checked']} netlists "
              f"rather than 400.")

    J.save("j38_close_manual", dict(clock_rule=clock, relabel=rel,
                                    clock_confirmed=ok_clock,
                                    relabel_confirmed=ok_rel))
    return 0 if (ok_clock and ok_rel) else 1


if __name__ == "__main__":
    sys.exit(main())
