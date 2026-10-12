"""j36 - the verification record for the citations added in the journal version.

Fifteen of the twenty-one bibliography entries are inherited from the audited
conference paper. Six were added for this version and were originally written
from memory, which is not an acceptable basis for a reference list. Each was
subsequently checked against the published record; this file records what was
checked, against what source, and what changed.

This is a record, not a test. It writes its contents to JSON so the state is
inspectable, and re-running it does not re-verify anything -- if a bibliography
entry is edited, verify it again by hand and update the entry here.
"""
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

VERIFIED = [
    dict(key="torralba2011", verified=True, date="2026-08-23",
         source="ACM DL 10.1109/CVPR.2011.5995347; CMU Robotics Institute "
                "publication record",
         confirmed="Torralba & Efros, CVPR 2011, pp. 1521-1528, "
                   "Colorado Springs",
         changed="added month (Jun.)"),
    dict(key="recht2019", verified=True, date="2026-08-23",
         source="dblp conf/icml/RechtRSS19; ICML 2019 virtual proceedings",
         confirmed="Recht, Roelofs, Schmidt, Shankar; ICML 2019, "
                   "pp. 5389-5400; Long Beach",
         changed="qualified as 36th ICML"),
    dict(key="liao2021", verified=True, date="2026-08-23",
         source="datasets-benchmarks-proceedings.neurips.cc, 2021 round 2",
         confirmed="Liao, Taori, Raji, Schmidt; NeurIPS Datasets and "
                   "Benchmarks Track, 2021",
         changed="venue name corrected to 'NeurIPS Track Datasets and "
                 "Benchmarks'"),
    dict(key="nondeterminism", verified=True, date="2026-08-23",
         source="proceedings.mlsys.org 2022 paper "
                "427e0e886ebf87538afdf0badb805b7f",
         confirmed="Zhuang, Zhang, Song, Hooker; MLSys 2022; "
                   "Santa Clara, CA, Aug 29 - Sep 1 2022",
         changed="added location"),
    dict(key="higham2002", verified=True, date="2026-08-23",
         source="SIAM catalogue; nhigham.com book page; ISBN 0-89871-521-0",
         confirmed="Higham, Accuracy and Stability of Numerical Algorithms, "
                   "2nd ed., SIAM, 2002",
         changed="none"),
    dict(key="sv2v", verified=True, date="2026-08-23",
         source="github.com/zachjs/sv2v; LICENSE file in the vendored copy at "
                "journal/tools/sv2v-Windows/LICENSE",
         confirmed="'sv2v: SystemVerilog to Verilog'; BSD 3-Clause; "
                   "Copyright 2019-2024 Zachary Snow, 2011-2015 Tom Hawkins",
         changed="added licence; access date"),
]

INHERITED = [
    "ginosar2011", "cummings2008", "commercial", "litterick2006",
    "meridianml", "cummings", "kipf2017gcn", "hamilton2017sage",
    "velickovic2018gat", "gnnre", "trojansaint", "trojangnn", "xaisurvey",
    "sanchez2020", "yosys",
]

# Carried over from the conference paper's own audit, where both were flagged
# and never closed. Both are now resolved, and the second required a correction
# to the CONFERENCE paper as well as this one.
INHERITED_RESOLVED = [
    dict(key="trojansaint", verified=True, date="2026-08-23",
         source="dblp conf/iscas/LashenAKS23, DOI "
                "10.1109/ISCAS46773.2023.10181403",
         confirmed="Lashen, Alrahis, Knechtel, Sinanoglu; ISCAS 2023; "
                   "pp. 1-5 correct as printed",
         changed="none -- the flagged page range was right"),
    dict(key="sanchez2020", verified=True, date="2026-08-23",
         source="dblp conf/nips/Sanchez-Lengeling20",
         confirmed="Sanchez-Lengeling et al.; NeurIPS 2020",
         changed="TWO CORRECTIONS. (1) page range 5898-5910 could not be "
                 "confirmed -- dblp lists no pages for this paper -- so it is "
                 "dropped rather than kept unsupported. (2) the venue was "
                 "given as Vancouver, BC; NeurIPS 2020 was held virtually. "
                 "Both fixes applied to paper/crossexplain_ieee.tex, "
                 "paper/crossexplain_ieee_single.tex and the journal draft."),
]

OUTSTANDING = []


def main():
    J.banner("j36  citation verification record")
    print(f"  {len(VERIFIED)} added for the journal version, all verified")
    for v in VERIFIED:
        print(f"    {v['key']:16s} {v['changed']}")
    print(f"\n  {len(INHERITED)} inherited from the audited conference "
          f"bibliography, of which\n  {len(INHERITED_RESOLVED)} carried "
          f"unresolved flags that are now closed:")
    for r in INHERITED_RESOLVED:
        print(f"    {r['key']:16s} {r['changed'][:60]}")
        for extra in [r["changed"][i:i + 60]
                      for i in range(60, len(r["changed"]), 60)]:
            print(f"    {'':16s} {extra}")

    print(f"\n  outstanding: {len(OUTSTANDING) or 'none'}")
    if not OUTSTANDING:
        print("  Every bibliography entry in both papers has now been checked "
              "against a\n  primary source. The sanchez2020 fix applies to the "
              "CONFERENCE paper too;\n  both .tex files there were updated.")

    J.save("j36_citations", dict(
        verified=VERIFIED, inherited=INHERITED,
        inherited_resolved=INHERITED_RESOLVED, outstanding=OUTSTANDING,
        n_total=len(VERIFIED) + len(INHERITED),
        all_verified=not OUTSTANDING))


if __name__ == "__main__":
    main()
