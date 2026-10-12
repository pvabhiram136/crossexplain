# CrossExplain — benchmark, corpus and analysis artifact

Everything needed to reproduce the paper: the CDC benchmark generator and its
five splits, the third-party corpus, the released checkpoints, every analysis
script, and the checkers that verify the paper's own numbers.

> **Start here if you only want to check one thing.** Run the checkers
> (§ *Verifying the paper*). They re-derive the paper's numeric claims from the
> result files and will tell you, in a couple of minutes, whether the numbers in
> the PDF match the data in this repository.

---

## What is here

| path | what |
|---|---|
| `crossexplain/` | RTL generator, netlist→graph, model, training, evaluation |
| `journal/scripts/` | every analysis script (`j01`–`j52`), including all checkers |
| `journal/results/` | the result JSON every number in the paper is derived from |
| `journal/corpus/netlists/` | synthesized netlists for the accepted third-party designs |
| `journal/corpus/patches/` | the documented source patches, and the rule for adding one |
| `data/dataset/` | the five benchmark splits (`train`, `val`, `test_id`, `test_big`, `test_real`) |
| `results/` | released checkpoints and the conference-version results |
| `fetch_corpus.py` | clones the upstream RTL at pinned commits |

### What is deliberately *not* here

**Third-party RTL is not redistributed.** The corpus is built from eight upstream
repositories. Four ship a LICENSE; some smaller cores in our working tree carry
none we could find. Rather than vendor all of it, `fetch_corpus.py` clones each
one **at the exact commit the paper's numbers came from**. That is better than a
snapshot anyway: it states the provenance instead of asking you to trust ours.

**Abandoned-phase data is not here.** Roughly 370 MB belonging to the "P2" phase
the paper reports as abandoned supports nothing the paper claims.

---

## Reproducing

```bash
pip install -r requirements.txt
python journal/artifact/fetch_corpus.py --dest journal/corpus   # upstream RTL
```

Rebuilding the corpus additionally needs **Yosys** and **sv2v v0.0.13**. You do
*not* need either to reproduce the results — the netlists and splits are shipped.

### The order matters

A corpus change invalidates most numbers, and the scripts must be re-run in
dependency order:

```
j12, j40, j46          corpus triage, expansion, merge
j25, j32               corpus and transfer tables
j23, j24, j27, j28, j30    everything that trains or evaluates on the corpus
j29, j31, j41, j42, j43, j26   figures and tables
j34, j37, j47, j48, j50, j51   the audits
```

`j30` is the long one: 180 trainings, 4 at a time, roughly an hour. It is
resumable — it skips cells whose JSON already exists — so an interrupted run can
simply be restarted.

---

## Verifying the paper

```bash
cd journal
for s in j34_audit_journal j37_audit_corpus_history j35_check_tex \
         j36_citations j38_close_manual j44_citation_order \
         j47_orphan_numbers j50_table_widths j51_conference_numbers; do
  python scripts/$s.py >/dev/null 2>&1; echo "$s exit=$?"
done
```

All should exit 0. What each one does:

| script | checks |
|---|---|
| `j34` | every numeric claim against the result JSON, including its own stated check count |
| `j37` | Section V's corpus history against the triage run logs |
| `j47` | **the other direction**: every numeral in the LaTeX must be derivable, asserted, or whitelisted with a reason |
| `j50` | every table's typeset width against the column it sits in |
| `j51` | the same numeral scan against the conference version |
| `j35` | structure: orphan bibitems, dangling refs, duplicate labels, unfilled placeholders |
| `j36` | the citation-verification record |
| `j44` | bibliography ordered by first citation |

`j34` and `j47` run in **opposite directions** and only the pair closes the
loop. `j34` compares the data against literals typed into `j34`; it cannot see a
number that drifts in the LaTeX. `j47` scans the LaTeX and demands each numeral
be accounted for. Section XI-E of the paper exists because, for most of the
paper's life, only the first of those existed.

---

## Licence

The code in `crossexplain/` and `journal/` is released under **[choose: MIT /
Apache-2.0]**. The splits and netlists are derived outputs and carry the same
terms. Upstream RTL fetched by `fetch_corpus.py` remains under its own licence —
see each cloned repository.
