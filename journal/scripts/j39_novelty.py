"""j39 - the novelty check, and what it changed.

Run 13 Sep 2026, before committing further effort to the journal draft. The
question was whether either paper's central claim is already in the literature.

This is a record of what was searched, what was found, and what the finding
required changing. It is not a test; re-run the searches by hand before
submission, since the literature moves.
"""
import sys
from pathlib import Path

JROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(JROOT))

import jlib as J                                                   # noqa: E402

FINDINGS = [
    dict(
        claim="Conference: no learned model applied to the circuit graph for "
              "CDC; no public CDC-ML benchmark with hazard labels",
        verdict="HOLDS",
        searched="CDC detection machine learning GNN netlist benchmark, "
                 "2025-2026",
        found="Only traditional static/formal CDC verification. One 2025 "
              "Springer paper builds a static CDC tool on Yosys netlists "
              "(adjacent method, no learning). No GNN-for-CDC work, no public "
              "CDC-ML benchmark.",
        action="none; novelty claim stands as written"),
    dict(
        claim="Journal: a saturated benchmark loses discriminative power",
        verdict="NOT NOVEL -- established",
        searched="saturated benchmark discriminative power ceiling effect "
                 "leaderboard validity",
        found="Akhtar et al., 'When AI Benchmarks Plateau' (arXiv 2602.16763, "
              "Feb 2026): 60 LLM benchmarks, 14 saturation properties, nearly "
              "half saturated. Defines saturation as exactly this loss of "
              "discrimination. Descriptive survey, LLM domain, no causal "
              "construction, no floating-point discussion.",
        action="Section II rewritten to concede the observation explicitly and "
               "state our delta: a causal construction rather than a survey, "
               "in a hardware domain with a real deployment distribution. "
               "Cited as akhtar2026."),
    dict(
        claim="Journal: floating-point non-associativity perturbs DL training",
        verdict="NOT NOVEL -- established",
        searched="floating point non-associativity reproducibility deep "
                 "learning training reduction order",
        found="Shanmugavelu et al. (arXiv 2408.05148, Aug 2024): FPNA across "
              "HPC and DL pipelines, accumulation in iterative algorithms, "
              "consequences for correctness testing and certification. Framed "
              "throughout as RUN-TO-RUN VARIABILITY.",
        action="Section II rewritten to concede this and state the two "
               "separations: (a) ours is not non-determinism -- both orderings "
               "are individually bit-reproducible, so there is nothing to pin "
               "down; (b) the existing effect is metric variance of a few "
               "points, ours moves real behaviour by 0.872 while the metric "
               "moves by exactly zero. Cited as fpna2024."),
    dict(
        claim="Journal: reduction order is a lever on model BEHAVIOUR, not "
              "merely on run-to-run variance",
        verdict="NOT NOVEL -- established, and closer than we had recorded",
        searched="Semantic Scholar graph API and arXiv, 19 Sep 2026: "
                 "'floating point non-associativity reduction order model "
                 "selection deep learning'",
        found="Shanmugavelu et al., Euro-Par 2025 (arXiv 2503.17173) -- the "
              "same group as fpna2024. FPNA under asynchronous GPU reduction "
              "flips a classifier's output with NO input perturbation; "
              "adversarial-robustness results are overstated when reduction "
              "order is ignored; and orderings that cause misclassification "
              "can be SEARCHED FOR, by Bayesian optimization over external "
              "workloads or by a learnable permutation. This is the closest "
              "prior work found to date.",
        action="Section II rewritten again. We now concede outright that "
               "'reduction order is a lever on behaviour' is theirs, and "
               "state three separations: (a) theirs is inference-time on one "
               "trained model, ours is training-time and yields two different "
               "trained models; (b) theirs is adversarial and sought, ours is "
               "an arbitrary benign choice already present in a repository; "
               "(c) theirs measures how far a metric moves, ours measures "
               "that the benchmark does not move at all. Cited as fpna2025, "
               "and also cited in Section IX where the 'well known' framing "
               "would otherwise understate the prior art."),
    dict(
        claim="Journal: the INTERSECTION -- a mathematically-null "
              "implementation choice producing near-total behavioural "
              "divergence while a saturated benchmark records no difference",
        verdict="NOVEL as far as these searches reach",
        searched="both of the above, plus the combined phrasing",
        found="Nothing combining the two. The saturation literature does not "
              "discuss floating-point or implementation order; the FPNA "
              "literature does not discuss benchmark discriminative power. "
              "Neither exhibits two models with a bit-identical forward pass "
              "that disagree on 87% of real inputs. Re-checked 19 Sep 2026 "
              "against the Euro-Par 2025 result above: it is inference-time "
              "and adversarial, and says nothing about benchmark "
              "discriminative power, so the intersection still stands.",
        action="this is the contribution to lead with; Sections IX and X "
               "already do"),
]

RESIDUAL_RISK = [
    "Second pass, 19 Sep 2026, added the Semantic Scholar graph API and "
    "arXiv to the English-language web search, which is what surfaced the "
    "Euro-Par 2025 result the first pass missed. IEEE Xplore and the ACM DL "
    "still have NOT been queried through their own interfaces, and neither "
    "have the DAC/DATE/ICCAD proceedings, where an EDA-specific version of "
    "this result would most plausibly sit. A DBLP query was attempted and "
    "refused by the service.",
    "The first pass recorded the FPNA literature as framed 'throughout as "
    "run-to-run variability'. That was wrong by omission: the same group had "
    "already published a stronger, behaviour-level result. A novelty check "
    "that reads only the most obvious paper by an author is not a novelty "
    "check, and the miss is recorded here rather than quietly corrected.",
    "The saturation literature is moving quickly -- the closest prior work is "
    "seven months old. Re-run before submission.",
    "'As far as these searches reach' is the correct strength of the novelty "
    "claim, and the paper should not phrase it more strongly than that.",
]


def main():
    J.banner("j39  novelty check")
    for f in FINDINGS:
        print(f"\n  [{f['verdict']}]")
        print(f"  claim   : {f['claim']}")
        print(f"  found   : {f['found']}")
        print(f"  action  : {f['action']}")

    J.banner("Residual risk")
    for r in RESIDUAL_RISK:
        print(f"  - {r}")

    J.save("j39_novelty", dict(date="2026-09-13", second_pass="2026-09-19", findings=FINDINGS,
                               residual_risk=RESIDUAL_RISK))


if __name__ == "__main__":
    main()
