"""
bleu_eval.py
-----------------
Compute BLEU scores for SMT and NMT outputs
against the reference WMT24++ Swedish test set.
"""

import math
from collections import Counter


# ===========================================
# Utility Functions
# ===========================================

def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def bleu_score(candidate, reference, max_n=4):
    """
    Basic BLEU implementation (no external libs).
    """
    cand = candidate.split()
    ref = reference.split()

    if len(cand) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n+1):
        c_ng = ngrams(cand, n)
        r_ng = ngrams(ref, n)

        overlap = sum(min(count, r_ng[ng]) for ng, count in c_ng.items())
        total = max(sum(c_ng.values()), 1)
        precisions.append(overlap / total)

    # geometric mean of precisions
    score = math.exp(sum(math.log(p + 1e-9) for p in precisions) / max_n)

    # brevity penalty
    c_len = len(cand)
    r_len = len(ref)

    if c_len < r_len:
        bp = math.exp(1 - r_len / c_len)
    else:
        bp = 1.0

    return bp * score


# ===========================================
# Evaluation Loop
# ===========================================

def evaluate(ref_file, sys_file):
    """
    Compute average BLEU for system output vs reference.
    """
    refs = open(ref_file, "r", encoding="utf-8").read().strip().split("\n")
    sys = open(sys_file, "r", encoding="utf-8").read().strip().split("\n")

    scores = []
    for r, s in zip(refs, sys):
        scores.append(bleu_score(s, r))

    return sum(scores) / len(scores), len(scores)


# ===========================================
# MAIN
# ===========================================

if __name__ == "__main__":
    print("Evaluating SMT...")
    smt_bleu, n1 = evaluate("data/test.sv", "eval/smt_output.sv")
    print(f"SMT BLEU = {smt_bleu:.4f} over {n1} sentences")

    print("\nEvaluating NMT...")
    nmt_bleu, n2 = evaluate("data/test.sv", "eval/nmt_output.sv")
    print(f"NMT BLEU = {nmt_bleu:.4f} over {n2} sentences")

    print("\nDone.")
