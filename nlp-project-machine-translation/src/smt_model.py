"""
smt_model.py
---------------
A faster, optimized Statistical Machine Translation (SMT) system for
English→Swedish using:

• IBM Model 1 word alignment
• Pruned Swedish vocabulary
• Trigram LM with Laplace smoothing
• Greedy decoder + LM-based swap refinement
• Lightweight BLEU scorer

This version is optimized for small data (5000 lines).

Time/Space Complexity Notes:
• IBM Model 1 Training:
    Time: O(N * I * S * T)
    Space: O(V_en * V_sv)
• Language Model:
    Time: O(total_tokens)
    Space: O(trigrams)
• Decoding:
    Time: O(L * V_pruned)
    Space: O(L)
V_pruned is used to drastically speed up decoding.
"""

import math
from collections import defaultdict, Counter


# ============================================================
# 1. Data Loading
# ============================================================

def load_parallel_data(en_file, sv_file, max_lines=None):
    """
    Load tokenized English–Swedish parallel data.

    Time Complexity:
        O(N * L)
    Space Complexity:
        O(N * L)
    """
    en_lines, sv_lines = [], []

    with open(en_file, "r", encoding="utf-8") as fe, \
         open(sv_file, "r", encoding="utf-8") as fs:
        for i, (en, sv) in enumerate(zip(fe, fs)):
            if max_lines and i >= max_lines:
                break
            en_lines.append(en.strip().split())
            sv_lines.append(sv.strip().split())

    return en_lines, sv_lines


# ============================================================
# 2. IBM Model 1 (EM Training)
# ============================================================

def initialize_translation_probs():
    """
    Create a translation table t(sv|en) with uniform initialization.

    Space: O(V_en * V_sv)
    """
    return defaultdict(lambda: defaultdict(lambda: 1.0))


def em_training(t, en_sentences, sv_sentences, iterations=5):
    """
    Perform EM training for IBM Model 1.

    Time:
        O(iter * N * S * T)
    """
    for it in range(iterations):
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)

        for en_words, sv_words in zip(en_sentences, sv_sentences):

            # Compute normalization per Swedish word
            for sv_w in sv_words:
                Z = sum(t[sv_w][en_w] for en_w in en_words)

                for en_w in en_words:
                    contrib = t[sv_w][en_w] / Z
                    count[sv_w][en_w] += contrib
                    total[en_w] += contrib

        # Update t-table
        for sv_w in count:
            for en_w in count[sv_w]:
                t[sv_w][en_w] = count[sv_w][en_w] / total[en_w]

        print(f"EM iteration {it+1} done.")

    return t


# ============================================================
# 3. Trigram Language Model
# ============================================================

class TrigramLM:
    """
    Trigram language model with Laplace smoothing.
    """

    def __init__(self):
        self.uni = Counter()
        self.bi = Counter()
        self.tri = Counter()
        self.V = 0

    def train(self, sentences):
        """
        Train LM.

        Time: O(total_tokens)
        """
        for words in sentences:
            words = ["<s>", "<s>"] + words + ["</s>"]

            for i, w in enumerate(words):
                self.uni[w] += 1
                if i >= 1:
                    self.bi[(words[i-1], w)] += 1
                if i >= 2:
                    self.tri[(words[i-2], words[i-1], w)] += 1

        self.V = len(self.uni)

    def prob(self, w1, w2, w3):
        return (self.tri[(w1, w2, w3)] + 1) / (self.bi[(w1, w2)] + self.V)


# ============================================================
# 4. Greedy Decoding (Optimized)
# ============================================================

def build_pruned_vocab(t_table, min_align=2):
    """
    Prune Swedish vocabulary by removing words with too few alignment links.

    This drastically speeds up decoding.

    Returns:
        list[str]: pruned Swedish vocabulary
    """
    return [sv for sv in t_table if len(t_table[sv]) >= min_align]


def greedy_decode(sentence_en, t_table, lm, sv_vocab):
    """
    Greedy decoding with pruned vocabulary.

    Time:
        O(L * V_pruned)
    """
    # Step 1: word-by-word best match
    sv_raw = []
    for en_w in sentence_en:
        best_sv, best_p = None, -1

        for sv_w in sv_vocab:
            p = t_table[sv_w][en_w]
            if p > best_p:
                best_p = p
                best_sv = sv_w

        sv_raw.append(best_sv)

    # Step 2: simple LM-based local swap
    improved = True
    while improved:
        improved = False
        for i in range(len(sv_raw) - 1):
            base = lm.prob("<s>", sv_raw[i], sv_raw[i+1])
            swap = lm.prob("<s>", sv_raw[i+1], sv_raw[i])
            if swap > base:
                sv_raw[i], sv_raw[i+1] = sv_raw[i+1], sv_raw[i]
                improved = True

    return sv_raw


# ============================================================
# 5. BLEU Score
# ============================================================

def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))


def compute_bleu(candidate, reference, max_n=4):
    """
    Lightweight BLEU.

    Time: O(n * L)
    """
    c_len, r_len = len(candidate), len(reference)
    if c_len == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n+1):
        cand_ngrams = ngram_counts(candidate, n)
        ref_ngrams = ngram_counts(reference, n)

        overlap = sum(min(v, ref_ngrams[k]) for k, v in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)

        precisions.append(overlap / total)

    # geometric mean
    score = math.exp(sum(math.log(p + 1e-9) for p in precisions) / max_n)

    # brevity penalty
    bp = 1.0 if c_len >= r_len else math.exp(1 - r_len / c_len)

    return bp * score


# ============================================================
# 6. End-to-End Training
# ============================================================

def train_smt(train_en, train_sv, subset=5000, em_iters=5):
    """
    Train SMT pipeline.
    """
    en_sents, sv_sents = load_parallel_data(train_en, train_sv, max_lines=subset)

    t = initialize_translation_probs()
    t = em_training(t, en_sents, sv_sents, iterations=em_iters)

    lm = TrigramLM()
    lm.train(sv_sents)

    # prune vocabulary for fast decoding
    sv_vocab = build_pruned_vocab(t, min_align=2)

    return t, lm, sv_vocab


# ============================================================
# 7. File Translation
# ============================================================

def translate_file(in_file, out_file, t_table, lm, sv_vocab):
    with open(in_file, "r", encoding="utf-8") as fi, \
         open(out_file, "w", encoding="utf-8") as fo:

        for line in fi:
            words = line.strip().split()
            sv = greedy_decode(words, t_table, lm, sv_vocab)
            fo.write(" ".join(sv) + "\n")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Training SMT model on 5000 sentences...")

    t_table, lm, sv_vocab = train_smt(
        "data/train_small.en",
        "data/train_small.sv",
        subset=5000,
        em_iters=5
    )

    print("Decoding test set...")
    translate_file("data/test.en", "eval/smt_output.sv", t_table, lm, sv_vocab)

    print("Done. Output saved to eval/smt_output.sv")
