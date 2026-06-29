from __future__ import annotations
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math

import pyconll

from ngram_model import NGramModel, PAD, END


class HMMTagger:
    """
    HMM POS tagger with Viterbi decoding (bigram transitions).

    Model:
      - Transition probability A(tag_t | tag_{t-1}) via NGramModel over tag sequences
      - Emission probability B(word_t | tag_t) from counts with add-k smoothing.
      - Decoding uses Viterbi DP with backpointers in log space.

    Required methods:
      - train(filename) -> Nones
      - predict(words) -> list[str]
      - test(filename) -> float (accuracy)
    """

    def __init__(self, k_transition: float = 0.1, k_emission: float = 0.1, unk_threshold: int = 1):
        self.k_transition = float(k_transition)
        self.k_emission = float(k_emission)
        self.unk_threshold = int(unk_threshold)

        # Emission stats
        self.tag_counts: Counter = Counter()                 # c(tag)
        self.emit_counts: Dict[str, Counter] = defaultdict(Counter)  # c(tag, word)

        # For unknown word handling
        self.word_counts: Counter = Counter()
        self.word_vocab: set[str] = set()

        # Tag set / states
        self.tags: List[str] = []

        # Transition model A(tag | prev_tag) using your NGramModel
        self.trans_model: NGramModel | None = None

    # Unknown word handling
    def _norm(self, w: str) -> str:
        # Lowercasing helps reduce sparsity for emissions.
        return w.lower()

    def _shape(self, w: str) -> str:
        # Map unseen/rare words to simple classes.
        if any(ch.isdigit() for ch in w):
            return "<NUM>"
        if "-" in w:
            return "<HYPH>"
        if w[:1].isupper():
            return "<CAP>"
        return "<UNK>"

    def _obs(self, w: str) -> str:
        """
        Convert raw word -> observed token for emission model.
        """
        wl = self._norm(w)
        if wl in self.word_vocab:
            return wl
        return self._shape(w)

    # Training
    def train(self, filename: str) -> None:
        """
        Loads the CoNLL-U file and collects statistics for:
          - Transition model over tags (bigram), using NGramModel (required).
          - Emission model P(word | tag) using add-k smoothing.
        """
        conll = pyconll.load_from_file(filename)

        # Pass 1: count word frequencies and store sentences
        sentences: List[List[Tuple[str, str]]] = []
        for sent in conll:
            pairs = []
            for tok in sent:
                if tok.form is None or tok.upos is None:
                    continue
                w = self._norm(tok.form)
                t = tok.upos
                self.word_counts[w] += 1
                pairs.append((tok.form, t))  # keep original form for shape features
            if pairs:
                sentences.append(pairs)

        # Build known word vocab, plus special buckets
        self.word_vocab = {w for w, c in self.word_counts.items() if c > self.unk_threshold}
        self.word_vocab.update({"<UNK>", "<NUM>", "<CAP>", "<HYPH>"})

        # Pass 2: collect emission counts + tag sequences for transition training
        tag_sentences: List[Tuple[str, ...]] = []
        for pairs in sentences:
            tags_seq = []
            for w_raw, t in pairs:
                w_obs = self._obs(w_raw)
                self.emit_counts[t][w_obs] += 1
                self.tag_counts[t] += 1
                tags_seq.append(t)
            tag_sentences.append(tuple(tags_seq))

        self.tags = sorted(self.tag_counts.keys())

        # Transition model must be based on your NGramModel (bigram over tags)
        self.trans_model = NGramModel(n=2, k=self.k_transition, sentences=tag_sentences)

    # Emission probability
    def _p_emit(self, word: str, tag: str) -> float:
        """
        Add-k smoothed emission:
          P(word | tag) = (k + c(tag,word)) / (k*V + c(tag))
        """
        w_obs = self._obs(word)
        V = len(self.word_vocab)

        c_tw = self.emit_counts[tag].get(w_obs, 0)
        c_t = self.tag_counts[tag]
        return (self.k_emission + c_tw) / (self.k_emission * V + c_t)

    # Viterbi decoding
    def predict(self, words: List[str]) -> List[str]:
        """
        Uses the Viterbi algorithm to find the most likely tag sequence
        for the input list of words.
        """
        if self.trans_model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if not words:
            return []

        states = self.tags
        T = len(words)

        # dp[t][s] = best log-prob up to position t ending in tag s
        dp: List[Dict[str, float]] = [defaultdict(lambda: float("-inf")) for _ in range(T)]
        back: List[Dict[str, str]] = [dict() for _ in range(T)]

        # Initialization (t=0): log A(tag | PAD) + log B(word0 | tag)
        for s in states:
            a = self.trans_model.p(s, (PAD,))
            b = self._p_emit(words[0], s)
            dp[0][s] = math.log(a) + math.log(b)
            back[0][s] = PAD

        # Recursion
        for t in range(1, T):
            for s in states:
                emit_log = math.log(self._p_emit(words[t], s))

                best_prev = None
                best_score = float("-inf")

                for sp in states:
                    trans_log = math.log(self.trans_model.p(s, (sp,)))
                    score = dp[t - 1][sp] + trans_log + emit_log
                    if score > best_score:
                        best_score = score
                        best_prev = sp

                dp[t][s] = best_score
                back[t][s] = best_prev if best_prev is not None else states[0]

        # Termination: transition to END
        best_last = None
        best_score = float("-inf")
        for s in states:
            score = dp[T - 1][s] + math.log(self.trans_model.p(END, (s,)))
            if score > best_score:
                best_score = score
                best_last = s

        # Backtrack
        tags_out = [best_last]
        for t in range(T - 1, 0, -1):
            tags_out.append(back[t][tags_out[-1]])
        tags_out.reverse()
        return tags_out

    # Evaluation
    def test(self, filename: str) -> float:
        """
        Loads CoNLL-U file, predicts tags for each sentence,
        returns accuracy = correct / total.
        """
        conll = pyconll.load_from_file(filename)

        correct = 0
        total = 0

        for sent in conll:
            words = []
            gold = []
            for tok in sent:
                if tok.form is None or tok.upos is None:
                    continue
                words.append(tok.form)
                gold.append(tok.upos)

            if not words:
                continue

            pred = self.predict(words)
            for g, p in zip(gold, pred):
                total += 1
                if g == p:
                    correct += 1

        return correct / total if total else 0.0


if __name__ == "__main__":
    # Paths given in the assignment / Athena post
    train_file = "/home/dsv/mariask/courses/lis050/lab1/en_ewt-ud-train.conllu"
    dev_file = "/home/dsv/mariask/courses/lis050/lab1/en_ewt-ud-dev.conllu"

    tagger = HMMTagger(k_transition=0.1, k_emission=0.1, unk_threshold=1)
    tagger.train(train_file)
    acc = tagger.test(dev_file)
    print(f"Dev accuracy: {acc*100:.2f}%")
