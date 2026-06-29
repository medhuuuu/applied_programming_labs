from __future__ import annotations
from typing import Iterable, Tuple, Dict
from collections import defaultdict
import math

# Use special tokens that will NOT collide with normal corpus words
PAD = "PAD"
END = "END"


class NGramModel:
    """
    N-gram language model with add-k smoothing.

    We store:
      - ngram_counts: counts of n-grams (tuple length n)
      - context_counts: counts of (n-1)-gram contexts (tuple length n-1)
      - vocab: set of word types in training data (plus END)
    """

    def __init__(self, n: int, k: float, sentences: Iterable[Tuple[str, ...]]):
        self.n = int(n)
        self.k = float(k)

        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

        # Vocabulary size V: number of different words in training corpus (+ END)
        # PAD is a boundary marker (context), not a real word type in corpus.
        self.vocab = set()
        total_unigram_tokens = 0  # only used when n == 1

        for sent in sentences:
            sent = tuple(sent)

            # collect vocab from actual training words
            for w in sent:
                self.vocab.add(w)

            # pad and append END for counting
            if self.n == 1:
                tokens = sent + (END,)
            else:
                tokens = (PAD,) * (self.n - 1) + sent + (END,)

            if self.n == 1:
                for w in tokens:
                    self.ngram_counts[(w,)] += 1
                    total_unigram_tokens += 1
            else:
                for i in range(self.n - 1, len(tokens)):
                    context = tokens[i - (self.n - 1): i]
                    w = tokens[i]
                    self.context_counts[context] += 1
                    self.ngram_counts[context + (w,)] += 1

        # END should be part of outcomes, so include it in vocabulary
        self.vocab.add(END)
        self.V = len(self.vocab)

        if self.n == 1:
            self.total_unigrams = total_unigram_tokens

    def p(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Add-k smoothing:
          P(word | context) = (k + c(context,word)) / (kV + c(context))
        """
        if self.n == 1:
            c_w = self.ngram_counts.get((word,), 0)
            return (self.k + c_w) / (self.k * self.V + self.total_unigrams)

        if len(context) != self.n - 1:
            raise ValueError(f"context must have length {self.n - 1}")

        c_context = self.context_counts.get(context, 0)
        c_ng = self.ngram_counts.get(context + (word,), 0)
        return (self.k + c_ng) / (self.k * self.V + c_context)

    def score(self, sentence: Tuple[str, ...]) -> float:
        """
        Sentence log-probability:
        pad with (n-1) PAD, append END, sum natural log probs.
        END MUST be included. (Assignment warning.)
        """
        sentence = tuple(sentence)

        if self.n == 1:
            tokens = sentence + (END,)
            return sum(math.log(self.p(w, ())) for w in tokens)

        tokens = (PAD,) * (self.n - 1) + sentence + (END,)
        total = 0.0
        for i in range(self.n - 1, len(tokens)):
            context = tokens[i - (self.n - 1): i]
            w = tokens[i]
            total += math.log(self.p(w, context))
        return total
