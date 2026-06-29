# Efficiency & NLP Algorithms

Dynamic programming for word segmentation, N-gram language modelling,
and a Viterbi-based Hidden Markov Model part-of-speech tagger.

**Course:** Applied Programming (LIS050), Stockholm University

---

## Contents

### 1. Word Segmentation (Dynamic Programming)
Three implementations of word segmentation using a unigram language model:

- `word_segment_slow.py` — recursive solution (baseline)
- `word_segment_fast.py` — recursive + dynamic programming (memoization)
- `word_segment_iter.py` — iterative dynamic programming solution

### 2. N-gram Language Model
- `ngram_model.py` — N-gram LM with additive smoothing, supports
  unigram/bigram/trigram
- `test_ngram.py` — test script for evaluating log-probabilities

### 3. HMM Part-of-Speech Tagger (Viterbi)
- `hmm_tagger.py` — bigram HMM tagger using the Viterbi algorithm,
  trained on Universal Dependencies English EWT treebank (~87% accuracy)

---

## Key Concepts

- Dynamic programming (recursive + iterative)
- N-gram language modelling with smoothing
- Hidden Markov Models (HMM)
- Viterbi algorithm for sequence labelling

---

## How to Run

```bash
# Word segmentation
python3 word_segment_fast.py ape-lexicon.json ape.txt

# N-gram language model test
python3 test_ngram.py

# HMM POS tagger (requires CoNLL-U treebank data)
python3 hmm_tagger.py
```

---

## Tech Stack

Python · Dynamic Programming · N-gram LM · HMM · Viterbi Algorithm ·
Universal Dependencies · pyconll