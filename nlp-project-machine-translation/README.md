# Machine Translation: SMT vs NMT (English → Swedish)

A comparative implementation of two machine translation approaches,
built as part of the Applied Programming course (LIS050) at Stockholm University.

## Overview

This project implements and compares two machine translation systems
for English → Swedish translation:

| System | Approach | BLEU |
|--------|----------|------|
| SMT | IBM Model 1 + Trigram LM (from scratch) | 0.0000 |
| NMT | LSTM Encoder-Decoder with Attention (PyTorch) | 0.0001 |

Although BLEU scores are low due to limited training data (10,000 sentences)
and domain mismatch with WMT24++, the project demonstrates that NMT
generalizes significantly better than SMT under low-resource conditions.

---

## System Architectures

### SMT (Statistical Machine Translation)
Built entirely from scratch without external ML libraries:
- **IBM Model 1** with EM training for word alignment probabilities
- **Trigram language model** for Swedish fluency
- **Greedy decoder** with local word swaps

### NMT (Neural Machine Translation)
Built with PyTorch:
- **Encoder:** 1-layer LSTM (embedding=64, hidden=128)
- **Decoder:** LSTM with dot-product attention
- **Tokenization:** BPE-like subword segmentation
- **Training:** 12 epochs, batch size 32, teacher forcing 0.7
- **Loss:** Cross-entropy with PAD masking

---

## Complexity Analysis

### SMT
| Component | Time | Space |
|-----------|------|-------|
| IBM Model 1 | O(N × I × S × T) | O(V_en × V_sv) |
| Trigram LM | O(Total Tokens) | O(V + B + T) |
| Decoder | O(S × V) | O(Sentence Length) |

### NMT
| Component | Time | Space |
|-----------|------|-------|
| Encoder LSTM | O(T × H²) | O(T × H) |
| Decoder LSTM | O(T × H²) | O(H) |
| Attention | O(T_src × T_tgt × H) | O(T_src × T_tgt) |

---

## Dataset

- **Training:** Europarl v10 (OPUS) — 10,000 parallel sentences extracted
- **Evaluation:** WMT24++ English–Swedish test set (news, literature, social media)

> Note: Full Europarl data not included due to size.
> Download from [OPUS](https://opus.nlpl.eu/Europarl.php) and place in `data/`.

---

## Project Structure

    nlp-project/
    ├── src/                    # Source code
    ├── eval/                   # Evaluation scripts
    ├── docs/                   # Pydoc HTML documentation
    │   ├── nmt_model.html
    │   └── smt_model.html
    ├── data/
    │   ├── test.en             # English test sentences
    │   ├── test.sv             # Swedish test sentences
    │   └── wmt24pp_test.jsonl
    └── extract_wmt.py          # WMT data extraction script

---

## Results & Key Findings

**NMT generalizes better** under low-resource conditions:
- SMT produces literal word-by-word translations with many `<unk>` tokens
- NMT learns meaningful fragments via subword units and attention

Example translation of *"This is very important."*:
- SMT output: `det`
- NMT output: `det är vikt igt`

---

## Tech Stack

Python · PyTorch · LSTM · Attention Mechanism · IBM Model 1 ·
N-gram Language Model · BLEU · Europarl v10 · WMT24++