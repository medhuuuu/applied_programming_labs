# Applied Programming Labs — Stockholm University (LIS050)

Master's-level assignments in applied programming, covering efficient algorithms,
NLP pipelines, neural networks, and multilingual word vector search.

**Course:** Applied Programming (LIS050), Stockholm University  
**Programme:** MSc in AI and Language  

---

## Projects

### 1. Efficiency & NLP Algorithms
Dynamic programming for word segmentation, N-gram language modelling,
and a Viterbi algorithm-based part-of-speech tagger trained on Universal Dependencies.

- Word segmentation with recursive and iterative dynamic programming
- N-gram language model with additive smoothing
- HMM-based POS tagger achieving ~87% accuracy on English EWT treebank

**Tech:** Python, pyconll, dynamic programming, Viterbi algorithm

---

### 2. Multilingual Text Search with Word Vectors
A cross-lingual semantic search engine using 300-dimensional ConceptNet
Numberbatch word vectors and cosine similarity.

- Query in one language, retrieve results in another
- Supports 79 languages via multilingual embeddings
- Efficient sentence vector computation with NumPy

**Tech:** Python, NumPy, SciPy, vector semantics, cosine similarity

---

### 3. Neural Essay Score Prediction (Backpropagation)
A neural regression model built from scratch using PyTorch to predict
essay quality scores from linguistic features.

- Implemented stochastic gradient descent with backpropagation
- Compared PyTorch model against scikit-learn LinearRegression baseline
- Features: essay length (n^1/4) and lexical diversity (OVIX)

**Tech:** Python, PyTorch, scikit-learn, NumPy

---

### 4. Machine Translation: SMT vs NMT (English → Swedish)
A comparative study of two machine translation approaches trained on
Europarl v10 and evaluated on WMT24++ English–Swedish test set.

- **SMT (from scratch):** IBM Model 1 with EM training + trigram language
  model + greedy decoder — no external ML libraries
- **NMT (PyTorch):** LSTM encoder-decoder with dot-product attention,
  BPE-like subword tokenization, teacher forcing
- Evaluated both systems using BLEU score
- Includes time and space complexity analysis for all components
- Full pydoc documentation

**Key finding:** NMT generalizes better under low-resource conditions
due to subword units and learned distributed representations,
while SMT struggles with sparse vocabularies and unseen words.

**Tech:** Python, PyTorch, LSTM, attention mechanism, IBM Model 1,
N-gram LM, BLEU evaluation, Europarl v10

---

## About

These assignments were completed as part of the
[MSc in AI and Language](https://www.su.se/english/education/course-catalogue/li/lis050)
at Stockholm University, Sweden.