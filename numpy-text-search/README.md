# Multilingual Text Search with Word Vectors

A cross-lingual semantic search engine using 300-dimensional word vectors
and cosine similarity, supporting queries across 79 languages.

**Course:** Applied Programming (LIS050), Stockholm University

---

## Overview

This system allows searching through a text database using queries in one
language and retrieving results in another language. For example, searching
with a Swedish query can return matching English sentences.

It uses multilingual word vectors from the ConceptNet Numberbatch project,
where semantic similarity between words — even across languages — can be
measured using cosine similarity.

---

## Contents

- `text_search.py` — main implementation (WordVectors + TextSearch classes)
- `multilingual_vectors.py` — multilingual word vector generation
  using inverted indexing (Søgaard et al., 2015)
- `test_text_search.py` — test script
- `search_configuration_1.json` — search configuration file

---

## Key Features

- Cross-lingual search: query in Swedish, find results in English
- Sentence vectors computed as mean of word vectors
- Cosine similarity for ranking results
- Supports 79 languages via ConceptNet Numberbatch vectors
- Efficient NumPy-based vector operations

---

## Example Searches

    Query (English) → "automotive industry"
    Query (Swedish) → "bilindustri"
    Query (German)  → "jupiter komet"

---

## How to Run

```bash
# Run the search system with a configuration file
python3 test_text_search.py search_configuration_1.json
```

> Note: Word vector files (.gz) not included due to size.
> Download from [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch)
> and place in a `vectors/` directory.

---

## Tech Stack

Python · NumPy · SciPy · Vector Semantics · Cosine Similarity ·
ConceptNet Numberbatch · Multilingual NLP