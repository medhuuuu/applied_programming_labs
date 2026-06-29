import gzip
import os
import numpy as np
from numpy.linalg import norm


class WordVectors:
    def __init__(self, filename):
        """
        Loads a word embedding file (e.g., en.vec.gz or sv.vec.gz)
        and stores all word vectors in a Python dictionary.

        File format (ConceptNet Numberbatch):
          1st line: "<vocab_size> <dimension>"
          Following lines: "<word> <v1> <v2> ... <v_dim>"

        We store vectors as float32 to reduce memory usage.
        """
        self.word2vec = {}


        with gzip.open(filename, "rt", encoding="utf-8") as f:
            # Parse header line to extract vocabulary size and vector dimension
            header = f.readline().strip().split()
            vocab_size, dim = int(header[0]), int(header[1])
            self.dim = dim  # Save dimensionality for later checks

            # Read each subsequent word + vector
            for line in f:
                parts = line.strip().split()
                raw = parts[0].lower()
                
                # Remove language prefix
                if "/" in raw:
                    raw = raw.split("/", 1)[1]
                
                # Skip non‑alphabetical tokens (optional but recommended)
                if not raw.isalpha():
                    continue
                
                word = raw
                vec = np.array(parts[1:], dtype=np.float32)  # store compact float32
                self.word2vec[word] = vec  # map: word → vector


    def make_sentence_vector(self, words):
        """
        Converts a sentence (list of tokens) into a single vector by:
        - Lowercasing all words
        - Removing duplicates (count word types once, not tokens)
        - Including only in‑vocabulary words
        - Averaging all included vectors

        Raises:
            ValueError if no words in the sentence are found in the embedding.
        """
        words = {w.lower() for w in words}  # deduplicate to follow lab instructions
        vectors = []

        # Collect vectors for all words that appear in the embedding dictionary
        for w in words:
            if w in self.word2vec:
                vectors.append(self.word2vec[w])

        # Assignment requirement: error if sentence has no valid words
        if not vectors:
            raise ValueError("No words in vocabulary")

        # Return the mean vector (float32)
        return np.mean(vectors, axis=0).astype(np.float32)


class TextSearch:
    def __init__(self, vector_filenames):
        """
        Loads multiple languages' word vectors.

        Example input: ["en.vec.gz", "sv.vec.gz"]
        The language code is extracted from the filename automatically.
        """
        self.models = {}
        for fname in vector_filenames:
            lang = os.path.basename(fname).split(".")[0]  # extract "en", "sv", etc.
            self.models[lang] = WordVectors(fname)

        # Storage for indexed sentences:
        # each entry = (language_code, filename, original_sentence, sentence_vector)
        self.index = []


    def index_text(self, filename, language_code, min_words=1, max_words=None):
        """
        Reads a gzipped text file with one sentence per line,
        computes a sentence vector for each sentence,
        and stores it for later similarity search.

        Sentences outside the min/max length range are skipped.
        Sentences with no in‑vocabulary words are also skipped.
        """
        with gzip.open(filename, "rt", encoding="utf-8") as f:
            for line in f:
                sent = line.strip()
                words = sent.split()

                # Enforce length limits (as required by the assignment)
                if len(words) < min_words:
                    continue
                if max_words is not None and len(words) > max_words:
                    continue

                try:
                    # Compute vector for the sentence using the correct language model
                    vec = self.models[language_code].make_sentence_vector(words)
                    self.index.append((language_code, filename, sent, vec))
                except ValueError:
                    # Sentence contained no valid words — ignore it
                    pass


    def search(self, query, language_code, n_matches=1):
        """
        Computes cosine similarity between the query vector and every
        indexed sentence, returning the top N matches (highest similarity).

        Returns:
            A list of (similarity_score, filename, sentence_string)
        """
        # Convert user query into a vector using the specified language model
        qvec = self.models[language_code].make_sentence_vector(query)

        scores = []
        for lang, fname, sentence, vec in self.index:
            # Cosine similarity formula: (A·B) / (||A|| * ||B||)
            # FIX: use NumPy-only cosine similarity
            sim = float(np.dot(qvec, vec) / (np.linalg.norm(qvec) * np.linalg.norm(vec)))
            
            scores.append((sim, fname, sentence))

        # Sort descending by similarity and return the top results
        scores.sort(key=lambda x: -x[0])
        return scores[:n_matches]
