import gzip
import os
from collections import Counter, defaultdict
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.decomposition import TruncatedSVD

EUROPARL_DIR = "/home/dsv/mariask/courses/lis050/lab2/europarl"
OUTPUT_DIR = "./my_vectors"
MIN_WORD_FREQ = 10
DIMENSIONS = 100


def read_parallel_files():
    """
    Returns:
        lang_files: { lang_code -> { sentence_id -> [tokens...] } }
        shared_ids: list of Europarl sentence IDs common to all languages
    """
    files = sorted(os.listdir(EUROPARL_DIR))
    lang_files = {}

    for f in files:
        if f.endswith(".gz"):
            lang = f.split(".")[0]
            lang_path = os.path.join(EUROPARL_DIR, f)
            print(f"Loading {lang_path}...")

            sent_map = {}

            with gzip.open(lang_path, "rt", encoding="utf-8") as inp:
                for line in inp:
                    line = line.strip()
                    if not line:
                        continue

                    # Europarl format: "<id>\t<token token token>"
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        continue

                    sentence_id = parts[0]       # keep ID as string
                    tokens = parts[1].split()
                    sent_map[sentence_id] = tokens

            lang_files[lang] = sent_map

    # === FIX: compute shared sentence IDs ===
    sentence_sets = [set(m.keys()) for m in lang_files.values()]
    shared_ids = set.intersection(*sentence_sets)

    print(f"Shared sentences across languages: {len(shared_ids)}")

    if len(shared_ids) == 0:
        raise ValueError("No shared sentence IDs across selected languages.")

    return lang_files, shared_ids


def build_vocabulary(lang_files, shared_ids):
    """
    Build vocabulary ONLY from shared sentences.
    Returns vocab list and word->index dictionary.
    """
    print("Building vocabulary...")
    freq = Counter()

    for lang, sent_map in lang_files.items():
        for sid in shared_ids:
            tokens = sent_map[sid]
            for word in tokens:
                freq[f"{lang}/{word.lower()}"] += 1

    vocab = [w for w, c in freq.items() if c >= MIN_WORD_FREQ]
    vocab.sort()

    word2id = {w: i for i, w in enumerate(vocab)}

    print(f"Final vocabulary size: {len(vocab)}")
    return vocab, word2id


def build_sparse_matrix(lang_files, shared_ids, vocab, word2id):
    """
    Builds sparse inverted index matrix using the TRUE Europarl sentence IDs.
    Uses ONLY the shared_ids.
    """
    vocab_size = len(vocab)

    # === FIX: sort only SHARED ids ===
    def sort_key(x):
        if ":" in x:
            prefix, _, num = x.rpartition(":")
            try:
                return (prefix, int(num))
            except:
                return (prefix, 999999999)
        return (x, 999999999)

    sentence_ids = sorted(shared_ids, key=sort_key)
    num_sentences = len(sentence_ids)

    print(f"Matrix size: {vocab_size} x {num_sentences}")

    # mapping: Europarl ID -> column index
    sentid_to_col = {sid: i for i, sid in enumerate(sentence_ids)}

    M = dok_matrix((vocab_size, num_sentences), dtype=np.float32)

    for lang, sent_map in lang_files.items():
        for sid in shared_ids:
            col = sentid_to_col[sid]
            tokens = sent_map[sid]

            words = set(w.lower() for w in tokens)
            for w in words:
                tok = f"{lang}/{w}"
                if tok in word2id:
                    M[word2id[tok], col] = 1.0

    return M.tocsr()


def reduce_dimensions(matrix):
    print("Performing Truncated SVD...")
    svd = TruncatedSVD(n_components=DIMENSIONS)
    reduced = svd.fit_transform(matrix)
    print("Reduction complete.")
    return reduced


def save_vectors(vocab, vectors):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    lang_map = defaultdict(list)
    for tok, vec in zip(vocab, vectors):
        lang, word = tok.split("/", 1)
        lang_map[lang].append((word, vec))

    for lang, entries in lang_map.items():
        path = os.path.join(OUTPUT_DIR, f"{lang}.vec.gz")
        with gzip.open(path, "wt", encoding="utf-8") as out:
            out.write(f"{len(entries)} {DIMENSIONS}\n")
            for word, vec in entries:
                line = " ".join(f"{x:.5f}" for x in vec)
                out.write(f"{word} {line}\n")

        print(f"Saved: {path}")


def main():
    lang_files, shared_ids = read_parallel_files()
    vocab, word2id = build_vocabulary(lang_files, shared_ids)
    matrix = build_sparse_matrix(lang_files, shared_ids, vocab, word2id)
    vectors = reduce_dimensions(matrix)
    save_vectors(vocab, vectors)


if __name__ == "__main__":
    main()
