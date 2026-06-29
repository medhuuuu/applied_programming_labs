import sys
import gzip
import ngram_model


def sentence_reader(filename, max_sentences=None):
    n_sentences = 0
    with gzip.open(filename, 'rt') as f:
        for line in f:
            yield tuple(line.split())
            n_sentences += 1
            if max_sentences and n_sentences >= max_sentences:
                return


def main():
    train_corpus = "/home/dsv/robe/lis050/lab1/english-train.txt.gz"
    test_corpus = "/home/dsv/robe/lis050/lab1/english-test.txt.gz"

    example_sentences = tuple(
            tuple(sentence.split()) for sentence in [
                'I have never heard of a gwerkomynx .',
                'I have never heard of a horse .',
                'I never heard have of a horse .'])

    with open("scores.txt", "w") as outf:
        for n in (1, 2, 3):
            print(f"Testing {n}-grams")
            model = ngram_model.NGramModel(
                    n, 0.001, sentence_reader(train_corpus))

            for i, sentence in enumerate(sentence_reader(test_corpus)):
                score = model.score(sentence)
                normal_score = score / len(sentence)
                print(f"{n}\t{i}\t{score:.1f}\t{normal_score:.3f}", file=outf)

            for sentence in example_sentences:
                score = model.score(sentence)
                print(f"    {score:.1f}    {' '.join(sentence)}")


if __name__ == '__main__':
    main()
