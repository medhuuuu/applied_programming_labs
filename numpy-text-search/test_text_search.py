import sys
import os
import pprint
import json

from text_search import TextSearch


def main():
    if len(sys.argv) < 2:
        print("As argument, you need to provide a json configuration file.")
        sys.exit(1)
    configuration_file = sys.argv[1]
    with open(configuration_file) as cf:
        configuration = json.load(cf)
    print(configuration)
    word_vectors_filenames = configuration["embeddingSpacesPaths"]
        
    languages = [os.path.basename(filename).split('.')[0]
                 for filename in word_vectors_filenames]

    print(f"Loading word vectors for: {', '.join(languages)}...")
    ts = TextSearch(word_vectors_filenames)

    print("Indexing sentences...")
    for tts in configuration["textsToSearchIn"]:
        ts.index_text(tts["textPath"], tts["language"],
                  min_words=5, max_words=30)

    while True:
        query = input("Please specify a search query: ").strip().split()

        while True:
            language = input(
                    f"Language code of query [{'/'.join(languages)}] ").strip()
            if language in languages:
                break
            print("Invalid language code, please try again.")

        try:
            matches = ts.search(query, language, 5)
            pprint.pp(matches)
        except ValueError:
            print("None of the query words are in the vocabulary, try again.")


main()

