import sys
from word_segment_slow import BasicWordSegmenter


class FastWordSegmenter(BasicWordSegmenter):
    """
    Recursive word segmenter with dynamic programming (memoization).

    Key idea:
    - slow version recomputes best segmentation for the same substrings many times.
    - we store (best_score, best_segmentation) for each substring in a dictionary.
    """

    # override
    def segment(self, text: str) -> str:
        # reuse the superclass input cleaning/checking
        text = self.format_and_check_text(text)

        # memo table: substring -> (best_score, best_segmentation_string)
        memo = {}

        try:
            score, best_segmentation = self._get_best_segmentation_fast(text, memo)
            return best_segmentation.strip()
        except RecursionError:
            # Assignment requirement: handle deep recursion, print error, continue next line
            print("Line too long to handle (recursion depth).", file=sys.stderr)
            return ""


    def _get_best_segmentation_fast(self, text: str, memo: dict):
        """
        Returns: (best_log_score, best_segmentation_string)
        Implements the same recurrence as the slow version, but caches results.
        """

        # Base case: empty string
        if len(text) == 0:
            return (0.0, "")

        # Memoization lookup: if the substring solved once reuse it
        if text in memo:
            return memo[text]

        segmentation_candidates = []

        # Only need to check suffixes up to the longest word in lexicon
        start_check_at = max(len(text) - self.len_longest_word, 0)

        for i in range(start_check_at, len(text)):
            potential_word = text[i:]

            # skip invalid word candidates (except single letters are always allowed)
            if not self.is_word_in_lexicon(potential_word):
                continue

            prefix = text[:i]

            # recursion on prefix (BUT now cached)
            prefix_score, prefix_segmentation = self._get_best_segmentation_fast(prefix, memo)

            # log probability for the last word
            word_score = self.get_log_probability_for_word(potential_word)

            total_score = prefix_score + word_score
            segmentation = (prefix_segmentation + " " + potential_word).strip()

            segmentation_candidates.append((total_score, segmentation))

        # best candidate by score
        best_candidate = max(segmentation_candidates, key=lambda t: t[0])

        memo[text] = best_candidate
        return best_candidate


if __name__ == "__main__":
    lexicon_filename = sys.argv[1]
    text_to_segment_filename = sys.argv[2]

    segmenter = FastWordSegmenter(lexicon_filename)

    with open(text_to_segment_filename, encoding="utf-8") as text_file:
        for line in text_file:
            try:
                print(segmenter.segment(line.strip()))
            except ValueError:
                # keep the same behaviour as slow.py: print blank line + error on stderr
                print()
                print("Line too long or contains newline: ", line.strip(), file=sys.stderr)
