import sys
from word_segment_slow import BasicWordSegmenter


class IterativeWordSegmenter(BasicWordSegmenter):
    """
    Iterative dynamic programming word segmenter (bottom-up DP).

    dp[i]   = best log-score for segmenting text[:i]
    back[i] = best previous cut position j for text[:i]
    """

    def segment(self, text: str) -> str:
        text = self.format_and_check_text(text)

        n = len(text)
        
        # dp[0] = 0 for empty string (log(1)=0)
        dp = [float("-inf")] * (n + 1)
        back = [-1] * (n + 1)

        dp[0] = 0.0
        back[0] = 0

        max_len = self.len_longest_word  # longest lexicon word length

        # Fill dp table from left to right
        for i in range(1, n + 1):
            best_score = float("-inf")
            best_j = -1

            # Try last word lengths L = 1..max_len, but not longer than i
            for L in range(1, min(max_len, i) + 1):
                j = i - L
                w = text[j:i]

                # Skip multi-letter words not in lexicon (single letters always allowed)
                if not self.is_word_in_lexicon(w):
                    continue

                candidate = dp[j] + self.get_log_probability_for_word(w)

                if candidate > best_score:
                    best_score = candidate
                    best_j = j

            dp[i] = best_score
            back[i] = best_j

        # Reconstruct segmentation using backpointers
        if back[n] == -1:
            # Should not happen, but safe fallback: split into characters
            return " ".join(list(text))

        words = []
        i = n
        while i > 0:
            j = back[i]
            words.append(text[j:i])
            i = j

        words.reverse()
        return " ".join(words).strip()


if __name__ == "__main__":
    lexicon_filename = sys.argv[1]
    text_to_segment_filename = sys.argv[2]

    segmenter = IterativeWordSegmenter(lexicon_filename)

    with open(text_to_segment_filename, encoding="utf-8") as text_file:
        for line in text_file:
            try:
                print(segmenter.segment(line.strip()))
            except ValueError:
                print()
                print("Line too long or contains newline: ", line.strip(), file=sys.stderr)
