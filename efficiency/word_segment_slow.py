"""
The class BasicWordSegmenter, and code for running it 
with lexicon and a text to segment.

Note that this class can only be used for segmenting
short texts. For longer texts, it is too slow.

With a lexicon file, with the filename `ape-lexicon.json' 
and a text file with the filename `ape.txt', the code is 
called as follows:
 
python3 word_segment_slow.py ape-lexicon.json ape.txt

The lexicon should be in json format, e.g. as follows. 
{
    "a": 120.0,
    "an": 89.0,
    "ape": 2.0
}

The file with texts to segment, should contain one text per line.

The resulting segmented text is printed on standard out. 
Segments are indicated by inserted whitespace.

"""

import math
import json
import sys


class BasicWordSegmenter:
    """
    A class for segmenting a text into "words", 
    based on the lexicon provided when creating
    the class.
    
    Attributes: 
        lexicon (dict): Words with frequency information
        log_N (float): Logarithm of total number of (non-unique) words in the lexicon
        len_longest_word (int): The length of the longest word in the lexicon
    """

    def __init__(self, filename):
        """
        Parameter:
            filename (str): The filename to json file
            that contains the lexicon
        """

        with open(filename, encoding="utf-8") as lexicon_file:
            self.lexicon = json.load(lexicon_file)

        self.log_N = math.log(sum(self.lexicon.values()))
        self.len_longest_word = max(len(key) for key in self.lexicon.keys())


    def is_word_in_lexicon(self, word):
        """
        Parameter:
            word (str): The word that is to be checked
        Returns:
            boolean: True if word in lexicon, otherwise False
        """
        return word in self.lexicon or len(word) == 1


    def get_log_probability_for_word(self, word):
        """
        Returns the logarithmic probability for a word, 
        based on a unigram model of word frequencies.
        
        Parameter:
            word (str): The word that is to be checked
        Returns:
            float: Logarithmic probability for word
        """
        if word in self.lexicon:
            return math.log(self.lexicon[word]) - self.log_N
        if len(word) == 1:
            return math.log(1) - self.log_N
        return 0.0

    def _get_best_segmentation_slow(self, text):#anaple
        """
        Private helper method.
        Returns the best segmentation for a text, 
        together with the probability score for that segmentation.
        
        Parameter:
            text (str): The text to segment
        Returns: 
            (float, str): A tuple with probability score and segmentation
        """

        MAX_TEXT_LENGTH = 30
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"The text to segment is not allowed to be longer \
             than {MAX_TEXT_LENGTH} letters")
        if len(text) == 0:
            return (0.0, "")

        # To store tuples with possible segmentations of the text
        # i.e, of score and segmentations.
        segmentation_candidates = []

        # No point in looking for a longer word than longest in lexicon
        start_check_at = max(len(text) - self.len_longest_word, 0) #5-3 =2

        for i in range(start_check_at, len(text)): #2
            potential_word = text[i:] #ape
            if not self.is_word_in_lexicon(potential_word):
                continue
            # print(f"Slowly searching for if '{potential_word}' as a good word candidate")

            part_of_text_before_potential_word = text[:i] #an
            best_score_before_potential_word, best_segmentation_before_potential_word = \
                self._get_best_segmentation_slow(part_of_text_before_potential_word) #best score of an, an

            potential_score = \
                best_score_before_potential_word + self.get_log_probability_for_word(potential_word)
            segmentation_candidates.append(
                (potential_score,
                    best_segmentation_before_potential_word + " " + potential_word))

        # Find max score based on first element in tuple.
        best_candidate = max(segmentation_candidates, key=lambda t: t[0])
        # print(f"Possible candidates for '{text}' were:")
        # print{f"{segmentation_candidates}. The winning was: '{best_candidate[1]}'")
        return best_candidate

    def format_and_check_text(self, text):
        """
        Removes newlines in end and beginning of text.
        Also check there are no other newlines
        
        Parameters:
            text (str): The text to format/check. 
        """
        text = text.strip()
        if "\n" in text:
            raise ValueError("The text to segment cannot contain a newline")
        return text

    def segment(self, text):
        """
        Returns the best segmentation for a text, 
        
        Parameters:
            text (str): The text to segment. 
        Returns: 
            str: The best segmentation for text, indicated by whitespace inserted in text
        """
        text = self.format_and_check_text(text)
        (_, best_segmentation) = self._get_best_segmentation_slow(text)
        return best_segmentation.strip()


if __name__ == '__main__':

    lexicon_filename = sys.argv[1]
    text_to_segment_filename = sys.argv[2]

    bws = BasicWordSegmenter(lexicon_filename)

    with open(text_to_segment_filename, encoding="utf-8") as text_file:
        for line in text_file:
            try:
                print(bws.segment(line.strip()))
            except ValueError:
                print()
                print("Line too long or contains newline: ", line.strip(), file=sys.stderr)
