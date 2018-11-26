from string import ascii_lowercase


class SpellChecker(object):

    DEFAULT_DICTIONARIES = {'english': open('englishdic.sec', 'r').read().splitlines(),
            'german': open('germandic-utf8.sec', 'r').read().splitlines()}

    def __init__(self, lang_vocab: list, fdist: dict = None, max_edit_distance: int = 2):
        self.lang_vocab = {letter: [word.lower() for word in lang_vocab if word.startswith(letter)]
                for letter in ascii_lowercase}
        self.fdist = fdist
        self.max_edit_distance = max_edit_distance
        
        if self.fdist is None: 
            if lang_vocab == SpellChecker.DEFAULT_DICTIONARIES['english']:
                from nltk.corpus import brown
                from nltk import FreqDist

                self.fdist = FreqDist(w.lower() for w in brown.words(categories='news'))

        else:
            raise TypeError('No frequency distribution index provided.')

    def word_probability(self, word: str) -> int: 
        """Divides the frequency of a word by overall token count."""
        return self.fdist[word] / len(self.fdist.keys())

    def spell_check(self, word: str) -> str:
        """Sorts the list of remaining words based on their probability in the corpus."""
        return max(self.candidates(word), key=self.word_probability)

    def candidates(self, word: str) -> tuple:
        """Filters out all words not in the language's vocabulary."""
        return (self.known([word]) or                       # word if it is known
                self.known(self.edit_distance1(word)) or    # known words with edit distance 1
                self.known(self.edit_distance2(word)) or    # known words with edit distance n
                [word])                                     # word, unknown

    def known(self, words: list) -> set:
        """Returns all the words of a list in the language's lexicon."""
        return set(w for w in words if len(w) > 1 and w in self.lang_vocab[w[0].lower()])

    @staticmethod
    def edit_distance1(word: str) -> set:
        """
        Thanks to Peter Norvig (http://norvig.com/spell-correct.html)

        Creates all the possible letter combinations that can be made
        with an edit distance of 1 to the word.

        splits = all ways of dividing the word, e.g. 
            'word' -> ('w', 'ord'); useful for making changes
        deletions = all ways of removing a single letter, e.g.
            'word'-> 'ord'
        transpositions = all ways of swapping two letters immediately
            adjacent to one another, e.g. 'word' -> 'owrd'
        replacements = all ways of replacing a letter with another
            letter, e.g. 'word' -> 'zord'
        insertions = all ways of inserting a letter at any point in the
            word, e.g. 'word' -> 'wgord'

        :param str word: the relevant word
        :return: a set of terms with an edit distance of 1 from the word
        :rtype: set
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletions = [left + right[1:] for left, right in splits if right]
        transpositions = [left + right[1] + right[0] + right[2:]
                for left, right in splits if len(right) > 1]
        replacements = [left + letter + right[1:] for left, right
                in splits if right for letter in ascii_lowercase]
        insertions = [left + letter + right for left, right in splits
                for letter in ascii_lowercase]

        return set(deletions + transpositions + replacements + insertions)

    def edit_distance2(self, word: str) -> tuple:
        #TODO: Should be obsolete now
        """Simply runs `edit_distance1` on every result from `edit_distance1(word)`"""
        return (edit2 for edit in self.edit_distance1(word) for edit2
            in self.edit_distance1(edit))
        
    def edit_distanceN(self, word: str) -> set:
        """Runs `edit_distance1` on the results of `edit_distance1` n times."""
        ret_val = set(word)

        for _ in range(self.max_edit_distance):
            for val in ret_val:
                ret_val = ret_val | self.edit_distance1(val)

        return ret_val

