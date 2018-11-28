class SpellChecker(object):
    DEFAULT_DICTIONARIES = {'english': open('englishdic.sec', 'r').read().splitlines(),
                            'german': open('germandic-utf8.sec', 'r').read().splitlines()}

    def __init__(self, lang_vocab: list, fdist: dict = None, max_edit_distance: int = 2):
        """
        Creates a `SpellChecker` object from a dictionary and a frequency distribution. The principle
        method is `spell_check`. Every other method is called in calling it. To best understand this
        class, start there and work your way through the call sequence: `spell_check` calls
        `candidates` which in turns calls `known` and the edit distance methods.

        :param lang_vocab: a list of words in a language's vocabulary
        :param fdist: a dictionary-like frequency distribution from a large corpus;
                if none is provided and the class default English dictionary is supplied to
                `lang_vocab`, the Brown corpus is imported and used
        :param max_edit_distance: the maximum edit distance at which words will still be considered
        """
        # At present, sticking with the German alphabet even for English
        self.alphabet = 'aäbcdefghijklmnoöpqrsßtuüvwxyz'
        # Key:Value pair of alphabet letters and lists of words beginning with those letters
        # in the provided list of dictionary terms to decrease the time it takes to do dictionary lookups.
        self.lang_vocab = {letter: [word.lower() for word in lang_vocab \
                                    if word.lower().startswith(letter)] for letter in self.alphabet}
        self.fdist = fdist
        self.max_edit_distance = max_edit_distance

        # If no fdist provided and `lang_vocab` is default English, use the Brown news corpus'.
        # In case you don't have a corpus big enough to create a strong frequency distribution
        if self.fdist is None:
            if lang_vocab == SpellChecker.DEFAULT_DICTIONARIES['english']:
                from nltk.corpus import brown
                from nltk import FreqDist

                self.fdist = FreqDist(w.lower() for w in brown.words(categories='news'))

            else:
                raise TypeError('No frequency distribution index provided.')

    def candidates(self, word: str) -> set:
        """
        Returns words within an edit distance of 2 in a ranked order, only generating words
        if there are no results from the previous method. If the word does not begin with
        a letter in `self.alphabet`, it is returned immediately as it was given.
        """
        try:
            return (self.known([word.lower()]) or                     # word if it is known
                    self.known(self.edit_distance1(word.lower())) or  # known words with edit distance 1
                    self.known(self.edit_distance2(word.lower())) or  # known words with edit distance 2
                    [word])                                           # word, unknown
        except KeyError:
            return [word]

    def edit_distance1(self, word: str) -> set:
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
                        in splits if right for letter in self.alphabet]
        insertions = [left + letter + right for left, right in splits
                      for letter in self.alphabet]

        return set(deletions + transpositions + replacements + insertions)

    def edit_distance2(self, word: str) -> set:
        """Simply runs edit_distance1 on every result from edit_distance1(word)"""
        return set(edit2 for edit in self.edit_distance1(word) for edit2
                   in self.edit_distance1(edit))

    def edit_distanceN(self, word: str) -> set:
        # FIXME
        """Runs `edit_distance1` on the results of `edit_distance1` n times."""
        ret_val = set(word)

        for _ in range(self.max_edit_distance):
            for val in ret_val:
                ret_val = ret_val | self.edit_distance1(val)

        return ret_val

    def in_dictionary(self, word: str) -> bool:
        """Returns whether the word is in the dictionary."""
        try:
            return word in self.lang_vocab[word[0].lower()]
        except KeyError:
            return False

    def known(self, words: list) -> set:
        """
        Walks through words in a list, checks them against `lang_vocab`,
        and returns a set of those that match.
        """
        return set(w for w in words if len(w) > 1 and self.in_dictionary(w))

    def spell_check(self, word: str) -> str:
        """Chooses the most likely word in a set of candidates based on `word_probability`."""
        return max(self.candidates(word), key=self.word_probability)

    def word_probability(self, word: str) -> int:
        """Divides the frequency of a word by overall token count."""
        try:
            return self.fdist[word.lower()] / len(self.fdist.keys())
        except KeyError:
            return 0

