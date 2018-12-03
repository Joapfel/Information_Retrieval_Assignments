import sys
import csv
import string
import re
import emoji
import nltk
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from spell_checker import SpellChecker

class Index:
    """
    This data structure is the value of the indices dictionary.
    """
    def __init__(self, size, pointer2postingsList):
        # size of the postings list
        
        self.size = size
        # pointer to the head of the postings list
        self.pointer2postingsList = pointer2postingsList

class PostingNode:
    """
    Linked list for the postings list
    """
    def __init__(self, val):
        self.val = val
        self.next = None

class TwitterIR(object):
    """
    Main Class for the information retrieval task.
    """
    __slots__ = 'id2doc', 'tokenizer', 'unicodes2remove', 'indices', \
                'urlregex', 'punctuation', 'emojis', 'stop_words', \
                'engSpellCheck', 'gerSpellCheck', 'correctedTerms'

    def __init__(self):
        # the original mapping from the id's to the tweets, 
        # which is kept until the end to index the tweets
        self.id2doc = {}
        self.tokenizer = TweetTokenizer()
        # bunch of punctuation unicodes which are not in 'string.punctuation'
        self.unicodes2remove = [
            # all kinds of quotes
            u'\u2018', u'\u2019', u'\u201a', u'\u201b', u'\u201c', \
            u'\u201d', u'\u201e', u'\u201f', u'\u2014',
            # all kinds of hyphens
            u'\u002d', u'\u058a', u'\u05be', u'\u1400', u'\u1806', \
            u'\u2010', u'\u2011', u'\u2012', u'\u2013',
            u'\u2014', u'\u2015', u'\u2e17', u'\u2e1a', u'\u2e3a', \
            u'\u2e3b', u'\u2e40', u'\u301c', u'\u3030',
            u'\u30a0', u'\ufe31', u'\ufe32', u'\ufe58', u'\ufe63', \
            u'\uff0d', u'\u00b4'
        ]
        # the resulting data structure which has the tokens as keys
        # and the Index objects as values
        self.indices = {}
        # regex to match urls (taken from the web)
        self.urlregex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]'
                                   '|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # keep @ to be able to recognize usernames
        self.punctuation = string.punctuation.replace('@', '') + \
                           ''.join(self.unicodes2remove)
        self.punctuation = self.punctuation.replace('#', '')
        self.punctuation = self.punctuation.replace('…', '')
        # a bunch of emoji unicodes
        self.emojis = ''.join(emoji.UNICODE_EMOJI)
        self.emojis = self.emojis.replace('#', '')
        # combined english and german stop words
        self.stop_words = set(stopwords.words('english') + stopwords.words('german'))
        self.engSpellCheck = self._initSpellCheck('english')
        self.gerSpellCheck = self._initSpellCheck('german')
        self.correctedTerms = []    # For demonstration purposes only

    def clean(self, s):
        """
        Normalizes a string (tweet) by removing the urls, punctuation, digits,
        emojis, by putting everything to lowercase and removing the
        stop words. Tokenization is performed aswell.

        :param s the string (tweet) to clean
        :return: returns a list of cleaned tokens
        """
        s = ' '.join(s.replace('[NEWLINE]', '').split())
        s = ' '.join(s.replace('…', '...').split())
        s = self.urlregex.sub('', s).strip()
        s = s.translate(str.maketrans('', '', self.punctuation + string.digits \
                                      + self.emojis)).strip()
        s = ' '.join(s.split())
        s = s.lower()
        s = self.tokenizer.tokenize(s)
        s = [w for w in s if w not in self.stop_words]
        return s

    def _detectLanguage(self, context):
        """
        Detects the language of a tweet based on a hierarchy of criteria:
        1. the number of stopwords from each language in a tweet
        2. the number of "normal" words from each language in a tweet
        3. the more common language, in this case English
        :param context: 
        :return: the determined language of the tweet
        """
        tokens = self.tokenizer.tokenize(context)
        stopsEN = [token for token in tokens if token in stopwords.words('english')]
        stopsDE = [token for token in tokens if token in stopwords.words('german')]

        # Chooses a language based on the number of stopwords
        if len(stopsEN) > len(stopsDE):
            return 'english'
        elif len(stopsDE) > len(stopsEN):
            return 'german'
        # If that comparison isn't conclusive, it compares the number of words
        # that exist in the respective dictionaries.
        else:
            cleaned = self.clean(context)

            wordsEN = [token for token in cleaned if self.engSpellCheck.in_dictionary(token)]
            wordsDE = [token for token in cleaned if self.gerSpellCheck.in_dictionary(token)]

            if len(wordsEN) > len(wordsDE):
                return 'english'
            elif len(wordsDE) > len(wordsEN):
                return 'german'
            # If it still cannot decide, it defaults to the more common language: English
            else:
                return 'english'

    @staticmethod
    def _getGermanFreqDist():
        """
        Walks through germanfreq.txt, splits the values into terms and frequencies
        then maps them to a dictionary.

        Ich	489637 -> {ich: 489637}
        ist	475043 -> {ich: 489637, ist: 475043}
        ich	440346 -> {ich: 929983, ist: 475043} - adds to the count of capital 'ich'

        :return: a frequency distribution dictionary of German words
        """
        with open('germanfreq.txt', 'r') as f:
            fdist = {}
            lines = f.read().splitlines()
            for line in lines:
                parts = line.split()

                # List comprehension is necessary because the file is sometimes organize "term freq" and sometimes
                # "freq term" and it only works because there are no cardinal number terms included in the file.
                term = [p for p in parts if not p.isdigit()][0]
                freq = int([p for p in parts if p.isdigit()][0])

                # The entries are split into capital and lowercase; here we combine the two.
                try:
                    fdist[term] = fdist[term] + freq
                except KeyError:
                    fdist[term] = freq

            return fdist

    def _getTokens2ids(self):
        """
        Indexes all the tokens and maps them to a list of tweetIDs.
        :return: a dictionary visualized as {token: [tweetID1, tweetID2, ...]}
        """
        # For the sake of time and presenting functionality, we're limiting the number
        # of tweets that we are indexing.
        MAX_DOCS_TO_INDEX = 25
        i = 0

        tokens2id = {}

        for id, doc in self.id2doc.items():
            doc = self.clean(doc)
            language = self._detectLanguage(' '.join(doc))

            # This print statement is for demonstration purposes 
            print(language, doc)

            for t in doc:
                if language == 'english':
                    # We are specifically excluding handles and hashtags
                    # Nor do we want to spellcheck words that are in the dictionary
                    if t[0] not in ['@', '#'] and not self.engSpellCheck.in_dictionary(t):
                        original = t
                        t = self.spellCheck(t, language)

                        # Collects corrected words for demonstration purposes
                        if original != t:
                            self.correctedTerms.append((original, t))

                elif language == 'german':
                    if t[0] not in ['@', '#'] and not self.gerSpellCheck.in_dictionary(t):
                        original = t
                        t = self.spellCheck(t, language)

                        if original != t:
                            self.correctedTerms.append((original, t))

                if t in tokens2id.keys():
                    tokens2id[t].add(id)
                else:
                    # a set is used to avoid multiple entries of the same tweetID
                    tokens2id[t] = {id}

            # Break the loop after MAX_DOCS_TO_INDEX iterations
            i += 1
            if i >= MAX_DOCS_TO_INDEX:
                break

        return tokens2id

    def index(self, path):
        """
        1) call the method to read the file in
        2) iterate over the original datastructure id2doc which keeps the mapping
        of the tweet ids to the actual tweets and do:
            2a) preprocessing of the tweets
            2b) create a mapping from each token to its postings list (tokens2id)
        3) iterate over the just created mapping of tokens to their respective 
        postings lists (tokens2id) and do:
            3a) calculate the size of the postingslist
            3b) sort the postings list numerically in ascending order
            3c) create a linked list for the postings list
            3d) create the Index object with the size of the postings list and
            the pointer to the postings list - add to the resulting datastructure 
        :param path: the path to the tweets.csv file
        :return:
        """
        self.initId2doc(path)
        self._indexPostings(self._getTokens2ids())

    def _indexPostings(self, tokens2id):
        """
        Creates an `Index` object, which contains a pointer to to the beginning
        of a postings list for every key/token in the `tokens2id` dictionary. It
        stores this in the master inverted index `self.indices`.

        :param tokens2id: 
        """
        for t, ids in tokens2id.items():
            # size of the postings list which belongs to token t
            size = len(ids)
            # sort in ascending order
            ids = sorted(ids)
            # use the first (and smallest) tweetID to be the head node of the 
            # linked list
            node = PostingNode(ids[0])
            # keep reference to the head of the linked list since node variable
            # is going to be overridden
            pointer = node
            for id in ids[1:]:
                # create further list items
                n = PostingNode(id)
                # and append to the linked list
                node.next = n
                # step further
                node = n
            # create the index object with size of the postings list 
            # and a link to the postings list itself
            i = Index(size, pointer)
            self.indices[t] = i

    def initId2doc(self, path):
        """
        Reads the file in and fills the id2doc datastructure.
        :param path: path to the tweets.csv file
        :return:
        """
        with open(path, 'r', encoding='utf-8', newline='') as f:
            r = csv.reader(f, delimiter='\t')
            for line in r:
                self.id2doc[line[1]] = line[4]
        f.close()

    def _initSpellCheck(self, lang):
        """
        Initializes two `SpellChecker` objects given the path to their dictionary files.

        :param lang: the language of the spell checker 
        :return: a `SpellChecker` object based on a dictionary in that language
        """

        if lang == 'english':
            # `SpellChecker` will use the Brown FreqDist if none is provided
            freq_dist = None
        elif lang == 'german':
            # For German, we need to create our own.
            freq_dist = self._getGermanFreqDist()
        else:
            raise Exception(f'{lang} is not a supported language.')

        return SpellChecker(SpellChecker.DEFAULT_DICTIONARIES[lang], fdist=freq_dist)

    def intersect(self, pointer1, pointer2):
        """
        Computes the intersection for two postings lists.
        :param pointer1: first postings list
        :param pointer2: second postings list
        :return: returns the intersection 
        """
        # create temporary head node
        node = PostingNode('tmp')
        # keep reference to head node
        rvalpointer = node
        while pointer1 and pointer2:
            val1 = pointer1.val
            val2 = pointer2.val
            # only append to the linked list if the values are equal
            if val1 == val2:
                n = PostingNode(val1)
                node.next = n
                node = n
                pointer1 = pointer1.next
                pointer2 = pointer2.next
            # otherwise the postings list with the smaller value 
            # at the current index moves one forward
            elif val1 > val2:
                pointer2 = pointer2.next
            elif val1 < val2:
                pointer1 = pointer1.next
        # return from the second element on since the first was the temporary one
        return rvalpointer.next

    def _query(self, term, lang):
        """
        Internal method to query for one term.
        :param: term the word which was queried for 
        :param: lang the language of the term for spellchecking
        :return: returns the Index object of the corresponding query term
        """
        if lang == 'english':
            if not self.engSpellCheck.in_dictionary(term):
                term = self.spellCheck(term, lang)
        elif lang == 'german':
            if not self.gerSpellCheck.in_dictionary(term):
                term = self.spellCheck(term, lang)

        try:
            return self.indices[term]
        except KeyError:
            return Index(0, PostingNode(''))

    def query(self, *arg):
        """
        Query method which can take any number of terms as arguments.
        It uses the internal _query method to get the postings lists for the single 
        terms. It calculates the intersection of all postings lists.
        :param *arg term arguments
        :return: returns a list of tweetIDs which all contain the query terms
        """
        language = self._detectLanguage(' '.join([t for t in arg]))
        print(language)  # For demonstration

        # at this point it's a list of Index objects
        pointers = [self._query(t, language) for t in arg if t not in self.stop_words]
        # here the Index objects get sorted by the size of the 
        # postings list they point to
        pointers = sorted(pointers, key=lambda i: i.size)
        # here it becomes a list of pointers to the postings lists
        pointers = [i.pointer2postingsList for i in pointers]
        # first pointer
        intersection = pointers[0]
        # step through the pointers
        for p in pointers[1:]:
            # intersection between the new postings list and the so far
            # computed intersection
            intersection = self.intersect(intersection, p)
            # if at any point the intersection is empty there is 
            # no need to continue
            if not intersection:
                return []
        # convert the resulting intersection to a normal list
        rval = []
        pointer = intersection
        while pointer:
            rval.append(pointer.val)
            pointer = pointer.next

        return rval

    def spellCheck(self, term, lang):
        """Runs the relevant spellchecker method."""
        return {'english': self.engSpellCheck,
                'german': self.gerSpellCheck}[lang].spell_check(term)

    def __len__(self):
        """The number of tokens in the inverted index."""
        return len(self.indices.keys())

