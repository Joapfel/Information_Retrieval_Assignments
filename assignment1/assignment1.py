import sys
import csv
import string
import re
import emoji
import nltk
#nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

class Index:
    """
    This data structure is the value of the indices dictionary
    """
    def __init__(self, size, pointer2postingsList):
        self.size = size
        self.pointer2postingsList = pointer2postingsList

class PostingNode:
    """
    Linked list for the postings list
    """
    def __init__(self, val):
        self.val = val
        self.next = None


class TwitterIR(object):
    __slots__ = 'id2doc', 'tokenizer', 'unicodes2remove', 'indices', 'cleanedPostings', 'urlregex', 'punctuation', 'emojis', 'stop_words'

    def __init__(self):
        self.id2doc = {}
        self.tokenizer = TweetTokenizer()
        self.unicodes2remove = [
            # all kinds of quotes
            u'\u2018', u'\u2019', u'\u201a', u'\u201b', u'\u201c', u'\u201d', u'\u201e', u'\u201f', u'\u2014',
            # all kinds of hyphens
            u'\u002d', u'\u058a', u'\u05be', u'\u1400', u'\u1806', u'\u2010', u'\u2011', u'\u2012', u'\u2013',
            u'\u2014', u'\u2015', u'\u2e17', u'\u2e1a', u'\u2e3a', u'\u2e3b', u'\u2e40', u'\u301c', u'\u3030',
            u'\u30a0', u'\ufe31', u'\ufe32', u'\ufe58', u'\ufe63', u'\uff0d'
        ]
        self.indices = {}
        self.cleanedPostings = []
        self.urlregex = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.punctuation = string.punctuation.replace('@', '') + ''.join(self.unicodes2remove)
        self.emojis = ''.join(emoji.UNICODE_EMOJI)
        self.stop_words = set(stopwords.words('english') + stopwords.words('german'))

    def initId2doc(self, path):
        """

        :param path:
        :return:
        """
        with open(path, 'r', encoding='utf-8', newline='') as f:
            r = csv.reader(f, delimiter='\t')
            for line in r:
                self.id2doc[line[1]] = line[4]
        f.close()

    def clean(self, s):
        s = self.urlregex.sub('', s).strip()
        s = s.translate(str.maketrans('', '', self.punctuation + string.digits + self.emojis)).strip()
        s = ' '.join(s.split())
        s = s.lower()
        self.cleanedPostings.append(s)
        s = self.tokenizer.tokenize(s)
        s = [w for w in s if w not in self.stop_words]
        return s

    def index(self, path):
        self.initId2doc(path)
        tokens2id = {}
        for id, doc in self.id2doc.items():
            doc = self.clean(doc)
            for t in doc:
                if t in tokens2id.keys():
                    tokens2id[t].add(id)
                else:
                    tokens2id[t] = {id}

        for t, ids in tokens2id.items():
            size = len(ids)
            ids = sorted(ids)
            node = PostingNode(ids[0])
            pointer = node
            for id in ids[1:]:
                n = PostingNode(id)
                node.next = n
                node = n
            i = Index(size, pointer)
            self.indices[t] = i

    def _query(self, term):
        if term in self.indices:
            return self.indices[term]
        return Index(0, PostingNode(''))

    def query(self, *arg):
        pointers = [self._query(t) for t in arg if t not in self.stop_words]
        pointers = sorted(pointers, key=lambda i: i.size)
        pointers = [i.pointer2postingsList for i in pointers]
        intersection = pointers[0]
        for p in pointers[1:]:
            intersection = self.intersect(intersection, p)
            if not intersection:
                return []
        rval = []
        pointer = intersection
        while pointer:
            rval.append(pointer.val)
            pointer = pointer.next
        return rval

    def intersect(self, pointer1, pointer2):
        node = PostingNode('tmp')
        rvalpointer = node
        while pointer1 and pointer2:
            val1 = pointer1.val
            val2 = pointer2.val
            if val1 == val2:
                n = PostingNode(val1)
                node.next = n
                node = n
                pointer1 = pointer1.next
                pointer2 = pointer2.next
            elif val1 > val2:
                pointer2 = pointer2.next
            elif val1 < val2:
                pointer1 = pointer1.next
        return rvalpointer.next


if __name__ == '__main__':
    twitterIR = TwitterIR()
    twitterIR.index('tweets.csv')

    index = twitterIR.query('nacht', 'schlafen')
    print(len(index))

    for id in index:
        print(twitterIR.id2doc[id])

