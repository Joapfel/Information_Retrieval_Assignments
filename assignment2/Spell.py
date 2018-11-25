import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())


def kgrams(w, k): return {w[i:i+k] for i,v in enumerate(w) if i < len(w)-(k-1)}
#WORDS = Counter(words(open('big.txt').read()))
WORDS = list(map(lambda x:x.strip(), open('englishdic.sec').readlines()))
WORDS2KGRAMS = {w:kgrams(w,2) for w in WORDS}
#def P(word, N=sum(WORDS.values())):
#    "Probability of `word`."
#    return WORDS[word] / N

def correction(word):
    "Most probable spelling correction for word."
    #return max(candidates(word), key=P)
    return candidates(word)[0]

def candidates(word):
    "Generate possible spelling corrections for word."
    #print(known([word]))
    #print(known(edits1(word)))
    #print(known(edits2(word)))
    print (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))



if __name__ == '__main__':
    #print(kgrams('going', 2))
    #print(WORDS2KGRAMS)
    correction('goinkgg')