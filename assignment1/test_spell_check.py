from spell_checker import *

with open('english_test_corpus.txt') as t:
    s = SpellChecker(open('englishdic.sec').read().splitlines())
    print(len(s.fdist.keys()))
    for line in t:
        print(f'{line[:-1]}\t{s.spell_check(line[:-1])}')
