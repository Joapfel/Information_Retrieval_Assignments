from twitterir import *

def main():
    twitterIR = TwitterIR()
    twitterIR.index('tweets.csv')

    for original, corrected in twitterIR.correctedTerms:
	print(f"'{original}' -> '{corrected}'.")

    query_result = twitterIR.query('blutbilt', 'schwer')
    print(query_result)
    print(twitterIR.id2doc[query_result[0]])
    
    query_result = twitterIR.query('major', 'senters', 'authers')
    print(query_result)
    print(twitterIR.id2doc[query_result[0]])

if __name__ == '__main__':
    main()
