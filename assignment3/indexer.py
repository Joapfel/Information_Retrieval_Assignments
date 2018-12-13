import heapq
from typing import *
from nltk.corpus import stopwords
from emoji import UNICODE_EMOJI
from string import punctuation
from nltk.tokenize import TweetTokenizer


class TwitterIQ(dict):
	"""
	This class creates an inverted index of a .csv file
	that contains information and content from a number of tweets.

	This class inherits from the default dictionary class in Python.
	This was done primarily to utilize the __missing__ function,
	which serves to make the code a bit more Pythonic. The class loads
	each token individually and uses the __missing__ method to determine
	whether it is already in the dictionary (self). In the case the it
	is, the id of the particular tweet/doc is added to the relevant
	postings list.

	Query methods are detailed in their respective methods' docstrings.

	Attributes:
		all_postings: List containing ALL postings lists
		tweet_content_dict: Dictionary whose keys are twitter_ids
			pointing to the content of the tokenized tweets.
		__current_tweet_id: ID of the doc/tweet that is currently
			being iterated over. This is used by the __missing__ method
			to propery organize the dictionary
	"""

	STOP_WORDS = stopwords.words('english') + stopwords.words('german')
	EXCLUSION_LIST = list(punctuation) + list(UNICODE_EMOJI.keys()) + ['...', 'de', 'com']

	def __init__(self, path: str = None, strip_handles: bool = True, **kwargs):
		"""
		Initializes by walking through each token and creating an
		inverted index as detailed above.

		:param str path: the path to the string
		"""
		super().__init__(**kwargs)
		self.all_postings = []
		self.tweet_content_dict = {}
		self.tokenizer = TweetTokenizer(strip_handles=strip_handles)
		self.__indexing = False
		self.length = 0

		if path:
			self.index(path)

	def __missing__(self, token: str):
		"""
		If entry is missing, add posting to all_postings, then creates
		a PostingNode and sets the entry to equal that value.

		:param str token: The token to be added to the dictionary
		:return: posting
		"""
		if self.__indexing:
			self.all_postings.append([self.length])
			self[token] = PostingNode(self.all_postings[-1])
			return self[token]
		else:
			return PostingNode([])

	def __index_tokens(self, tweet_content: list) -> None:
		"""
		Walks the the list of tokens from each tweet. If a token
		is not clean, it exits without adding an entry. It then
		adds to (or uses the __missing__ method to create) the
		dictionary (self) entry for each token. It then adds to the
		posting list of each token.

		:param list tweet_content: a list of tokens from tweet
		:returns: None
		"""
		for token in tweet_content:
			token = self.__clean(token)
			if not token:
				return

			# creates entry or assigns posting_node to existing one
			posting_node = self[token]
			if self.length not in posting_node.postings_list:
				# adds to end of posting list and increments freq
				posting_node.postings_list.append(self.length)
				posting_node.freq += 1

	def __clean(self, token: str) -> Union[str, None]:
		"""
		Removes stop words for English and German and makes
		all tokens lowercase.

		:param str token: the token to be cleaned
		:return: The token if it can be cleaned, None if not
		:rtype: str
		"""

		if token in TwitterIQ.EXCLUSION_LIST or token.startswith(('http')):
			return

		if token.startswith('#'):
			token = token[1:]

		token = token.lower()

		if token in TwitterIQ.STOP_WORDS:
			return

		return token

	def get_tokens_from_tweet(self, tweet_id: int) -> str:
		"""
		Returns the tokens from a specific tweet/doc ID

		:param int tweet_id: the tweet to be returned
		:return: a list of tokens from a tweet
		:rtype: List[str]
		"""
		return self.tweet_content_dict[tweet_id]

	def get_most_freq_words(self, n: int = 10) -> List[str]:
		"""
		Returns the words with the three largest frequencies.

		:param int n: the optional n number of words to return
		:return: the most frequently used words in the corpus
		:rtype: list
		"""
		return heapq.nlargest(n, self, key=lambda x: self[x])

	def index(self, path: str) -> None:
		self.__indexing = True

		with open(path, 'r') as corpus:
			# combs through each doc/tweet individually

			for doc in corpus:
				raw_tweet = doc.split('\t')[4]
				self.tweet_content_dict[self.length] = raw_tweet
				tokenized_doc = self.tokenizer.tokenize(raw_tweet)

				self.__index_tokens(tokenized_doc)
				self.length += 1

		self.__indexing = False

	def query(self, term1: str, term2: str = None) -> List[int]:
		"""
		Gets the postings list for a term or the intersection or the
		posting lists of two different terms.

		:param str term1: the first (or only) string to query
		:param str term2: the optione 2nd string to intersect with
		:return: the positings list or intersection of two
		"""
		if term2 is None:
			return self[term1].postings_list
		else:
			return list(set(self.query(term1)) & set(self.query(term2)))

	def print_query(self, term1: str, term2: str = None) -> None:
		"""
		Pretty prints the query method.

		:param str term1: the first term to query
		:param str term2: the optional second term to intersect with
		"""
		for tweet_id in self.query(term1, term2):
			print(f'{tweet_id}:', self.tweet_content_dict[tweet_id])

	def __len__(self):
		return self.length


class PostingNode(object):
	"""
	A node that contains a reference to the postings list and the
	length of that posting list, or the word's frequency
	"""

	def __init__(self, postings_list: list):
		self.postings_list = postings_list
		self.freq = len(postings_list)

	def __str__(self) -> str:
		ret = f'(Frequency: {self.freq}, {self.postings_list[:5]}'
		if self.freq > 5:
			ret += '...'
		ret += ')'

		return ret

	def __repr__(self) -> str:
		return 'PostingNode' + self.__str__()

	def __ne__(self, other) -> bool:
		return self.freq == other.freq

	def __gt__(self, other) -> bool:
		return self.freq > other.freq