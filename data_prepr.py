import sys
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing



#ENGLISH_STOP_WORDS is a frozenset.Transform it to set in order to add words later
stopword_set=set(ENGLISH_STOP_WORDS)
stopword_set.update({"just","did","need","make","come","like","really","ago","later","took","thing",
					 "got","getting","know","does","doing","going","lot","seen","saw","let","use","used","having",
					 "that's","there's","didn't","can't","it's","she's","she'd","he's","he'd","don't","doesn't",
					 "I'm","I've","I'd","you're","they're","haven't","hasn't"})


def data_preprocessing(input_file,title_weight=1,lsi=True,n_comp=100,ret_vectorizers=False):
	#read dataset
	train_data=pd.read_csv(input_file,sep="\t")

	#label encoder for categories
	#category_labels ->numeric representation of categories of  the train data
	le = preprocessing.LabelEncoder()
	category_labels=le.fit_transform(train_data["Category"])

	#vectorize content and titles
	count_vectorizer = CountVectorizer(stop_words=stopword_set)
	data_count_matrix = count_vectorizer.fit_transform( title_weight*(train_data['Title']+" ") + train_data['Content'])
	'''
	for word in count_vectorizer.vocabulary_:
		print word
	print("\n")
	'''
	#transform count vectors to  frequency vectors
	tfidf_transformer = TfidfTransformer(sublinear_tf=True, use_idf=True)
	data_tfidf_matrix = tfidf_transformer.fit_transform(data_count_matrix)

	if lsi==False:
		return [data_tfidf_matrix,category_labels]
	else:
		#use LSI on vector matrix
		svd = TruncatedSVD(n_components=n_comp)
		data_redu_matrix = svd.fit_transform(data_tfidf_matrix)
		
		if ret_vectorizers==True:
			return [data_redu_matrix,category_labels,count_vectorizer,tfidf_transformer,svd,le]
		else:
			#return matrix to train classifier
			return [data_redu_matrix,category_labels]
	


def btb_data_preprocessing(input_file,title_weight=1,lsi=True,n_comp=100,ret_vectorizers=False):
	#read dataset
	train_data=pd.read_csv(input_file,sep="\t")

	#label encoder for categories
	#category_labels ->numeric representation of categories of  the train data
	le = preprocessing.LabelEncoder()
	category_labels=le.fit_transform(train_data["Category"])

	#vectorize content and titles
	count_vectorizer = CountVectorizer(tokenizer=my_tokenizer,max_df=0.85,ngram_range=(1,1))
	data_count_matrix = count_vectorizer.fit_transform( title_weight*(train_data['Title']+" ") + train_data['Content'])
	'''
	for word in count_vectorizer.vocabulary_:
		print word
	print("\n")
	'''
	#transform count vectors to  frequency vectors
	tfidf_transformer = TfidfTransformer(sublinear_tf=True, use_idf=True)
	data_tfidf_matrix = tfidf_transformer.fit_transform(data_count_matrix)

	if lsi==False:
		return [data_tfidf_matrix,category_labels]
	else:
		#use LSI on vector matrix
		svd = TruncatedSVD(n_components=n_comp)
		data_redu_matrix = svd.fit_transform(data_tfidf_matrix)
		
		if ret_vectorizers==True:
			return [data_redu_matrix,category_labels,count_vectorizer,tfidf_transformer,svd,le]
		else:
			#return matrix to train classifier
			return [data_redu_matrix,category_labels]



import re
from nltk import word_tokenize,pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


def my_tokenizer(text):
	tokens = word_tokenize(text)
	tokens = remove_punctuation(tokens)
	tokens = remove_digits(tokens)

	#stemming
	#tokens = remove_stopwords(tokens)
	#tokens = remove_n_char_words(tokens,2)
	#tokens = stem_words(tokens)

	#or lemmatization
	tokens = lemmatize_verbs(tokens)
	tokens = remove_stopwords(tokens)
	tokens = remove_n_char_words(tokens,2)

	return tokens


#remove punctuation from list of words
def remove_punctuation(words):
	new_words = []
	for word in words:
		new_word = re.sub(r'[^\w\s]', '', word)
		if new_word != '':
			new_words.append(new_word)
	return new_words

#remove digits and words containing digits from list of words
def remove_digits(words):
	new_words = []
	for word in words:
		if word.isalpha()==True:
			new_words.append(word)
	return new_words

#remove n char words from list of words
def remove_n_char_words(words,n):
	new_words = []
	for word in words:
		if len(word)>n:
			new_words.append(word)
	return new_words

#remove stopwords from list of words
def remove_stopwords(words):
	new_words = []
	for word in words:
		if word not in stopword_set:
			new_words.append(word)
	return new_words

#apply stemming to list of words
def stem_words(words):
	stemmer = SnowballStemmer('english')
	stems = []
	for word in words:
		stem = stemmer.stem(word)
		stems.append(stem)
	return stems

#apply lemmatization of verbs to list of words
def lemmatize_verbs(words):
	"""Lemmatize verbs in list of tokenized words"""
	lemmatizer = WordNetLemmatizer()
	lemmas = []
	for word in words:
		lemma = lemmatizer.lemmatize(word, pos='v')
		lemmas.append(lemma)
	return lemmas

#apply lemmatization to list of words
def lemmatize_words(words):
	lemmatizer = WordNetLemmatizer()
	pos_tagged=pos_tag(words)
	lemmas = []
	for word,tag in pos_tagged:
		wn_tag=get_wordnet_pos(tag)
		lemma = lemmatizer.lemmatize(word,wn_tag)
		lemmas.append(lemma)
	return lemmas

#transform treebank pos tag to wordnet pos tag
def get_wordnet_pos(treebank_tag):
	if treebank_tag.startswith('J'):
		return wordnet.ADJ
	elif treebank_tag.startswith('V'):
		return wordnet.VERB
	elif treebank_tag.startswith('N'):
		return wordnet.NOUN
	elif treebank_tag.startswith('R'):
		return wordnet.ADV
	else:
		return wordnet.NOUN