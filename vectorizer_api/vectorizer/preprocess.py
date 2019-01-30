from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def train_tokenize(df):
	tokenizer = RegexpTokenizer(r'\w+')
	df["tokens"] = df["text"].apply(tokenizer.tokenize)

	max_length = get_max_length(df["tokens"])
	padded_docs = pad_sequences(df["tokens"], maxlen=max_length, padding='post')
	return padded_docs

def inference_tokenize(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokenized_text = tokenizer.tokenize(text)
	return tokenized_text

def keras_tokenizer(df):
	t = Tokenizer()
	t.fit_on_texts(df)

	vocab_size = len(t.word_index) + 1
	encoded_docs = t.texts_to_sequences(tweet_doc)

	max_length = get_max_length(encoded_docs)
	padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
	return padded_docs, t

def get_max_length(docs)
	sentence_lengths = []
	for doc in docs:
	    sentence_lengths.append(len(doc))
	return max(sentence_lengths)
