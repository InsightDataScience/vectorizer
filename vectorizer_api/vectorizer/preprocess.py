from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

NLTK_STOPWORDS = set(stopwords.words('english'))

def tokenize(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokenized_text = tokenizer.tokenize(text)
	return tokenized_text

def remove_stop_words(tokens):
	no_stopwords = []
	for token in tokens:
		if token not in NLTK_STOPWORDS:
			no_stopwords.append(token)
	return no_stopwords

def lemmatize_words(tokens):
	lmtzr = WordNetLemmatizer()
	lemmatized = [lmtzer.lemmatize(token) for token in tokens]
	return lemmatized
