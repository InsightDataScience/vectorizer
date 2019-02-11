from nltk.tokenize import RegexpTokenizer

def tokenize(text):
	tokenizer = RegexpTokenizer(r'\w+')
	tokenized_text = tokenizer.tokenize(text)
	return tokenized_text
