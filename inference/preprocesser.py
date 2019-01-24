from nltk.tokenize import RegexpTokenizer

def tokenize(df):
	tokenizer = RegexpTokenizer(r'\w+')
	df["tokens"] = df["text"].apply(tokenizer.tokenize)
	return df
