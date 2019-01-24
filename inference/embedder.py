from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv(data):
	count_vectorizer = CountVectorizer()
	corpus_data = data['text'].tolist()
	embedding = count_vectorizer.fit_transform(corpus_data)

	return embedding
