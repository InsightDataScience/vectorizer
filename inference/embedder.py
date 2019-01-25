from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv(data):
	count_vectorizer = CountVectorizer()
	embedding = count_vectorizer.fit_transform(data)

	return embedding
