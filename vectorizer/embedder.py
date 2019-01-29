from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy

def cv(df):
	count_vectorizer = CountVectorizer()
	embedding = count_vectorizer.fit_transform(df)

	return embedding, count_vectorizer

def glove_embedding(df, tokenizer):
	embedding_matrix = np.zeros((vocab_size, 300))
	nlp = spacy.load('en_vectors_web_lg')
	for word, i in tokenizer.word_index.items():
    embedding_vector = nlp.vocab.get_vector(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
