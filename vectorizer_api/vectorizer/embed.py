from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import spacy

# GLOVE_EMBEDDING = spacy.load('en_vectors_web_lg')

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

	return embedding_matrix

def inference_glove_embedding(preprocessed_text):
	embedding_matrix = np.zeros((len(preprocessed_text), 300))
	for i in range(len(preprocessed_text)):
		word = preprocessed_text[i]
		embedding_vector = GLOVE_EMBEDDING.vocab.get_vector(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# using float to ensure accuracy
	averaged_embedding = np.mean(embedding_matrix, axis=0, dtype=np.float64)
	return averaged_embedding
