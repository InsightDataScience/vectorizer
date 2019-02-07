import sys
import numpy as np
import spacy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

embedding_models_path = './vectorizer/word2vec_twitter_model'
sys.path.append(embedding_models_path)

import word2vecReader

TWITTER_WORD2VEC_EMBEDDING_PATH = "./vectorizer/word2vec_twitter_model/" + \
	"word2vec_twitter_model.bin"
TWITTER_WORD2VEC_EMBEDDING = word2vecReader.Word2Vec.\
	load_word2vec_format(TWITTER_WORD2VEC_EMBEDDING_PATH,
	binary=True)
# GLOVE_EMBEDDING = spacy.load('en_vectors_web_lg')

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
		# embedding_vector = GLOVE_EMBEDDING.vocab.get_vector(word)
		embedding_vector = TWITTER_WORD2VEC_EMBEDDING.vocab(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# using float to ensure accuracy
	averaged_embedding = np.mean(embedding_matrix, axis=0, dtype=np.float64)
	return averaged_embedding
