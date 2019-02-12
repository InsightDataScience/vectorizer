import sys
import numpy as np
import spacy

embedding_models_path = './vectorizer/word2vec_twitter_model'
sys.path.append(embedding_models_path)

import word2vecReader

# TWITTER_WORD2VEC_EMBEDDING_PATH = "./vectorizer/word2vec_twitter_model/" + \
# 	"word2vec_twitter_model.bin"
# TWITTER_WORD2VEC_EMBEDDING = word2vecReader.Word2Vec.\
#	load_word2vec_format(TWITTER_WORD2VEC_EMBEDDING_PATH,
#	binary=True)
# get vector dimension from sample word 'hello'
# HELLO_INDEX = TWITTER_WORD2VEC_EMBEDDING.vocab['hello'].index
# VECTOR_DIMENSION = len(TWITTER_WORD2VEC_EMBEDDING.syn0[HELLO_INDEX])
GLOVE_EMBEDDING = spacy.load('en_vectors_web_lg')
VECTOR_DIMENSION = len(GLOVE_EMBEDDING.vocab.get_vector('hello'))

def inference_glove_embedding(preprocessed_text, averaged_embedding=True):
	embedding_matrix = np.zeros((len(preprocessed_text), VECTOR_DIMENSION))
	for i in range(len(preprocessed_text)):
		word = preprocessed_text[i]
		embedding_vector = GLOVE_EMBEDDING.vocab.get_vector(word)
		# try:
		#	word_index = TWITTER_WORD2VEC_EMBEDDING.vocab[word].index
		#	embedding_vector = TWITTER_WORD2VEC_EMBEDDING.syn0[word_index]
		# except KeyError:
		#	print('word not in embedding')
		#	embedding_vector = None
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector

	# using float to ensure accuracy
	# if averaged_embedding:
	embedding_matrix = np.mean(embedding_matrix, axis=0)

	return embedding_matrix
