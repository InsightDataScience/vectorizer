import click
import requests
import json
import logging

import models

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

# current sample data setup
DATA_PATH = '../../data/testdata.text_only.csv'
LABEL_PATH = '../../data/renumbered_test_labels.csv'

def load_data(path):
	dataset = pd.read_csv(path, index_col=0)
	dataset.columns = ['text']
	return dataset

def load_labels(path):
	labels = pd.read_csv(path, index_col=0)
	labels.columns = ['labels']
	return labels

def main():
	dataset = load_data(DATA_PATH)
	labels = load_labels(LABEL_PATH)

	# call vectorizer_api to generate embedding_matrix
	matrix_embedding = np.zeros((len(dataset), 300))
	for i in range(len(dataset)):
		text = dataset['text'][i]
		input = {'text' : text}
		response = requests.get('http://0.0.0.0:5000/infer', data=input)
		vector_embedding = json.loads(response.text)
		matrix_embedding[i] = vector_embedding

	embedding_size = matrix_embedding.shape[1]

	# initialize keras model
	model = models.keras_model(embedding_size)

	# reformat labels for keras model
	categorical_labels = to_categorical(labels, num_classes=3)

	# 80/20 train test split
	X_train, X_test, y_train, y_test = train_test_split(matrix_embedding,
	categorical_labels,
	test_size=0.2,
	random_state=40)

	# fit model
	model.fit(X_train, y_train, epochs=5, verbose=0)

	# evaluate model
	loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
	print('Accuracy: %f' % (accuracy*100))

	return


if __name__ == '__main__':
	main()
