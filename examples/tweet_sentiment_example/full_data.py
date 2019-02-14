import requests
import json

import models
import evaluate

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import os

DATA_PATH = 'training.1600000.processed.noemoticon.csv'


def load_data(path):
	dataset = pd.read_csv(path, header=None, encoding="ISO-8859-1")
	renumbered_labels =[]
	for label in dataset[0]:
		if label == 4:
			renumbered_labels.append(1)
		else:
			renumbered_labels.append(0)

	dataset[0] = renumber_labels
	return dataset

def main():
	dots = "." * 6
	print('loading dataset{}'.format(dots))
	dataset = load_data(DATA_PATH)
	labels = dataset[0]

	train_text, test_text, train_labels, test_labels = train_test_split(dataset[5],
	labels,
	stratify=labels,
	test_size=0.2,
	random_state=40)

	print('generating embeddings{}'.format(dots))
	print('calling vectorizer api{}'.format(dots))
	matrix_embedding = np.zeros((len(dataset), 300))
	for i in range(len(dataset)):
		text = dataset[5][i]
		input = {'text' : text}
		response = requests.get('http://vectorizer.host/embed', data=input)
		vector_embedding = json.loads(response.text)
		vector_embedding = np.mean(vector_embedding, axis=0)
		matrix_embedding[i] = vector_embedding

	embedding_size = matrix_embedding.shape[1]

	# specifying exact numbers now, need to convert to variables
	print('initializing model{}'.format(dots))
	model = models.keras_model(embedding_size)

	categorical_labels = to_categorical(labels, num_classes=2)

	print('train test split')
	X_train, X_test, y_train, y_test = train_test_split(matrix_embedding,
	categorical_labels,
	stratify=labels,
	test_size=0.2,
	random_state=40)

	# fit model
	print('fitting model{}'.format(dots))
	model.fit(X_train, y_train, epochs=5, verbose=0)

	# evaluate model
	print('generating predictions{}'.format(dots))
	y_predicted = model.predict_classes(X_test)

	accuracy, precision, recall, f1 = evaluate.get_metrics(test_labels, y_predicted)

	with open("metrics_output.txt", "w") as text_file:
		print('accuracy: {} precision: {} recall: {} f1: {}'
			.format(accuracy, precision, recall, f1), file=text_file)

	print('plotting confusion matrix{}'.format(dots))
	cm = confusion_matrix(test_labels, y_predicted)
	print('generated confusion matrix')
	print(cm)
	fig = plt.figure(figsize=(10, 10))
	plot = evaluate.plot_confusion_matrix(cm, classes=['Negative','Positive'], normalize=False, title='Confusion matrix')
	plt.savefig('tweet_sentiment_class_confusion_matrix_full_data.png')
	plt.close()

	return


if __name__ == '__main__':
	main()
