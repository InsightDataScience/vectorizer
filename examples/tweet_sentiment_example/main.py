import requests
import json

import models
import evaluate

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
import os

# MAC OS only issue see link (stackoverflow) for additional info: https://bit.ly/2S85FrV
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# current sample data setup
DATA_PATH = '../../data/testdata.text_only.csv'
LABEL_PATH = '../../data/renumbered_test_labels.csv'

def load_data(path):
	# current implementation assumes data input as single column text only csv
	dataset = pd.read_csv(path, index_col=0)
	dataset.columns = ['text']
	return dataset

def load_labels(path):
	labels = pd.read_csv(path, index_col=0)
	labels.columns = ['labels']
	return labels

def main():
	dots = "." * 6
	print('loading dataset{}'.format(dots))
	dataset = load_data(DATA_PATH)
	labels = load_labels(LABEL_PATH)

	train_text, test_text, train_labels, test_labels = train_test_split(dataset.text,
	labels.labels,
	test_size=0.2,
	random_state=40)

	print('generating embeddings{}'.format(dots))
	print('calling vectorizer api{}'.format(dots))
	matrix_embedding = np.zeros((len(dataset), 300))
	for i in range(len(dataset)):
		text = dataset['text'][i]
		input = {'text' : text}
		response = requests.get('http://0.0.0.0:5000/embed', data=input)
		vector_embedding = json.loads(response.text)
		matrix_embedding[i] = vector_embedding

	embedding_size = matrix_embedding.shape[1]

	# specifying exact numbers now, need to convert to variables
	print('initializing model{}'.format(dots))
	model = models.keras_model(embedding_size)

	categorical_labels = to_categorical(labels, num_classes=3)

	print('train test split')
	X_train, X_test, y_train, y_test = train_test_split(matrix_embedding,
	categorical_labels,
	test_size=0.2,
	random_state=40)

	# fit model
	print('fitting model{}'.format(dots))
	model.fit(X_train, y_train, epochs=5, verbose=0)

	# evaluate model
	print('generating predictions{}'.format(dots))
	y_predicted = model.predict_classes(X_test)
	y_string_predict = []
	for prediction in y_predicted:
		if prediction == 0:
			y_string_predict.append('Negative')
		if prediction == 1:
			y_string_predict.append('Neutral')
		if prediction == 2:
			y_string_predict.append('Positive')

	accuracy, precision, recall, f1 = evaluate.get_metrics(test_labels, y_predicted)
	print('accuracy: {} precision: {} recall: {} f1: {}'
	.format(accuracy, precision, recall, f1))

	print('plotting confusion matrix{}'.format(dots))
	cm = confusion_matrix(test_labels, y_predicted)
	print('generated confusion matrix')
	print(cm)
	fig = plt.figure(figsize=(10, 10))
	plot = evaluate.plot_confusion_matrix(cm, classes=['Negative','Neutral','Positive'], normalize=False, title='Confusion matrix')
	plt.savefig('tweet_sentiment_class_confusion_matrix.png')
	plt.close()

	print('generating output dataframe{}'.format(dots))
	output_df = pd.DataFrame(test_text)
	output_df.columns = ['Test Tweets']
	output_df['Target Sentiment'] = test_labels
	output_df['Predicted Sentiment'] = y_string_predict
	output_df.to_csv('tweet_sentiment_prediction_table.csv')

	return


if __name__ == '__main__':
	main()
