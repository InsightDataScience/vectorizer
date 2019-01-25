import click
import sys

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import cleaner
import preprocesser
import embedder

# current sample data setup
# should be made generic so users can specify file path
DATA_PATH = 'data/raw/testdata.text_only.csv'
LABEL_PATH = 'data/raw/test_labels.csv'

def load_data(path):
	# current implementation assumes data input as single column text only csv
	dataset = pd.read_csv(path, index_col=0)
	dataset.columns = ['text']
	return dataset

def load_labels(path):
	labels = pd.read_csv(path, indx_col=0)
	labels.columns = ['labels']
	return labels

@click.command()
@click.option("--clean_data", is_flag=True)
@click.option('--preprocess_data', is_flag=True)
@click.option('--embed_data', is_flag=True)
@click.option('--evaluate', is_flag=True)
def main(clean_data, preprocess_data, embed_data, evaluate):
	# logging.getLogger().setLevel(logging.INFO) # TODO: setup logging

	if clean_data:
		# logging.INFO('Cleaning input data')
		dataset = load_data(DATA_PATH)
		print('before cleaning \n')
		print(dataset.head(2))
		dataset = cleaner.remove_URL(dataset, 'text')
		dataset = cleaner.remove_special_characters(dataset, 'text')
		dataset = cleaner.lowercase(dataset, 'text')
		print('after cleaning \n')
		print(dataset.head(2))

	if preprocess_data:
		dataset = preprocesser.tokenize(dataset)
		# df is now two column ['text', 'tokens']
		print(dataset.head(1))

	if evaluate:
		list_corpus = dataset['text'].tolist()
		list_labels = labels['labels'].tolist()

		X_train, X_test, y_train, y_test = train_test_split(list_corpus,
		list_labels,
		test_size=0.2,
		random_state=40)

		X_train_emb = embedder.cv(X_train)
		X_test_emb = embedder.cv(X_test)

	else:
		if embed_data:
			embedding = embedder.cv(dataset)
			print(embedding)


	return


if __name__ == '__main__':
	main()
