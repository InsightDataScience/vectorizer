import click
import sys

import logging

import numpy as np
import pandas as pd

import clean
# import preprocess

DATA_PATH = 'data/raw/testdata.text_only.csv'

def load_data(path):
	dataset = pd.read_csv(path, index_col=0)
	dataset.columns = ['text']
	return dataset

@click.command()
@click.option("--clean_data", is_flag=True)
@click.option('--preprocess_data', is_flag=True)
def main(clean_data, preprocess_data):
	# logging.getLogger().setLevel(logging.INFO) # TODO: setup logging

	if clean_data:
		# logging.INFO('Cleaning input data')
		dataset = load_data(DATA_PATH)
		print('before cleaning \n')
		print(dataset.head(2))
		dataset = clean.remove_URL(dataset, 'text')
		dataset = clean.remove_special_characters(dataset, 'text')
		dataset = clean.lowercase(dataset, 'text')
		print('after cleaning \n')
		print(dataset.head(2))

	return


if __name__ == '__main__':
	main()
