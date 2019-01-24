import click
import sys

import logging

import numpy as np
import pandas as pd

import clean
# import preprocess

DATA_PATH = 'data/raw/testdata.manual.2009.06.14.csv'

def load_data(path):
	dataset = pd.read_csv(path, header=None)
	dataset.columns = ['text']
	return dataset

@click.command()
@click.option("--clean_data", is_flag=True)
@click.option('--preprocess_data', is_flag=True)
def main(clean_data, preprocess_data):
	logging.getLogger().setLevel(logging.INFO)
	# logging.INFO('Cleaning input data')
	if clean_data:
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
