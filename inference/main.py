import argparse
import sys

import logging

import clean
import preprocess

def main():

	logging.getLogger().setLevel(logging.INFO)

	if FLAGS.clean:
		logging.INFO('Cleaning input data')
		clean

	if FLAGS.preprocess:
		logging.INFO('Preprocessing data')

	if FLAGS.embed:
        logging.INFO('Embedding data and generating vector')

	pass


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--clean',
      default=False,
      help='Train the model.',
      action='store_true')
  parser.add_argument(
      '--preprocess',
      default=False,
      help='Evaluate the trained model.',
      action='store_true')
  parser.add_argument(
      '--embed',
      default=False,
      help='Predict.',
      action='store_true')
  parser.add_argument(
      '--inference_model_train',
      default=False,
      help='Predict.',
      action='store_true')
  parser.add_argument(
      '--inferecnce_model_predict',
      default=False,
      help='Predict.',
      action='store_true')
  FLAGS, unparsed = parser.parse_known_args()
