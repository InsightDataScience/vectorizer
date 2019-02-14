#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import click
import logging
import data
import preprocess
import utilities
from ngram_train import NgramTrain
import pandas as pd
from time import time
import evaluation_statistics
from ngram_test import NgramTest
from sklearn.model_selection import train_test_split
import gensim.downloader as api
from gensim.models import KeyedVectors

__author__ = "Pujaa Rajan"
__email__ = "pujaa.rajan@gmail.com"

@click.command()
@click.option('--output_file_path', is_flag=False, required=False, help = "output")
@click.option('--input_file_path', is_flag=False, required=False, help = "Input")
@click.option('--create_train_test_data', is_flag=False, required=False, help = "Training data")
@click.option('--training_data_file_path', is_flag=False, required=False, help = "Training data")
@click.option('--testing_data_file_path', is_flag=False, required=False, help = "Testing data")
@click.option('--cli', is_flag=True, required = False, help="Will run code using cli.")
@click.option('--run_ngram_train', is_flag=True, required=False, help='Will run train and test')
@click.option('--run_ngram_test', is_flag=True, required=False, help='Will run train and test')
def main(output_file_path, input_file_path, create_train_test_data, training_data_file_path, testing_data_file_path, run_ngram_train, run_ngram_test,  cli):
    """ Main function
    """

    utilities.logger()
    log = logging.getLogger('Enron_email_analysis.main')
    log.info('Starting to run main.py.')
    start = time()

    if create_train_test_data:
        log.info('Starting to create training and testing data')
        email_data = data.read_data(create_train_test_data) # Read in data from csv or s3 using file path
        email_content = email_data['content']
        training_emails, testing_emails = train_test_split(email_content, test_size=.2, train_size=.8, shuffle=True)
        log.info('Split data sent into training and testing')
        training_emails.to_csv(f'{input_file_path}/training_email_data.csv', header=['email_text'])
        log.info(f'Successfully created training emails csv: {input_file_path}/training_email_data.csv')
        testing_emails.to_csv(f'{input_file_path}/testing_email_data.csv', header=['email_text'])
        log.info(f'Successfully created testing emails csv: {input_file_path}/testing_email_data.csv')
        data.create_test_data(testing_emails, input_file_path)
        log.info(f'Successfully created fill in the blank test csv: {input_file_path}')

    if run_ngram_train:
        log.info("Starting to train ngram model")
        training_emails = pd.read_csv(training_data_file_path)
        preprocessed_training_emails = preprocess.PreprocessText(training_emails['email_text']).preprocessed_text
        NgramTrain(preprocessed_training_emails, input_file_path) # Done creating training data and
        log.info("Successfully finished training ngram model")

    if run_ngram_test:
        log.info("Starting to test ngram model")
        test_fill_in_the_blank = pd.read_csv(testing_data_file_path)
        ngram_test = NgramTest(test_fill_in_the_blank, output_file_path)
        evaluation_statistics.Evaluation(ngram_test)
        log.info("Successfully finished testing ngram model")

    if cli:

        log.info("Welcome to the Personalized Thesaurus.")
        log.info("ABOUT: This thesaurus recommends you the best word based on your previous emails and the"
                 "\nmost similar word.")
        log.info("Starting to reading in forward and backward probability pickle files")
        bigram_forward_probability = data.read_pickle_file(f'model_input_data/bigram_forward_probability.pkl')
        log.info("Successfully finished reading in 1/4 pickle files.")
        bigram_backward_probability = data.read_pickle_file(f'model_input_data/bigram_backward_probability.pkl')
        log.info("Successfully finished reading in 2/4 pickle files.")

        trigram_forward_probability = data.read_pickle_file(f'model_input_data/trigram_forward_probability.pkl')
        log.info("Successfully finished reading in 3/4 pickle files.")
        trigram_backward_probability = data.read_pickle_file(f'model_input_data/trigram_backward_probability.pkl')
        log.info("Successfully finished reading in 4/4 pickle files.")

        word_vectors = api.load("glove-wiki-gigaword-100")

        while True:
            log.info('Ready for user input')
            before_blank_tokens, after_blank_tokens, word_to_replace = utilities.take_input('cli')
            log.info(f'Before the word to replace: {before_blank_tokens}')
            log.info(f'After the word to replace: {after_blank_tokens}')
            after_predictions = data.predict_next_word(before_blank_tokens, bigram_forward_probability, trigram_forward_probability, 'forward')
            before_predictions = data.predict_next_word(after_blank_tokens, bigram_backward_probability, trigram_backward_probability, 'backward')
            merged_predictions = after_predictions+before_predictions
            word_embedding_output = data.get_similar_words(word_to_replace, word_vectors)
            print(f'Personalized Output:')
            for probability, word in merged_predictions:
                print(word + '\t' + str(probability))
            print(f'Similar Words:')
            for word, probability  in word_embedding_output:
                print(word + '\t' + str(probability))

    end = time()
    time_difference = end - start
    summary_statistics = open(f'summary_statistics.txt', 'a')
    summary_statistics.write(f'The program ran for: {time_difference}\n')
    summary_statistics.close()


if __name__ == "__main__":
    main()
