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
import evaluation_statistics
from ngram_test import NgramTest


@click.command()
@click.option('--output_file_path', is_flag=False, required=False, help = "output")
@click.option('--training_data_file_path', is_flag=False, required=False, help = "Training data")
@click.option('--testing_data_file_path', is_flag=False, required=False, help = "Testing data")
@click.option('--cli', is_flag=True, required = False, help="Will run code using cli.")
@click.option('--run_ngram_train', is_flag=True, required=False, help='Will run train and test')
@click.option('--run_ngram_test', is_flag=True, required=False, help='Will run train and test')
def main(output_file_path, training_data_file_path, testing_data_file_path, run_ngram_train, run_ngram_test,  cli):
    """ Main function
    """
    utilities.logger()
    log = logging.getLogger('Enron_email_analysis.main')
    log.info('Starting to run main.py.')
    start = time()

    if run_ngram_train:
        logging.info("Starting to train ngram model")
        training_emails = pd.read_csv(training_data_file_path)
        preprocessed_training_emails = preprocess.PreprocessText(training_emails['email_text']).preprocessed_text
        NgramTrain(preprocessed_training_emails, output_file_path)
        logging.info("Successfully finished training ngram model")

    if run_ngram_test:
        logging.info("Starting to test ngram model")
        testing_emails = pd.read_csv(testing_data_file_path)
        ngram_test = NgramTest(testing_emails, output_file_path) # CHANGE INPUT TO USE PICKLED FILES
        evaluation_statistics.Evaluation(ngram_test, output_file_path)
        logging.info("Successfully finished testing ngram model")

    # TODO Add loop so users can answer questions multiple times
    if cli:
        preprocessed_data = preprocess.PreprocessText(email_content).preprocessed_text
        ngram_data = ngram.Ngram(preprocessed_data)
        #take user input
        before_blank_tokens, after_blank_tokens = utilities.take_input('cli')
        log.info(f'Before blank words: {before_blank_tokens}')
        log.info(f'After blank words: {after_blank_tokens}')

        ### PREDICTION
        predict_next_word(before_blank_tokens, ngram_data.bigram_forward_probability, 'forward')
        predict_next_word(before_blank_tokens, ngram_data.bigram_backward_probability, 'backward')
        #choose most probable words for prediction

    end = time()
    time_difference = end - start
    summary_statistics = open(f'{output_file_path}/summary_statistics.txt', 'a')
    summary_statistics.write(f'The program ran for: {time_difference}\n')
    summary_statistics.close()

    # if create_train_test_data:
    #email_data = data.read_data(input_file_path) # Read in data from csv or s3 using file path
    #     training_emails, testing_emails = data.train_test_data(email_content)
    #     training_emails.to_csv(f'/Users/pujaarajan/Documents/GitHub/vectorizer/nlp_fib/input/training_email_data.csv')
    #     testing_emails.to_csv(f'/Users/pujaarajan/Documents/GitHub/vectorizer/nlp_fib/input/testing_email_data.csv')
    # self.create_test_data(testing_emails)


if __name__ == "__main__":
    logging.info('Going to run main function.')
    main()
