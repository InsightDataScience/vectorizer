import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import click
import logging
import data
import preprocess
import utilities
import ngram
import pandas as pd
from time import time
import evaluation_statistics
from ngram_test import NgramTest

@click.command()
@click.argument('input_file_path', type=str, required=True)
@click.option('--cli', is_flag=True, required = False, help="Will run code using cli.")
@click.option('--run_test', is_flag=True, required=False, help='Will run train and test')
def main(input_file_path, run_test,  cli):
    """ Main function
    """
    utilities.logger()
    log = logging.getLogger('Enron_email_analysis.main')
    log.info('Starting to run main.py.')
    log.info('Going to read in Enron email data.')
    email_data = data.read_data(input_file_path) # Read in data from csv or s3 using file path
    log.info(f'Read in {email_data.shape[0]} emails.')
    log.info('Extracting only email content.')
    email_content = email_data['content']
    start = time()

    if run_test:
        training_emails, testing_emails = data.train_test_data(email_content)
        preprocessed_training_emails = preprocess.PreprocessText(training_emails).preprocessed_text
        ngram_data = ngram.Ngram(preprocessed_training_emails)

        ngram_test = NgramTest(testing_emails, ngram_data)

        evaluation_statistics.Evaluation(ngram_test)
        logging.info("Done running tests")


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
    summary_statistics = open("summary_statistics.txt", "a")
    summary_statistics.write(f'The program ran for: {time_difference}\n')
    summary_statistics.close()




if __name__ == "__main__":
    logging.info('Going to run main function.')
    main()
