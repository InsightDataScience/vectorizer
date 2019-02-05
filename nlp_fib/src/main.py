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
from sklearn.model_selection import train_test_split


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
        evaluation_statistics.Evaluation(ngram_test, output_file_path)
        log.info("Successfully finished testing ngram model")

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
    summary_statistics = open(f'summary_statistics.txt', 'a')
    summary_statistics.write(f'The program ran for: {time_difference}\n')
    summary_statistics.close()


if __name__ == "__main__":
    logging.info('Going to run main function.')
    main()
