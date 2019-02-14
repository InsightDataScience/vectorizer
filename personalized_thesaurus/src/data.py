#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle
import re
from io import BytesIO
from random import randint

import boto3
import pandas as pd
from nltk import word_tokenize, tokenize
from sklearn.model_selection import train_test_split

__author__ = "Pujaa Rajan"
__email__ = "pujaa.rajan@gmail.com"

log = logging.getLogger('Enron_email_analysis.data')


def read_data(input_filepath):
    """ Reads in data from csv file or s3 bucket. """
    logging.info(f'Starting to read in raw data from {input_filepath} (may be CSV or s3)')
    try:
        dataframe = pd.read_csv(input_filepath)
        logging.info('Successfully finished reading raw data (may be CSV or s3)')
        logging.info(f'Read in dataframe of shape: {dataframe.shape}\n')
        logging.info(f'Sample of dataframe: {dataframe.head()}\n')
        return dataframe
    except Exception as e:
        logging.error(f'Failed to read in raw data from {input_filepath} (may be CSV or s3). {e}')
        exit()


def train_test_data(dataframe):
    """ Splits data into train (80%) and test sets (20%). """
    try:
        training_emails, testing_emails = train_test_split(dataframe, test_size=.2, train_size=.8, shuffle=True)
        return training_emails, testing_emails
    except Exception as e:
        logging.error(f'Failed to split data into train and test. {e}')
        exit()


def write_pickle_file(what_to_pickle, input_filepath, output_filename, s3):
    """ Writes input data to a pickle file to local computer or to S3. """
    if s3:
        bucket = 'pujaa-rajan-enron-email-data'
        key = f'model_input_data/{output_filename}'
        pickle_byte_obj = pickle.dumps(what_to_pickle)
        s3_resource = boto3.resource('s3')
        s3_resource.Object(bucket, key).put(Body=pickle_byte_obj)
    else:
        with open(f'{input_filepath}/{output_filename}', 'wb') as picklefile:
            pickle.dump(what_to_pickle, picklefile)


def read_pickle_file(path):
    """ Reads pickle file from S3. """
    try:
        s3 = boto3.resource('s3')
        bucket = 'pujaa-rajan-enron-email-data'
        with BytesIO() as data:
            s3.Bucket(bucket).download_fileobj(path, data)
            data.seek(0)
            pickled_data = pickle.load(data)
        return pickled_data
    except Exception:
        raise Exception('Failed to read in pickle file.')
        exit()


def create_blanks(sentence_row, answer_column):
    """ Removes words and creates blanks to output test sentences for the language model to predict the word in the
    blank. """
    remove_punctuation = sentence_row[:-1]
    word_index = randint(3, len(word_tokenize(remove_punctuation)) - 4)
    words_in_sentence = word_tokenize(remove_punctuation)
    answer_column.append(words_in_sentence[word_index])  # Saving the answer before replacing it with a blank
    words_in_sentence[word_index] = '_'  # Replacing with a blank
    full_sentence = " ".join(words_in_sentence)
    return full_sentence


def filter_sentence(sentence_row):
    """Filter for good sentences"""
    m = re.search(r'https?:|@\w+|\d', sentence_row)
    n = re.search(r'[^A-Za-z\s!?.]', sentence_row)
    if m or n or len(word_tokenize(sentence_row)) < 8:
        pass
    else:
        return sentence_row


def create_test_data(dataframe, input_file_path):
    """Create test data. Input is the test part of the train_test_data function. Output is csv with input and labels."""
    log.info("Starting to create test data")
    dataframe_no_nan = dataframe.dropna()
    email_dataframe = dataframe_no_nan.apply(lambda row: tokenize.sent_tokenize(row))
    sentence_dataframe = email_dataframe.apply(lambda row: pd.Series(row)).stack().reset_index(level=1, drop=True)
    filtered_sentences = sentence_dataframe.apply(lambda row: filter_sentence(row)).dropna()
    answer_column = []
    blank_sentences = filtered_sentences.apply(lambda row: create_blanks(row, answer_column))
    test_dataframe = pd.DataFrame({'fill in the blank': blank_sentences.values, 'answer': answer_column})
    test_dataframe.to_csv(f'{input_file_path}/testing_email_data.csv')
    logging.info("End of creating test data")


def merge_predictions(forward_predictions, backward_predictions):
    """ Merge the predicted words from the forward and backward ngram model. """
    merged_predictions = forward_predictions + backward_predictions
    print(merged_predictions)
    return merged_predictions


def predict_next_word(words_before_or_after_blank, bigram_probability, trigram_probability, direction):
    """Predict the next word using the ngram model. """
    if direction == 'forward':
        logging.info('Predicting the next word using a FORWARD n gram model.')
        look_up_unigram = words_before_or_after_blank[-1]  # Start with word right before blank
        look_up_bigram = words_before_or_after_blank[-2:]  # Start with 2 words right before blank
    elif direction == 'backward':
        logging.info('Predicting the next word using a BACKWARD n gram model.')
        look_up_unigram = words_before_or_after_blank[0]  # Start with first word after blank
        look_up_bigram = words_before_or_after_blank[:2]  # Start with first two words after blank

    else:
        raise RuntimeError('Specify direction as forward or backward for ngram_probability function.')
    if look_up_unigram in bigram_probability:
        bigram_answers = bigram_probability[look_up_unigram]
        if tuple(look_up_bigram) in trigram_probability:
            trigram_answers = trigram_probability[tuple(look_up_bigram)]
            if bigram_answers and trigram_answers:
                answers = bigram_answers + trigram_answers
                sorted_probabilities = sorted(answers, key=lambda x: x[0], reverse=True)
                all_answers = sorted_probabilities[0:10]
                logging.info(f'Final word probabilities for {direction} direction:{all_answers[:3]}')
                return all_answers
        else:
            sorted_probabilities = sorted(bigram_answers, key=lambda x: x[0], reverse=True)
            return sorted_probabilities[0:10]
    else:
        return None  # if the look up word is not in probability matrix


def get_similar_words(word, word_vectors):
    """ Get similar words using Gensim's word2vec embedding. """
    try:
        similar_words = word_vectors.similar_by_word(word, topn=10)
        return similar_words
    except KeyError:
        return None


def get_contextual_words_genism(words, word_vectors):
    return word_vectors.predict_output_word(words,  topn=10)

