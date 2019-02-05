# -*- coding: utf-8 -*-
import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import tokenize
import re
from nltk.tokenize import word_tokenize
from random import randint
import preprocess
import csv
import pickle
import boto3
from io import BytesIO


log = logging.getLogger('Enron_email_analysis.data')


def read_data(input_filepath):
    """Reads in data from csv file or s3 bucket.

    # TODO: Add descriptive statistics about emails and create some graphs that you can save.
    # TODO: Add reading in guard rails and properties. More specific exception?
    """
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
    training_emails, testing_emails = train_test_split(dataframe, test_size=.2, train_size=.8, shuffle=True)
    return training_emails, testing_emails


def write_pickle_file(what_to_pickle, input_filepath, output_filename, s3):
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
    s3 = boto3.resource('s3')
    bucket = 'pujaa-rajan-enron-email-data'
    with BytesIO() as data:
        s3.Bucket(bucket).download_fileobj(path, data)
        data.seek(0)  # move back to the beginning after writing
        pickled_data = pickle.load(data)
    return pickled_data


def create_blanks(sentence_row, answer_column):
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


def doInterpolatedPredictionAdd1(sen, bi_dict, vocab_dict, token_len, word_choice, param):
    # TODO How to pick just one word?
    pred = ''
    max_prob = 0.0
    V = len(vocab_dict)
    # for each word choice find the interpolated probability and decide
    for word in word_choice:
        key = sen + ' ' + word[1]
        quad_token = key.split()

        prob = (param[2] * ((bi_dict[' '.join(quad_token[2:4])] + 1) / (vocab_dict[quad_token[2]] + V))
                + param[3] * ((vocab_dict[quad_token[3]] + 1) / (token_len + V))
                )

        if prob > max_prob:
            max_prob = prob
            pred = word
    # return only pred to get word with its prob
    if pred:
        return pred[1]
    else:
        return ''
