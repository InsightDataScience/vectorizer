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


def read_data(file_path):
    """Reads in data from csv file or s3 bucket.

    # TODO: Add descriptive statistics about emails and create some graphs that you can save.
    # TODO: Add reading in guard rails and properties. More specific exception?
    """
    logging.info(f'Starting to read in raw CSV from {file_path}')
    try:
        dataframe = pd.read_csv(file_path)
        logging.info('Successfully finished reading raw CSV')
        logging.info(f'Read in dataframe of shape: {dataframe.shape}\n')
        logging.info(f'Sample of dataframe: {dataframe.head()}\n')
        return dataframe
    except Exception:
        logging.error(f'Failed to read in raw CSV from {input_filepath}')


def train_test_data(dataframe):
    training_emails, testing_emails = train_test_split(dataframe, test_size=.2, train_size=.8, shuffle=True)
    return training_emails, testing_emails


def write_to_csv(test_data):
    with open('test_fill_in_the_blank.csv', 'w') as out:
        csv_out = csv.writer(out, delimiter=',')
        csv_out.writerow(['fill in the blank', 'answer'])
        for row in test_data:
            csv_out.writerow(row)



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
