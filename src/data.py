# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import boto3
from sklearn.model_selection import train_test_split

boto3.set_stream_logger('boto3', logging.INFO)
#client = boto3.client('s3', aws_access_key_id='AKIAISJAQSB7RJADVFHQ', aws_secret_access_key='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') # Low-level functional API
client = boto3.client('s3') # Low-level functional API


resource = boto3.resource('s3') # High-level object-oriented API
bucket = resource.Bucket('pujaa-rajan-enron-email-data') # Subsitute this for your s3 bucket name.

def get_data_from_s3():
    s3_file_data = client.get_object(Bucket='pujaa-rajan-enron-email-data', Key='enron_emails.csv')
    print("connecting to s3 ", s3_file_data)
    bucket = 'pujaa-rajan-enron-email-data'  # Or whatever you called your bucket
    data_key = 'enron_emails.csv'  # Where the file is within your bucket
    data_location = 's3://{}/{}'.format(bucket, data_key)
    print("reading in data")
    df = pd.read_csv(data_location)
    print(df.head())

    return df.head()

def read_csv(input_filepath):
    """Reads in CVS.

    TODO: Add reading in guard rails and properties. More specific exception?
    TODO: When to split into test and training? Think about it and implement.
    """
    logging.info(f'Starting to read in raw CSV from {input_filepath}')
    try:
        dataframe = pd.read_csv(input_filepath)
        logging.info('Successfully finished reading raw CSV')
        logging.info(f'Read in dataframe of shape: {dataframe.shape}\n')
        logging.info(f'Sample of dataframe: {dataframe.head()}\n')
        return dataframe
    except Exception:
        logging.error(f'Failed to read in raw CSV from {input_filepath}')

def train_test_data(dataframe):
    training_emails, testing_emails = train_test_split(dataframe, test_size=.2, train_size=.8, shuffle=True)
    return training_emails, testing_emails


def doInterpolatedPredictionAdd1(sen, bi_dict, vocab_dict, token_len, word_choice, param):
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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
