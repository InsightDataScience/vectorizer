# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import spacy
from collections import defaultdict
from nltk.util import ngrams
from collections import Counter
import corpus
import pandas as pd


def read_csv(input_filepath):
    """Reads in CVS.

    TODO: Add reading in guard rails and properties. More specific exception?
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

def findBigramProbAdd1(vocab_dict, bi_dict, bi_prob_dict):

    V = len(vocab_dict)

    # create a dictionary of probable words with their probabilities for bigram probabilites
    for bi in bi_dict:
        # unigram for key
        unigram = bi[0]

        # find the probability
        # add 1 smoothing has been used
        prob = (bi_dict[bi] + 1) / (vocab_dict[unigram] + V)

        # bi_prob_dict is a dict of list
        # if the unigram sentence is not present in the Dictionary then add it
        if unigram not in bi_prob_dict:
            bi_prob_dict[unigram] = []
            bi_prob_dict[unigram].append([prob, bi[-1]])
        # the unigram sentence is present but the probable word is missing,then add it
        else:
            bi_prob_dict[unigram].append([prob, bi[-1]])

    prob = None
    bi_token = None
    unigram = None

def sortProbWordDict(bi_prob_dict):
    #sort bigram dict
    for key in bi_prob_dict:
        if len(bi_prob_dict[key])>1:
            bi_prob_dict[key] = sorted(bi_prob_dict[key],reverse = True)


def add_words_to_vocab(doc, vocab_dict):
    # add new unique words in doc to the vocaulary set if available
    for word in doc:
        if word not in vocab_dict:
            vocab_dict[word] = 1
        else:
            vocab_dict[word] += 1


def chooseWords(sen, bi_prob_dict):
    word_choice = []
    token = sen.split()
    if token[-1] in bi_prob_dict:
        word_choice += bi_prob_dict[token[-1]][:1]
        # print('Word Choice bi dict')

    return word_choice

def takeInput():
    cond = False
    #take input
    while(cond == False):
        sen = input('Enter the string\n')
        # ? We may need to remove puncuation from the input
        temp = sen.split()
        if len(temp) < 3:
            print("Please enter atleast 3 words !")
        else:
            cond = True
            temp = temp[-3:]
    sen = " ".join(temp)
    return sen


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
