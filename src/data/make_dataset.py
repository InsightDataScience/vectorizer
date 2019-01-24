# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import spacy
from collections import defaultdict
from nltk.util import ngrams
from collections import Counter




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    #variable declaration
    vocab_dict = defaultdict(int)          #for storing the different words with their frequencies
    bi_dict = Counter()   # for keeping count of sentences of two words
    bi_prob_dict = defaultdict(list)           #for storing the probable  words for Bigram sentences

    nlp = spacy.load('en_core_web_sm')

    nlp.add_pipe(preprocess_text, name='preprocesser', after='tagger')
    print(nlp.pipe_names)  # ['tagger', 'preprocesser','parser', 'ner']

    token_len = loadCorpus(input_filepath, bi_dict, vocab_dict, nlp)

    param = [0.7,0.1,0.1,0.1]
    #create bigram Probability Dictionary
    findBigramProbAdd1(vocab_dict, bi_dict, bi_prob_dict)

    # sort the probability dictionaries
    sortProbWordDict(bi_prob_dict)

    #take user input
    input_sen = takeInput()

    ### PREDICTION
    #choose most probable words for prediction
    word_choice = chooseWords(input_sen, bi_prob_dict)
    prediction = doInterpolatedPredictionAdd1(input_sen, bi_dict, vocab_dict,token_len, word_choice, param)
    print('Word Prediction:',prediction)


def preprocess_text(doc):
    """ Preprocess text. Tokenize, lowercase, and remove punctuation and stopwords and new lines """
    nlp = spacy.load('en_core_web_sm') # ? NOT SURE IF I SHOULD BE REWRITING THIS HERE
    # ? REMOVE DATES TIMES AND EMAILS
    # CCONSIDER PENN TREE BANK AND WIKI DATASET
    stopwords = spacy.lang.en.STOP_WORDS
    # Tokenize, remove punctuation, symbols (#), stopwords
    doc = [tok.text for tok in doc if (tok.text not in stopwords and tok.pos_ != "PUNCT" and tok.pos_ != "SYM" and tok.pos_ != "X" and tok.text != "\n" and tok.text.strip() != '')]
    # Lowercase tokens
    doc = [tok.lower() for tok in doc]
    doc = ' '.join(doc)
    return nlp.make_doc(doc)

def loadCorpus(file_path, bi_dict, vocab_dict, nlp):
    w1 = ''  # for storing the 3rd last word to be used for next token set
    w2 = ''  # for storing the 2nd last word to be used for next token set
    w3 = ''  # for storing the last word to be used for next token set
    token = []
    # total no. of words in the corpus
    word_len = 0

    # open the corpus file and read it line by line
    # read in better
    spacy_tokenized_emails = []
    corpus_size = 0
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip() # checking to see if line is blank
            if not line:  # line is blank
                continue
            line_tokens = []
            doc = nlp(line)
            for token in doc:
                line_tokens.append(token.text)
            spacy_tokenized_emails.append(line_tokens)
            # ? missing relationships of words across lines
            corpus_size = corpus_size + len(line_tokens)

            add_words_to_vocab(line_tokens, vocab_dict)

            bi_dict.update(ngrams(line_tokens, 2))
    return len(vocab_dict)

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
