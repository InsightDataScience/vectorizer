import data
import pandas as pd
import utilities
import data
from random import randint
from nltk import tokenize, word_tokenize
import logging
import re
import ngram_test


class NgramTest:
    """ This class takes in the probability files
    """
    def __init__(self, test_file_path, output_file_path):
        self.log = logging.getLogger('Enron_email_analysis.ngram_test')
        self.test_fill_in_the_blank = test_file_path
        self.bigram_backward_probability = data.read_pickle_file(f'model_input_data/bigram_backward_probability.pickle')
        self.bigram_forward_probability = data.read_pickle_file(f'model_input_data/bigram_forward_probability.pickle')
        forward_answers = []
        backward_answers = []
        self.test_fill_in_the_blank['fill in the blank'].apply(
            lambda fib: self.answer_fib(fib, forward_answers, backward_answers, self.bigram_forward_probability, self.bigram_backward_probability))
        self.test_fill_in_the_blank['Forward Answers'] = self.get_values(forward_answers, 1)
        self.test_fill_in_the_blank['Forward Answer Probability'] = self.get_values(forward_answers, 0)
        self.test_fill_in_the_blank['Backward Answers'] = self.get_values(backward_answers, 1)
        self.test_fill_in_the_blank['Backward Answer Probability'] = self.get_values(backward_answers, 0)
        self.test_fill_in_the_blank.to_csv(f'{output_file_path}/test_fib_answers.csv')

    def answer_fib(self, fib, forward_answers, backward_answers, forward_probability, backward_probability):
        before_blank_tokens, after_blank_tokens = utilities.take_input(fib)
        ### PREDICTION
        forward_answers.append(self.predict_next_word(before_blank_tokens, forward_probability, 'forward'))
        backward_answers.append(self.predict_next_word(before_blank_tokens, backward_probability, 'backward'))
        # choose most probable words for prediction
        return None

    def get_values(self, answers, n):
        output_list = []
        for sentence_answers in answers:
            output_list.append([i[n] for i in sentence_answers])
        return output_list

    def predict_next_word(self, words_before_or_after_blank, bigram_probability, direction):
        if direction == 'forward':
            logging.info('Predicting the next word using a FORWARD n gram model.')
            look_up_word = words_before_or_after_blank[-1]  # Start with word right before blank
        elif direction == 'backward':
            logging.info('Predicting the next word using a BACKWARD n gram model.')
            look_up_word = words_before_or_after_blank[0]  # Start with first word after blank
        else:
            raise RuntimeError('Specify direction as forward or backward for ngram_probability function.')
            # ? Is this the right error to raise?
        if look_up_word in bigram_probability:
            token_probabilities = bigram_probability[look_up_word]
            sorted_probabilities = sorted(token_probabilities, key=lambda x: x[0], reverse=True)
            logging.info(f'Final word probabilities for {direction} direction:{sorted_probabilities[:3]}')
            return sorted_probabilities[0:10]
        else:
            # TODO FIX THIS PART WHICH IS WHAT TO DO IF THE LOOK UP WORD IS NOT IN THE PROBABILITY
            return ('unknown probability', 'word not in probability matrix')