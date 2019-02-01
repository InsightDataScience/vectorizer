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
    def __init__(self, testing_emails, ngram_data):
        self.log = logging.getLogger('Enron_email_analysis.ngram')
        self.create_test_data(testing_emails)
        self.test_fib_dataframe = pd.read_csv('test_fill_in_the_blank.csv')
        forward_answers = []
        backward_answers = []
        self.test_fib_dataframe['fill in the blank'].apply(
            lambda fib: self.answer_fib(fib, forward_answers, backward_answers, ngram_data))
        self.test_fib_dataframe['Forward Answers'] = self.get_values(forward_answers, 1)
        self.test_fib_dataframe['Forward Answer Probability'] = self.get_values(forward_answers, 0)
        self.test_fib_dataframe['Backward Answers'] = self.get_values(backward_answers, 1)
        self. test_fib_dataframe['Backward Answer Probability'] = self.get_values(backward_answers, 0)
        self.test_fib_dataframe.to_csv('test_fib_answers.csv')

    def answer_fib(self, fib, forward_answers, backward_answers, ngram_data):
        before_blank_tokens, after_blank_tokens = utilities.take_input(fib)
        ### PREDICTION
        forward_answers.append(ngram_data.predict_next_word(before_blank_tokens, ngram_data.bigram_forward_probability, 'forward'))
        backward_answers.append(ngram_data.predict_next_word(before_blank_tokens, ngram_data.bigram_backward_probability, 'backward'))
        # choose most probable words for prediction
        return None

    def get_values(self, answers, n):
        output_list = []
        for sentence_answers in answers:
            output_list.append([i[n] for i in sentence_answers])
        return output_list

    def create_test_data(self, dataframe):
        """Create test data. Input is the test part of the train_test_data function. Output is csv with input and labels."""
        self.log.info("in create test data")
        dataframe_no_nan = dataframe.dropna()
        sentence_dataframe = dataframe_no_nan.apply(lambda row: tokenize.sent_tokenize(row))
        good_sentences_only = []
        sentence_dataframe.apply(lambda row: self.check_if_good_sentence(row, good_sentences_only))
        fill_in_blanks = self.create_blanks(good_sentences_only)
        data.write_to_csv(fill_in_blanks)
        logging.info("End of creating test data")
        return None

    def create_blanks(self, good_sentences_only):
        fill_in_blanks = []
        for sentence in good_sentences_only:
            word_index = randint(3, len(word_tokenize(sentence)) - 4)
            words_in_sentence = word_tokenize(sentence)
            answer = words_in_sentence[word_index]  # Saving the answer before replacing it with a blank
            words_in_sentence[word_index] = '_'  # Replacing with a blank
            full_sentence = " ".join(words_in_sentence)
            fill_in_blanks.append((full_sentence, answer))
        return fill_in_blanks

    def check_if_good_sentence(self, row, good_sentences_only):
        # TODO DOESN'T WORK
        for s in row:
            m = re.search(r'https?:|@\w+|\d', s)
            n = re.search(r'[^A-Za-z\s!?.]', s)
            if m or n:
                pass
            elif len(word_tokenize(s)) < 8:
                pass
            else:
                good_sentences_only.append(s[:-1])