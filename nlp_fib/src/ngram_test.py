import data
import pandas as pd
import utilities
import data
import logging

__author__ = "Pujaa Rajan"
__email__ = "pujaa.rajan@gmail.com"

class NgramTest:
    """ This class takes in the probability files
    """
    def __init__(self, test_file_path, output_file_path):
        self.log = logging.getLogger('Enron_email_analysis.ngram_test')
        self.test_fill_in_the_blank = test_file_path

        self.bigram_forward_probability = data.read_pickle_file(f'model_input_data/bigram_forward_probability.pkl')
        self.bigram_backward_probability = data.read_pickle_file(f'model_input_data/bigram_backward_probability.pkl')

        self.trigram_forward_probability = data.read_pickle_file(f'model_input_data/trigram_forward_probability.pkl')
        self.trigram_backward_probability = data.read_pickle_file(f'model_input_data/trigram_backward_probability.pkl')

        forward_answers = []
        backward_answers = []
        self.test_fill_in_the_blank['fill in the blank'].apply(
            lambda fib: self.answer_fib(fib, forward_answers, backward_answers, self.bigram_forward_probability, self.bigram_backward_probability, self.trigram_forward_probability, self.trigram_backward_probability))
        self.test_fill_in_the_blank['Forward Answers'] = self.get_values(forward_answers, 1)
        self.test_fill_in_the_blank['Forward Answer Probability'] = self.get_values(forward_answers, 0)
        self.test_fill_in_the_blank['Backward Answers'] = self.get_values(backward_answers, 1)
        self.test_fill_in_the_blank['Backward Answer Probability'] = self.get_values(backward_answers, 0)
        self.test_fill_in_the_blank.dropna(inplace=True)
        self.test_fill_in_the_blank.to_csv(f'{output_file_path}/test_fib_answers.csv')

    def get_values(self, answers, n):
        output_list = []
        for sentence_answers in answers:
            if sentence_answers is not None:
                sentence_list = []
                for word_probability in sentence_answers:
                    if word_probability is not None:
                        sentence_list.append(word_probability[n])
                output_list.append(sentence_list)
            else:
                output_list.append(None)
        return output_list

    def answer_fib_file(self, fib, forward_answers, backward_answers, bigram_forward_probability, bigram_backward_probability, trigram_forward_probability, trigram_backward_probability):
    ### PREDICTION
        before_blank_tokens, after_blank_tokens = utilities.take_input(fib)
        forward_answers.append(data.predict_next_word(before_blank_tokens, bigram_forward_probability, trigram_forward_probability, 'forward'))
        backward_answers.append(data.predict_next_word(after_blank_tokens, bigram_backward_probability, trigram_backward_probability, 'backward'))        # choose most probable words for prediction

