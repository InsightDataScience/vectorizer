#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import data
import gensim.downloader as api
import utilities

__author__ = "Pujaa Rajan"
__email__ = "pujaa.rajan@gmail.com"


class NgramTest:
    """ The purpose of this class is to test the N gram model. """

    def __init__(self, test_file_path, output_file_path):
        self.log = logging.getLogger('Enron_email_analysis.ngram_test')
        self.test_fill_in_the_blank = test_file_path

        self.bigram_forward_probability = data.read_pickle_file(f'model_input_data/bigram_forward_probability.pkl')
        self.bigram_backward_probability = data.read_pickle_file(f'model_input_data/bigram_backward_probability.pkl')

        self.trigram_forward_probability = data.read_pickle_file(f'model_input_data/trigram_forward_probability.pkl')
        self.trigram_backward_probability = data.read_pickle_file(f'model_input_data/trigram_backward_probability.pkl')

        forward_answers = []
        backward_answers = []
        merged_answers = []
        self.test_fill_in_the_blank['fill in the blank'].apply(
            lambda fib: self.answer_fib_file(fib, forward_answers, backward_answers, merged_answers,
                                             self.bigram_forward_probability, self.bigram_backward_probability,
                                             self.trigram_forward_probability, self.trigram_backward_probability))

        self.test_fill_in_the_blank['Forward Answers'] = self.get_values(forward_answers, 1)
        self.test_fill_in_the_blank['Forward Answer Probability'] = self.get_values(forward_answers, 0)
        self.test_fill_in_the_blank['Backward Answers'] = self.get_values(backward_answers, 1)
        self.test_fill_in_the_blank['Backward Answer Probability'] = self.get_values(backward_answers, 0)

        self.test_fill_in_the_blank['Merged Ngram Answers'] = self.get_values(merged_answers, 1)
        self.test_fill_in_the_blank['Merged Ngram Probability'] = self.get_values(merged_answers, 0)
        word_vectors = api.load("glove-wiki-gigaword-100")

        word2vec_answers = self.test_fill_in_the_blank['answer'].apply(
            lambda row: data.get_similar_words(row, word_vectors)).to_list()

        self.test_fill_in_the_blank['Word2Vec Similar Words'] = self.get_values(word2vec_answers, 0)
        self.test_fill_in_the_blank['Word2Vec Probability'] = self.get_values(word2vec_answers, 1)

        self.test_fill_in_the_blank.dropna(inplace=True)
        self.test_fill_in_the_blank.to_csv(f'{output_file_path}/test_fib_answers.csv')

    @staticmethod
    def get_values(answers, n):
        """ A helper function to help extract the correct values from lists of lists of tuples. """

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

    @staticmethod
    def answer_fib_file(fib, forward_answers, backward_answers, merged_answers, bigram_forward_probability,
                        bigram_backward_probability, trigram_forward_probability, trigram_backward_probability):
        """ Returns answers to the input test file sentences that were missing a word. """
        before_blank_tokens, after_blank_tokens, word_to_replace = utilities.take_input(fib)
        predicted_forward_words = data.predict_next_word(before_blank_tokens, bigram_forward_probability,
                                                         trigram_forward_probability, 'forward')
        forward_answers.append(predicted_forward_words)
        predicted_backward_words = data.predict_next_word(after_blank_tokens, bigram_backward_probability,
                                                          trigram_backward_probability, 'backward')
        backward_answers.append(predicted_backward_words)  # choose most probable words for prediction
        if predicted_forward_words and predicted_backward_words:
            temp_merge_answers = predicted_forward_words + predicted_backward_words
            no_nones_merge_answers = [x for x in temp_merge_answers if x is not None]
            sorted_merged_answers = sorted(no_nones_merge_answers, key=lambda x: x[0], reverse=True)
            merged_answers.append(sorted_merged_answers)
        else:
            merged_answers.append(None)
