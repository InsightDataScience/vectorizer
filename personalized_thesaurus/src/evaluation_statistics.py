#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Pujaa Rajan"
__email__ = "pujaa.rajan@gmail.com"


class Evaluation:
    """ This class generates the summary statistics output. """

    def __init__(self, ngram_test):
        self.correct_answer = ngram_test.test_fill_in_the_blank['answer']
        self.forward_ngram_model_answer = ngram_test.test_fill_in_the_blank['Forward Answers']
        self.backward_ngram_model_answer = ngram_test.test_fill_in_the_blank['Backward Answers']
        self.merged_ngram_model_answer = ngram_test.test_fill_in_the_blank['Merged Ngram Answers']

        self.forward_accuracy = self.calculate_accuracy(self.correct_answer, self.forward_ngram_model_answer)
        self.backward_accuracy = self.calculate_accuracy(self.correct_answer, self.backward_ngram_model_answer)
        self.merged_accuracy = self.calculate_accuracy(self.correct_answer, self.merged_ngram_model_answer)
        self.create_summary_statistics(self.forward_accuracy, self.backward_accuracy, self.merged_accuracy)

    @staticmethod
    def calculate_accuracy(correct_answer, model_answer):
        """ Returns the accuracy of the model's output compared to the correct answer. """
        correct_count = 0
        correct_answer_index = 0
        correct_answer_list = correct_answer.tolist()
        for answer_list in model_answer.tolist():
            if correct_answer_list[correct_answer_index] in answer_list:
                correct_count = correct_count + 1
            correct_answer_index = correct_answer_index + 1
        return correct_count / len(correct_answer)

    @staticmethod
    def create_summary_statistics(forward_accuracy, backward_accuracy, merged_accuracy):
        """ Writes the summary statistics to an output file. """
        summary_statistics = open(f'summary_statistics.txt', 'a')
        summary_statistics.write(f'The forward model has an accuracy of: {forward_accuracy}\n')
        summary_statistics.write(f'The backward model has an accuracy of: {backward_accuracy}\n')
        summary_statistics.write(f'The merged model has an accuracy of: {merged_accuracy}\n')
        summary_statistics.close()
