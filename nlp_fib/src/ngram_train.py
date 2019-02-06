import logging
from nltk.util import ngrams
import nltk
from collections import Counter
import data
import pandas as pd


class NgramTrain:
    def __init__(self, preprocessed_dataframe, input_file_path):
        self.log = logging.getLogger('Enron_email_analysis.ngram')
        self.log.info('Starting to create ngram model inputs.')

        # Do we need this? Keeping for now, but delete in future if not.
        self.word_in_document_count = self.word_in_document_counter(preprocessed_dataframe)

        # TODO: You could try to just update unigram_counter and bigram_counter directly without passing it to the function
        # and returning the value too. It has been initialized so I think you can update and access it directly.
        unigram_counter = Counter()
        self.unigrams, self.unigram_count = self.ngram_generator_and_counter(preprocessed_dataframe, 1, unigram_counter)
        self.log.info(f'Unigram count example: {list(self.unigram_count)[:1]}')

        bigram_counter = Counter()
        self.bigrams, self.bigram_count = self.ngram_generator_and_counter(preprocessed_dataframe, 2, bigram_counter)
        self.log.info(f'Bigram count: {list(self.bigram_count)[:1]}')
        data.write_pickle_file(self.bigram_count, input_file_path, 'bigram_count.pkl', True)

        self.bigram_forward_probability = Counter()
        self.ngram_probability(self.unigram_count, self.bigram_count, self.bigram_forward_probability, 'forward')
        self.log.info(f'Forward bigram probability example: {self.bigram_forward_probability.most_common(1)}')
        data.write_pickle_file(self.bigram_forward_probability, input_file_path, 'bigram_forward_probability.pkl', True)

        self.bigram_backward_probability = Counter()
        self.ngram_probability(self.unigram_count, self.bigram_count, self.bigram_backward_probability, 'backward')
        self.log.info(f'Backward bigram probability example: {self.bigram_backward_probability.most_common(1)}')
        data.write_pickle_file(self.bigram_backward_probability, input_file_path, 'bigram_backward_probability.pkl', True)

    def word_in_document_counter(self, preprocessed_dataframe):
        """This is creating a word count per document."""
        return preprocessed_dataframe.apply(lambda row: nltk.FreqDist(row))

    def ngram_generator_and_counter(self, preprocessed_dataframe, n, counter):
        # TODO: Explain logic well here. It is a little confusing.
        # TODO: Better variable names here

        self.log.info(f'Creating {n} grams and ngram counts')
        list_of_ngrams = [] # We need to append the generator objects items to this because it disappears after returning once
        if n>1:
            # This if statement exists if we are calculating bigrams or above because then the nltk ngram function
            # doesn't work properly unless we are iterating through the documents and calculating ngrams
            for row in preprocessed_dataframe:
                list_of_ngrams.append(list(ngrams(row, n)))
            for x in list_of_ngrams:
                counter.update(x)
            return list_of_ngrams, counter
        ngrams_for_all_docs = ngrams(preprocessed_dataframe,n)
        for doc in ngrams_for_all_docs:
            for x in doc:
                list_of_ngrams.append(x)
                counter.update(x)
        return list_of_ngrams, counter

    def ngram_probability(self, unigram_count, ngram_count, ngram_probability, direction):
        """for creating prob dict for bigram probabilities
        creates dict for storing probable words with their probabilities for a trigram sentence
        ADD 1 Smoothing used"""

        logging.info(f'Calculating ngram probabilities in the {direction} direction')
        unique_word_count = len(unigram_count)
        # create a dictionary of probable words with their probabilities for bigram probabilites
        for ngram_token in ngram_count:
            # unigram for key
            if direction=='forward':
                unigram_token = ngram_token[0]
            elif direction=='backward':
                unigram_token = ngram_token[-1]  # Start with the second or last word first and count backwards
            else:
                raise RuntimeError('Specify direction as forward or backward for ngram_probability function.')
                # ? Is this the right error to raise?

            # find the probability and add 1 smoothing has been used
            probability = (ngram_count[ngram_token] + 1) / (unigram_count[unigram_token] + unique_word_count)
            # HELP ? FOR BACKWARDS I AM ASSUMING THE NUMERATOR IS THE SAME AS FORWARDS PROBABILITY

            # bi_prob_dict is a dict of list and if the unigram sentence is not present in the Dictionary then add it
            if unigram_token not in ngram_probability:
                ngram_probability[unigram_token] = []
                if direction == 'forward':
                    last_ngram_token = ngram_token[-1]
                elif direction == 'backward':
                    last_ngram_token = ngram_token[0]
                ngram_probability[unigram_token].append([probability, last_ngram_token])
            # the unigram sentence is present but the probable word is missing,then add it
            else:
                ngram_probability[unigram_token].append([probability, last_ngram_token])


