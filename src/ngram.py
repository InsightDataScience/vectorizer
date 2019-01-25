from collections import defaultdict
import logging
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from collections import Counter


class Ngram:
    def __init__(self, preprocessed_dataframe):
        self.log = logging.getLogger('Enron_email_analysis.ngram')
        self.log.info('Starting to create ngram model inputs')

        self.word_in_document_count = self.word_in_document_counter(preprocessed_dataframe)

        # TODO: You could try to just update unigram_counter and bigram_counter directly without passing it to the function
        # and returning the value too. It has been initialized so I think you can update and access it directly.
        unigram_counter = Counter()
        self.unigrams, self.unigram_count = self.ngram_generator_and_counter(preprocessed_dataframe, 1, unigram_counter)
        bigram_counter = Counter()
        self.bigrams, self.bigram_count = self.ngram_generator_and_counter(preprocessed_dataframe, 2, bigram_counter)

        self.bigram_probability = Counter()
        print(self.bigram_probability)
        self.ngram_probability(self.unigram_count, self.bigram_count, self.bigram_probability)
        print(self.bigram_probability)


    def word_in_document_counter(self, preprocessed_dataframe):

        # Sci kit learn version below is an option
        # I think the sci kit learn one saves more memory
        #count_vectorizer = CountVectorizer()
        #preprocessed_dataframe.apply(lambda row: count_vectorizer.fit_transform(row))

        # NLTK version below. This is creating a word count per document.
        return preprocessed_dataframe.apply(lambda row: nltk.FreqDist(row))

    def ngram_counter(self, counter, ngram, n):
        if n>1:
            for doc in ngram:
                print("this part doesn't work still")
                counter.update(doc)
        for doc in ngram:
            for word in doc:
                counter.update(word)
        return counter

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

    def ngram_probability(self, unigram_count, ngram_count, ngram_probability):
        """for creating prob dict for bigram probabilities
        creates dict for storing probable words with their probabilities for a trigram sentence
        ADD 1 Smoothing used"""

        unique_word_count = len(unigram_count)

        # create a dictionary of probable words with their probabilities for bigram probabilites
        for ngram_token in ngram_count:
            # unigram for key
            unigram_token = ngram_token[0]

            # find the probability and add 1 smoothing has been used
            probability = (ngram_count[ngram_token] + 1) / (unigram_count[unigram_token] + unique_word_count)

            # bi_prob_dict is a dict of list and if the unigram sentence is not present in the Dictionary then add it
            if unigram_token not in ngram_probability:
                ngram_probability[unigram_token] = []
                ngram_probability[unigram_token].append([probability, ngram_token[-1]])
            # the unigram sentence is present but the probable word is missing,then add it
            else:
                ngram_probability[unigram_token].append([probability, ngram_token[-1]])