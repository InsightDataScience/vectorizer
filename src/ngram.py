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
        self.unigrams = self.ngram_generator(preprocessed_dataframe, 1)
        self.word_in_document_count = self.word_in_document_counter(preprocessed_dataframe)
        self.unigram_count = self.ngram_counter(Counter(), self.unigrams)

        # self.word_count = self.word_counter(Counter(), preprocessed_dataframe) # same as unigram_count, not used


    def ngram_generator(self, preprocessed_dataframe, n):
        self.log.info(f'Creating {n} grams')
        return ngrams(preprocessed_dataframe,n)

    def word_in_document_counter(self, preprocessed_dataframe):

        # Sci kit learn version below is an option
        # I think the sci kit learn one saves more memory
        #count_vectorizer = CountVectorizer()
        #preprocessed_dataframe.apply(lambda row: count_vectorizer.fit_transform(row))

        # NLTK version below. This is creating a word count per document.
        return preprocessed_dataframe.apply(lambda row: nltk.FreqDist(row))

    def ngram_counter(self, counter, ngram):
        for doc in ngram:
            for word in doc:
                counter.update(word)
        return counter

    # I don't think I need this anymore. It was originally used to create word count, but I greated ngram_counter
    # to be able to count not only individual words but also bigrams etc. Saving this just in case it fails.
    #
    # def word_counter(self, counter, preprocessed_dataframe):
    #     # Because the data is stored as a list of lists of word, I need to iterate through them to be able to count.
    #     for row in preprocessed_dataframe:
    #         counter.update(row)
    #     return counter