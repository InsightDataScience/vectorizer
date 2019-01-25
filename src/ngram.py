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

        unigram_counter = Counter()
        self.unigrams, self.unigram_count = self.ngram_generator_and_counter(preprocessed_dataframe, 1, unigram_counter)
        bigram_counter = Counter()
        self.bigrams, self.bigram_count = self.ngram_generator_and_counter(preprocessed_dataframe, 2, bigram_counter)
        print(self.bigram_count)

        # OLD CODE THAT I CHANGED ABOVE BUT SAVING THIS IN CASE IT FAILS

        # self.word_count = self.word_counter(Counter(), preprocessed_dataframe) # same as unigram_count, not used

        # self.unigrams = self.ngram_generator(preprocessed_dataframe, 1) # currently a generator object
        # self.unigram_count = self.ngram_counter(Counter(), self.unigrams, 1)

        # self.bigrams = self.ngram_generator(preprocessed_dataframe, 2) # list of bigrams
        # self.bigram_count = self.ngram_counter(Counter(), self.bigrams, 2)

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

    # I don't think I need this anymore. It was originally used to create word count, but I greated ngram_counter
    # to be able to count not only individual words but also bigrams etc. Saving this just in case it fails.
    #
    # def word_counter(self, counter, preprocessed_dataframe):
    #     # Because the data is stored as a list of lists of word, I need to iterate through them to be able to count.
    #     for row in preprocessed_dataframe:
    #         counter.update(row)
    #     return counter
    #
    # def ngram_generator(self, preprocessed_dataframe, n):
    #     self.log.info(f'Creating {n} grams')
    #     if n>1:
    #         # This if statement exists if we are calculating bigrams or above because then the nltk ngram function
    #         # doesn't work properly unless we are iterating through the documents and calculating ngrams
    #         list_of_ngrams = []
    #         for row in preprocessed_dataframe:
    #             list_of_ngrams.append(list(ngrams(row, n)))
    #         return list_of_ngrams
    #     return ngrams(preprocessed_dataframe,n)
