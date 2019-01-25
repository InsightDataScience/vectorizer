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
        self.word_count = self.word_counter(Counter(), preprocessed_dataframe)
        print(self.word_count)

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

    def word_counter(self, counter, preprocessed_dataframe):
        # Because the data is stored as a list of lists of word, I need to iterate through them to be able to count.
        for row in preprocessed_dataframe:
            counter.update(row)
        return counter

    def count_unigram(self, preprocessed_dataframe, bi_dict, vocab_dict, nlp):

        token = []
        # total no. of words in the corpus
        word_len = 0

        # open the corpus file and read it line by line
        # read in better
        spacy_tokenized_emails = []
        corpus_size = 0
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()  # checking to see if line is blank
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
        return vocab_dict
