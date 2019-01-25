from collections import defaultdict
import logging


class Corpus:
    def __init__(self, data):
        self.unigram_counts = count_unigram(data)
        self.total_words = len(self.unigram_counts)
        self.bigram_dict = default_dict(int)

    def count_unigram(self, data, bi_dict, vocab_dict, nlp):
        w1 = ''  # for storing the 3rd last word to be used for next token set
        w2 = ''  # for storing the 2nd last word to be used for next token set
        w3 = ''  # for storing the last word to be used for next token set
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
