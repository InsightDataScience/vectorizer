import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import utilities

class PreprocessText():
    def __init__(self, dataframe):
        self.log = logging.getLogger('Enron_email_analysis.preprocess')
        self.preprocessed_text = self.preprocess(dataframe) # Returns dataframe

    def preprocess(self, dataframe):
        # TODO: Find a way to run these all at once. Doing these individually is inefficient.
        # TODO: Consider using *args or *kwargs
        # TODO: Assuming we want to run these all right now. Use true false to indicate whether to run in future.
        # TODO: Bad naming of dataframeX below.
        # TODO: These functions still need to be fully tested.
        # TODO: Add try excepts to all functions to be able to tell what fails.
        # TODO: Add logging saying each function succeeded if it did.

        self.log.info('Starting to pre-process text data.')

        dataframe1 = self.remove_nan(dataframe)
        dataframe2 = self.lowercase(dataframe1)

        dataframe3 = self.remove_whitespace(dataframe2)

        # Remove emails and websites before removing special characters
        dataframe4 = self.remove_emails(dataframe3)
        dataframe5 = self.remove_website_links(dataframe4)

        dataframe6 = self.remove_special_characters(dataframe5)
        dataframe7 = self.remove_numbers(dataframe6)
        self.remove_stop_words()
        dataframe8 = self.tokenize(dataframe7)

        self.log.info(f'Successfully finished pre-processing text data.')

        return dataframe8

    def remove_nan(self, dataframe):
        """Pass in a dataframe to remove NAN from those columns."""
        return dataframe.dropna()

    def lowercase(self, dataframe):
        logging.info('Converting text data to lowercase')
        lowercase_dataframe = dataframe.str.lower()
        return lowercase_dataframe

    def remove_whitespace(self, dataframe):
        self.log.info('Removing whitespace from text data')
        # replace more than 1 space with 1 space
        merged_spaces = dataframe.str.replace(r"\s\s+", ' ', regex=True)
        # delete beginning and trailing spaces
        trimmed_spaces = merged_spaces.apply(lambda x: x.strip())
        return trimmed_spaces

    def remove_emails(self, dataframe):
        self.log.info('Removing emails from text data')
        no_emails = dataframe.str.replace('\S*@\S*\s?', '')
        return no_emails

    def remove_website_links(self, dataframe):
        self.log.info('Removing website links from text data')
        no_website_links = dataframe.str.replace('http\S+', '')
        return no_website_links

    def remove_special_characters(self, dataframe):
        self.log.info('Removing special characters from text data')
        no_special_characters = dataframe.replace(r'[^A-Za-z0-9 ]+', '', regex=True)
        return no_special_characters

    def remove_numbers(self, dataframe):
        self.log.info('Removing numbers from text data')
        no_numbers = dataframe.str.replace('\d+', '')
        return no_numbers

    def remove_stop_words(self):
        # TODO: An option to pass in a custom list of stopwords would be cool.
        logging.info('Removing stop words from text data')
        nltk.download('stopwords')
        set(stopwords.words('english'))

    def tokenize(self, dataframe):
        logging.info('Tokenizing text data')
        tokenized_dataframe = dataframe.apply(lambda row: word_tokenize(row))
        return tokenized_dataframe

    def expand_contractions(self, dataframe):
        # TODO: Not a priority right now. Come back to this later.
        return dataframe