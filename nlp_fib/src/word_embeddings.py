class WordEmbeddings:
    """ This class reads in embeddings and attaches them to results
    """
    def __init__(self, input_sentence, output_file_path):
        self.log = logging.getLogger('Enron_email_analysis.word_embeddings')
        self.word_embeddings = test_file_path