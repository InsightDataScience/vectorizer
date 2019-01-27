import click
import logging
import data
import preprocess
import utilities
import ngram

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Main function
    """
    utilities.logger()
    log = logging.getLogger('Enron_email_analysis.main')

    log.info('Starting to run main.py')

    log.info('Going to read in Enron email data')
    email_data = data.read_csv(input_filepath)

    log.info('Extracting only email content')
    email_content = email_data['content']

    log.info('Going to preprocess data')
    preprocessed_data = preprocess.PreprocessText(email_content).preprocessed_text

    log.info('Successfully preprocessed data')

    log.info('Starting to create Ngram datasets for model')

    saved_ngram = ngram.Ngram(preprocessed_data)

    while True:
        #take user input
        before_blank_tokens, after_blank_tokens = utilities.takeInput()
        print(f'Before blank words: {before_blank_tokens}')
        print(f'Before blank words: {after_blank_tokens}')

        ### PREDICTION
        predicted_word = predict_next_word_forward(before_blank_tokens, saved_ngram.bigram_probability)

        #choose most probable words for prediction
        print('Word Prediction:', predicted_word)


def predict_next_word_forward(before_blank_words, bigram_probability):
    if before_blank_words[-1] in bigram_probability:
        token_probabilities = bigram_probability[before_blank_words[-1]]
        sorted_probabilities = sorted(token_probabilities, key=lambda x: x[0], reverse=True)
        print(sorted_probabilities[:10])
        return sorted_probabilities[0]

def predict_next_word_backward(after_blank_tokens, bigram_probability):
    # TODO
    return None

if __name__ == "__main__":
    logging.info("Going to run main function")
    main()
