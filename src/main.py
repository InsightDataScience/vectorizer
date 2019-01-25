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

    #take user input
    input_sen = utilities.takeInput()

    ### PREDICTION
    predicted_word = predict_next_word(input_sen, saved_ngram.bigram_probability)

    #choose most probable words for prediction
    print('Word Prediction:', predicted_word)

def predict_next_word(sentence, bigram_probability):
    token = sentence.split()
    if token[-1] in bigram_probability:
        token_probabilities = bigram_probability[token[-1]]
        sorted_probabilities = sorted(token_probabilities, key=lambda x: x[0], reverse=True)
        print(sorted_probabilities)
        return sorted_probabilities[0]

if __name__ == "__main__":
    logging.info("Going to run main function")
    main()
