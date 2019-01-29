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

    email_data = data.get_data_from_s3()
    exit()

    #email_data = data.read_csv(input_filepath)

    log.info('Extracting only email content')
    email_content = email_data['content']

    log.info('Going to preprocess data')
    preprocessed_data = preprocess.PreprocessText(email_content).preprocessed_text

    log.info('Successfully preprocessed data')

    log.info('Starting to create Ngram datasets for model')

    ngram_data = ngram.Ngram(preprocessed_data)

    # Add loop so users can answer questions multiple times

    #take user input
    before_blank_tokens, after_blank_tokens = utilities.takeInput()
    print(f'Before blank words: {before_blank_tokens}')
    print(f'After blank words: {after_blank_tokens}')

    ### PREDICTION
    predicted_word_forward = predict_next_word(before_blank_tokens, ngram_data.bigram_forward_probability, 'forward')
    predicted_word_backward = predict_next_word(before_blank_tokens, ngram_data.bigram_backward_probability, 'backward')


    #choose most probable words for prediction

def predict_next_word(words_before_or_after_blank, bigram_probability, direction):
    if direction == 'forward':
        look_up_word = words_before_or_after_blank[-1] # Start with word right before blank
    elif direction == 'backward':
        look_up_word = words_before_or_after_blank[0]  # Start with first word after blank
    else:
        raise RuntimeError('Specify direction as forward or backward for ngram_probability function.')
        # ? Is this the right error to raise?

    if look_up_word in bigram_probability:
        token_probabilities = bigram_probability[look_up_word]
        sorted_probabilities = sorted(token_probabilities, key=lambda x: x[0], reverse=True)
        print(f'Final word probabilities for {direction} direction')
        print(sorted_probabilities[:10])
        return sorted_probabilities[0]


if __name__ == "__main__":
    logging.info("Going to run main function")
    main()
