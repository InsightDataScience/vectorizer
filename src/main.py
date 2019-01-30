import click
import logging
import data
import preprocess
import utilities
import ngram

@click.command()
@click.argument('input_file_path', type=str, required=True)
@click.option('--cli', is_flag=True, required = False, help="Will run code using cli.")
@click.option('--run_test', is_flag=True, required=False, help='Will run train and test')
def main(input_file_path, run_test,  cli):
    """ Main function
    """
    utilities.logger()
    log = logging.getLogger('Enron_email_analysis.main')

    log.info('Starting to run main.py.')
    log.info('Going to read in Enron email data.')
    email_data = data.read_data(input_file_path) # Read in data from csv or s3 using file path
    log.info(f'Read in {email_data.shape[0]} emails.')
    log.info('Extracting only email content.')
    email_content = email_data['content']
    preprocessed_data = preprocess.PreprocessText(email_content).preprocessed_text
    ngram_data = ngram.Ngram(preprocessed_data)

    # TODO Add loop so users can answer questions multiple times
    if cli:
        #take user input
        before_blank_tokens, after_blank_tokens = utilities.take_input()
        log.info(f'Before blank words: {before_blank_tokens}')
        log.info(f'After blank words: {after_blank_tokens}')

        ### PREDICTION
        predict_next_word(before_blank_tokens, ngram_data.bigram_forward_probability, 'forward')
        predict_next_word(before_blank_tokens, ngram_data.bigram_backward_probability, 'backward')
        #choose most probable words for prediction

def predict_next_word(words_before_or_after_blank, bigram_probability, direction):
    if direction == 'forward':
        logging.info('Predicting the next word using a FORWARD n gram model.')
        look_up_word = words_before_or_after_blank[-1] # Start with word right before blank
    elif direction == 'backward':
        logging.info('Predicting the next word using a BACKWARD n gram model.')
        look_up_word = words_before_or_after_blank[0]  # Start with first word after blank
    else:
        raise RuntimeError('Specify direction as forward or backward for ngram_probability function.')
        # ? Is this the right error to raise?
    if look_up_word in bigram_probability:
        token_probabilities = bigram_probability[look_up_word]
        sorted_probabilities = sorted(token_probabilities, key=lambda x: x[0], reverse=True)
        logging.info(f'Final word probabilities for {direction} direction:{sorted_probabilities[:10]}')
        return sorted_probabilities[0]
    else:
        # ? FIX THIS PART WHICH IS WHAT TO DO IF THE LOOK UP WORD IS NOT IN THE PROBABILITY
        return 'word not in probability matrix.'


if __name__ == "__main__":
    logging.info('Going to run main function.')
    main()
