import click
import logging
import data
import preprocess
import utilities
import ngram
import pandas as pd
from sklearn.metrics import accuracy_score
from time import time


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

    if run_test:
        training_emails, testing_emails = data.train_test_data(email_content)
        preprocessed_training_emails = preprocess.PreprocessText(training_emails).preprocessed_text
        ngram_data = ngram.Ngram(preprocessed_training_emails)
        data.create_test_data(testing_emails)
        test_fib_dataframe = pd.read_csv('test_fill_in_the_blank.csv')
        forward_answers = []
        backward_answers = []
        test_fib_dataframe['fill in the blank'].apply(lambda fib: answer_fib(fib, forward_answers, backward_answers, ngram_data))
        test_fib_dataframe['Forward Answers'] = [i[1] for i in forward_answers]
        test_fib_dataframe['Forward Answer Probability'] = [i[0] for i in forward_answers]
        test_fib_dataframe['Backward Answers'] = [i[1] for i in backward_answers]
        test_fib_dataframe['Backward Answer Probability'] = [i[0] for i in backward_answers]
        test_fib_dataframe.to_csv('test_fib_answers.csv')
        create_summary_statistics(test_fib_dataframe['answer'], test_fib_dataframe['Forward Answers'], test_fib_dataframe['Backward Answers'],)
        logging.info("Done running tests")

        exit()

    # TODO Add loop so users can answer questions multiple times
    if cli:
        preprocessed_data = preprocess.PreprocessText(email_content).preprocessed_text
        ngram_data = ngram.Ngram(preprocessed_data)
        #take user input
        before_blank_tokens, after_blank_tokens = utilities.take_input('cli')
        log.info(f'Before blank words: {before_blank_tokens}')
        log.info(f'After blank words: {after_blank_tokens}')

        ### PREDICTION
        predict_next_word(before_blank_tokens, ngram_data.bigram_forward_probability, 'forward')
        predict_next_word(before_blank_tokens, ngram_data.bigram_backward_probability, 'backward')
        #choose most probable words for prediction



def create_summary_statistics(correct_answer, forward_answer, backward_answer):
    forward_accuracy = accuracy_score(correct_answer, forward_answer)
    backward_accuracy = accuracy_score(correct_answer, backward_answer)
    summary_statistics = open("summary_statistics.txt", "a")
    summary_statistics.write(f'The forward model has an accuracy of: {forward_accuracy}\n')
    summary_statistics.write(f'The backward model has an accuracy of: {backward_accuracy}\n')
    summary_statistics.close()



def answer_fib(fib, forward_answers, backward_answers, ngram_data):
    before_blank_tokens, after_blank_tokens = utilities.take_input(fib)
    ### PREDICTION
    forward_answers.append(predict_next_word(before_blank_tokens, ngram_data.bigram_forward_probability, 'forward'))
    backward_answers.append(predict_next_word(before_blank_tokens, ngram_data.bigram_backward_probability, 'backward'))
    # choose most probable words for prediction
    return None


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
        logging.info(f'Final word probabilities for {direction} direction:{sorted_probabilities[:3]}')
        return sorted_probabilities[0]
    else:
        # TODO FIX THIS PART WHICH IS WHAT TO DO IF THE LOOK UP WORD IS NOT IN THE PROBABILITY
        return ('unknown probability', 'word not in probability matrix')


if __name__ == "__main__":
    logging.info('Going to run main function.')
    start = time()
    main()
    end = time()
    time = start - end
    summary_statistics = open("summary_statistics.txt", "a")
    print("here")
    summary_statistics.write(f'The program ran for: {time}\n')
    summary_statistics.close()