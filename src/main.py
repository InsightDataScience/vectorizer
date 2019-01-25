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

    ngram.Ngram(preprocessed_data)

    quit()

    token_len = corpus.Corpus()

    loadCorpus(input_filepath, bi_dict, vocab_dict, nlp)

    param = [0.7,0.1,0.1,0.1]
    #create bigram Probability Dictionary
    findBigramProbAdd1(vocab_dict, bi_dict, bi_prob_dict)

    # sort the probability dictionaries
    sortProbWordDict(bi_prob_dict)

    #take user input
    input_sen = takeInput()

    ### PREDICTION
    #choose most probable words for prediction
    word_choice = chooseWords(input_sen, bi_prob_dict)
    prediction = doInterpolatedPredictionAdd1(input_sen, bi_dict, vocab_dict,token_len, word_choice, param)
    print('Word Prediction:',prediction)


if __name__ == "__main__":
    logging.info("Going to run main function")
    main()
