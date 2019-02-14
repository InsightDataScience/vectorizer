from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
import utilities
import logging
import data
import gensim.downloader as api
from collections import Counter, OrderedDict


class OrderedCounter(Counter, OrderedDict):
    pass

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


class ReusableForm(Form):
    input_sentence = TextField('Sentence:', validators=[validators.required()])
    word_to_replace = TextField('Word to Edit:', validators=[validators.required()])


@app.route('/', methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
    print(form.errors)
    if request.method == 'POST':
        sentence = request.form['input_sentence']
        word_to_replace = request.form['word_to_replace']
        print(sentence, " ", " ", word_to_replace)
        before_blank_tokens, after_blank_tokens, word_to_replace = utilities.take_input('flask_app', sentence=sentence, word_to_replace=word_to_replace)

    if form.validate():
        # Save the comment here.
        after_predictions = data.predict_next_word(before_blank_tokens, bigram_forward_probability,
                                                   trigram_forward_probability, 'forward')
        before_predictions = data.predict_next_word(after_blank_tokens, bigram_backward_probability,
                                                    trigram_backward_probability, 'backward')
        merged_predictions = after_predictions + before_predictions
        removed_dupes = OrderedCounter([word[1] for word in merged_predictions]).keys()
        final_answers =[word for word in removed_dupes if not word == word_to_replace]
        word_embedding_output = data.get_similar_words(word_to_replace, word_vectors)

        flash(f'Here is your sentence:\t\t{sentence}')
        flash(f'Here is the word you want to replace:\t\t{word_to_replace}')
        flash('Here are suggestions for this word based on your previous emails:\t\t' + " ".join(final_answers))
        flash('Here are the most similar words:\t\t' + " ".join([word[0] for word in word_embedding_output]))

    else:
        flash('Error: All the form fields are required. ')

    return render_template('hello.html', form=form)


if __name__ == "__main__":
    utilities.logger()
    log = logging.getLogger('Enron_email_analysis.main')
    log.info("Welcome to the Personalized Thesaurus.")
    log.info("ABOUT: This thesaurus recommends you the best word based on your previous emails and the most similar word.")
    log.info("Starting to reading in forward and backward probability pickle files")
    bigram_forward_probability = data.read_pickle_file(f'model_input_data/bigram_forward_probability.pkl')
    log.info("Successfully finished reading in 1/4 pickle files.")
    bigram_backward_probability = data.read_pickle_file(f'model_input_data/bigram_backward_probability.pkl')
    log.info("Successfully finished reading in 2/4 pickle files.")

    trigram_forward_probability = data.read_pickle_file(f'model_input_data/trigram_forward_probability.pkl')
    log.info("Successfully finished reading in 3/4 pickle files.")
    trigram_backward_probability = data.read_pickle_file(f'model_input_data/trigram_backward_probability.pkl')
    log.info("Successfully finished reading in 4/4 pickle files.")

    log.info("Starting to load word embedding.")
    word_vectors = api.load("glove-wiki-gigaword-100")
    log.info("Successfully loaded the word embedding.")

    app.run(use_reloader=False)