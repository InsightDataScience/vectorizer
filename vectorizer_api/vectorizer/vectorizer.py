from vectorizer import app
from flask import Flask, request, jsonify

import spacy

from vectorizer import clean
from vectorizer import preprocess
from vectorizer import embed

@app.route('/embed', methods=['GET', 'POST'])
def infer():
    text = request.form.get('text')
    averaged_embedding = request.form.get('averaged_embedding')

    # cleaning
    cleaned_text = clean.clean_str(text)

    # preprocessing
    tokenized_text = preprocess.tokenize(cleaned_text)
    removed_stop_words = preprocess.remove_stop_words(tokenized_text)
    preprocessed_text = preprocess.lemmatize(removed_stop_words)

    # embedding
    embedded_text = embed.inference_glove_embedding(preprocessed_text,
        averaged_embedding=True)

    # convert to list in order to jsonify
    embedded_text_list = embedded_text.tolist()
    return jsonify(embedded_text_list)

if __name__ == '__main__':
    app.run(debug=True)
