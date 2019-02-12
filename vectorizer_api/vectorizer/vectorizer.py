from vectorizer import app
from flask import Flask, request, jsonify
# from flask_restful import Resource, Api

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
    preprocessed_text = preprocess.tokenize(cleaned_text)

    # embedding
    embedded_text = embed.inference_glove_embedding(preprocessed_text,
        averaged_embedding=True)

    # convert to list in order to jsonify
    embedded_text_list = embedded_text.tolist()
    return jsonify(embedded_text_list)

if __name__ == '__main__':
    app.run(debug=True)
