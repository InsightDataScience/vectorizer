from vectorizer import app
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

import spacy

from vectorizer import clean
from vectorizer import preprocess
from vectorizer import embed

# api = Api(app)

@app.route('/infer', methods=['POST'])
def infer():
    text = request.form.get('text')
    print('raw text: {}'.format(text))

    # cleaning
    text = clean.remove_URL(text)
    text = clean.remove_special_characters(text)
    cleaned_text = clean.lowercase(text)

    # preprocessing
    preprocessed_text = preprocess.inference_tokenize(cleaned_text)

    # embedding
    embedded_text = embed.inference_glove_embedding(preprocessed_text)
    print(embedded_text)
    return str(embedded_text)

if __name__ == '__main__':
    app.run(debug=True)
