from vectorizer import app
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

import spacy

from vectorizer import clean
from vectorizer import preprocess

# api = Api(app)

@app.route('/infer', methods=['POST'])
def infer():
    tweet = request.form.get('tweet')
    print('raw tweet: {}'.format(tweet))

    # cleaning
    tweet = clean.remove_URL(tweet)
    tweet = clean.remove_special_characters(tweet)
    cleaned_tweet = clean.lowercase(tweet)
    print('cleaned tweet: {}'.format(tweet))

    # preprocessing
    preprocessed_tweet = preprocess.inference_tokenize(cleaned_tweet)

    # embedding
    # TODO

    return tweet

    # word = request.args['word']
    # glove_embedding = spacy.load('en_vectors_web_lg')
    # vector = glove_embedding.vocab.get_vector(word)
    # return str(vector)

# class HelloWorld(Resource):
#     def get(self):
#         return {'hello': 'world'}

if __name__ == '__main__':
    app.run(debug=True)
