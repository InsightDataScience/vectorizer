from vectorizer import app
from flask import Flask, flash, request, redirect, url_for
from flask_restful import Resource, Api
from werkzeug.utils import secure_filename

import spacy

from vectorizer import clean
from vectorizer import preprocess
from vectorizer import embed

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

@app.route('/infer', methods=['GET', 'POST'])
def infer():
    text = request.form.get('text')

    # cleaning
    text = clean.remove_URL(text)
    text = clean.remove_special_characters(text)
    cleaned_text = clean.lowercase(text)

    # preprocessing
    preprocessed_text = preprocess.inference_tokenize(cleaned_text)

    # embedding
    embedded_text = embed.inference_glove_embedding(preprocessed_text)

    return str(embedded_text)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
    print(type(file))
    return 'file upload successful'

if __name__ == '__main__':
    app.run(debug=True)
