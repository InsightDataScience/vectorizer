import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

FILE_PATH = 'training.1600000.processed.noemoticon.csv'
NLTK_STOPWORDS = stopwords.words('english')

def main():
    textdata = pd.read_csv(FILE_PATH, header=None, usecols=[5])
    textdata[5].apply(clean_str)

    tokenizer = RegexpTokenizer(r'\w+')
    textdata[5] = textdata[5].apply(tokenizer.tokenize)
    textdata[5] = textdata[5].apply(remove_stopwords)

    corpus = ''
    for sample in textdata[5]:
        corpus = corpus + sample + "\n"

    with open("train.txt", "w") as text_file:
        print(corpus, file=text_file)

    return

def clean_str(string):
	# 2/2/19 Adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string cleaning
	"""
    string = re.sub(r"@\S+", "", string) # removes username handles
    string = re.sub(r"http\S+", "", string) # removes URL
    string = re.sub(r"http", "", string)
    string = re.sub(r"[^A-Za-z0-9]", " ", string) # removes special characters
    string = re.sub(r"\s{2,}", " ", string) # removes consecutive white spaces
    return string.strip().lower()

def remove_stopwords(tokens):
    return ' '.join([w for w in tokens if not w in NLTK_STOPWORDS])


if __name__ == '__main__':
	main()
