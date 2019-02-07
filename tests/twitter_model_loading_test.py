from ..vectorizer_api.vectorizer.word2vec_twitter_model.word2vecReader import word2vecReader 

try:
    test = word2vecReader()
except:
    print('load failed')
