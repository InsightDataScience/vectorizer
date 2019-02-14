import unittest

embedding_models_path = './vectorizer_api/vectorizer/word2vec_twitter_model'
sys.path.append(embedding_models_path)

import word2vecReader

class TwitterW2VModelTest(unittest.TestCase):
    """
    This test requires the source code to be changed. The default embedding model
    is the pretrained GloVe model.
    """
    def test_load_twitter_w2v_model(self):
        model_path = "./word2vec_twitter_model.bin"
        print("Loading the model, this can take some time...")
        model = word2vecReader.Word2Vec.load_word2vec_format(model_path,
        binary=True)
        self.assertTrue(len(model.vocab) > 0)

if __name__ == '__main__':
    unittest.main()
