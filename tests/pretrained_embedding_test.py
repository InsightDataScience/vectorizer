import unittest
import requests
import json

# this tests requires the vectorizer server to be running on EC2
# to run local tests, local server needs to be started and URL needs to be edited

class EmbeddingInputTest(unittest.TestCase):
    def test_single_word_input_with_correct_spelling(self):
        input = {'text': 'test'}
        response = requests.get('http://vectorizer.host/embed', data=input)
        vector = json.loads(response.text)
        self.assertTrue(len(vector[0]) == 300)

    def test_several_word_input_with_correct_spelling(self):
        string_input = 'Test proper sentence input with multiple words.'
        input = {'text': string_input}
        response = requests.get('http://vectorizer.host/embed', data=input)
        vector = json.loads(response.text)
        self.assertTrue(len(vector[0]) == 300)

    def test_word_with_spelling_error(self):
        input = {'text': 'hadfasdf asdflkajlehasdfl'}
        response = requests.get('http://vectorizer.host/embed', data=input)
        vector = json.loads(response.text)
        self.assertTrue(len(vector[0]) == 300)

    def test_non_string_input(self):
        input = {'text': 5}
        response = requests.get('http://vectorizer.host/embed', data=input)
        vector = json.loads(response.text)
        self.assertTrue(len(vector[0]) == 300)

    def test_space_string_input(self):
        input = {'text': ' '}
        response = requests.get('http://vectorizer.host/embed', data=input)
        vector = json.loads(response.text)
        self.assertTrue(len(vector) == 0)

if __name__ == '__main__':
    unittest.main()
