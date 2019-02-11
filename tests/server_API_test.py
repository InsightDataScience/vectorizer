import unittest
import requests
import json

class APITests(unittest.TestCase):
    def test_domain_name(self):
        input = {'text': 'test'}
        response = requests.get('http://vectorizer.host/embed', data=input)
        vector = json.loads(response.text)
        self.assertTrue(len(vector[0]) == 300)

    def test_make_consecutive_request(self):
        for i in range(1000):
            input = {'text': 'test'}
            response = requests.get('http://vectorizer.host/embed', data=input)
            vector = json.loads(response.text)
            self.assertTrue(len(vector[0]) == 300)


if __name__ == '__main__':
    unittest.main()
