# vectorizer

Vectorizer is a natural language vectorization tool that allows users to get vectors for their text inputs. It utilizes the most common practices to clean, process, and embed input data to generate vectors.

Vectorizer is currently serving as a RESFUl API.

[Here](http://bit.ly/vectorizer_slides) are the slides for project Vectorizer.
---
## Setup Environment
The following setup instructions is for if you want to clone the repo to run locally. If you would like to use the API service directly, refer to [API call instructions](#api-call-instructions).

Set up conda environement with requirements.yml
```bash
conda env create -f environment.yml
source activate nlp
```

Additional spaCy dependencies
```bash
python -m spacy download en_vectors_web_lg

```

## API call instructions
```python
import requests
import json

input = {'text': 'iput_text'}
response = requests.get('http://vectorizer.host/embed', data=input)
vector = json.loadas(response.text)
```

## Sample applications

Two sample use cases for this service can be found under the example directory which are twitter sentiment classification and similar tweet recommendation.
