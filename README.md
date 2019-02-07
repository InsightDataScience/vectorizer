# NLP Preprocessing

This project is one of Insight Data Science's open initiative projects proposed by Manu Ameisen and it aims to address the pain points of the lack of best practices and resources/libraries available for natural language processing (NLP) workflows. It achieves this by by taking a significant part of a typical NLP workflow and building a service around it so data scientists can focus their work on modeling and other aspects of the workflow.

Specifically, this natural language vectorization service encapsulates three major steps necessary in a typical NLP workflow, namely: data cleaning, data preprocessing, and word vector generation.

Two sample use cases for this service can be found under the example directory which are twitter sentiment classification and similar tweet recommendation.

Another use case that uses this services but is more of an independent project is NLP Fill In The Blank by Pujaa Rajan. Further information can be found in the `nlp_fib` directory.

### Dependencies

```bash
python -m spacy download en_vectors_web_lg
```
