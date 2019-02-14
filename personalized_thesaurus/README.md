# Personalized Thesaurus

## About

Personalized Thesaurus suggests a word for you in the middle of a sentence using phrases you've used in the past and the context of the surrounding words. 

## Set Up

**STEP 1.** Clone this Github repository.

`git clone https://github.com/InsightDataCommunity/vectorizer.git`

**STEP 2.** `cd vectorizer/`

**STEP 3.** Install a Python virtual environment.

`python3 -m pip install --user virtualenv`

**STEP 4.** Activate the virtual environment.

`source env/bin/activate`

**STEP 5.** Install dependencies.

`pip install -r requirements.txt`

**STEP 6.** Change to project folder.

`cd personalized_thesaurus/`

## How to Run 

**STEP 1.** Create the input and output datasets.

`python3 src/main.py --input_file_path input/ --create_train_test_data input/enron_sample_1000.csv --output_file_path /output`

**STEP 2.** Train the model.

`python3 src/main.py  --output_file_path output/ --run_ngram_train --training_data_file_path input/training_email_data.csv`

**STEP 3.** Test the model. (OPTIONAL)

`python3 src/main.py  --output_file_path output/ --run_ngram_test --testing_data_file_path input/test_fill_in_the_blank.csv`

**STEP 4.** Play with the application.

`python src/flask_app.py`

### How to run on AWS using S3

**STEP 1.** Ssh into your EC2 instance. Note: I used a p2.xlarge. Replace words in brackets with your information.

`ssh -i aws_key.pem ubuntu@[INSTANCE_ID].[REGION].compute.amazonaws.com`

**STEP 2.** Follow "Set Up" Steps.

**STEP 3.** Follow "How to Run" steps. Use S3 for data. Here are example commands:

Create data set:
`python3 src/main.py --create_train_test_data s3://pujaa-rajan-enron-email-data/raw_email_data/enron_sample_1k.csv --output_file_path s3://pujaa-rajan-enron-email-data/model_output_data --input_file_path  s3://pujaa-rajan-enron-email-data/model_input_data`

Train:
`python3 src/main.py  --input_file_path s3://pujaa-rajan-enron-email-data/model_input_data --run_ngram_train --training_data_file_path s3://pujaa-rajan-enron-email-data/model_input_data/training_email_data.csv`

Test:
`python3 src/main.py  --output_file_path s3://pujaa-rajan-enron-email-data/model_output_data --run_ngram_test --testing_data_file_path s3://pujaa-rajan-enron-email-data/model_input_data/testing_email_data.csv`

Command line:
`python3 nlp_fib/src/main.py  --input_file_path s3://pujaa-rajan-enron-email-data/model_input_data --cli`

# Questions

Feel free to contact me at pujaa.rajan@gmail.comw with any questions, comments, or concerns.
