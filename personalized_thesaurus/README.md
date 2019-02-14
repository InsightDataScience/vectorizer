# Personalized Thesaurus

##

STEP 1. git clone https://github.com/InsightDataCommunity/vectorizer.git

STEP 2. cd into folder

STEP 3. python3 -m pip install --user virtualenv

STEP 4. source env/bin/activate

STEP 5. pip install -r requirements.txt 

INSERT PICTURE OF TERMINAL HERE


##Commands

### How to run locally

python3 nlp_fib/src/main.py --input_file_path nlp_fib/input --create_train_test_data input/enron_sample_1000.csv --output_file_path /output

python nlp_fib/src/main.py  --output_file_path nlp_fib/output --run_ngram_train --training_data_file_path input/training_email_data.csv 

python nlp_fib/src/main.py  --output_file_path nlp_fib/output --run_ngram_test --testing_data_file_path input/test_fill_in_the_blank.csv 

### How to run on AWS

ssh -i aws_key.pem ubuntu@ec2-54-214-100-221.us-west-2.compute.amazonaws.com
### Create data set
python3 src/main.py --create_train_test_data s3://pujaa-rajan-enron-email-data/raw_email_data/enron_sample_1k.csv --output_file_path s3://pujaa-rajan-enron-email-data/model_output_data --input_file_path  s3://pujaa-rajan-enron-email-data/model_input_data
### Training model
python3 src/main.py  --input_file_path s3://pujaa-rajan-enron-email-data/model_input_data --run_ngram_train --training_data_file_path s3://pujaa-rajan-enron-email-data/model_input_data/training_email_data.csv
### Testing model
python3 src/main.py  --output_file_path s3://pujaa-rajan-enron-email-data/model_output_data --run_ngram_test --testing_data_file_path s3://pujaa-rajan-enron-email-data/model_input_data/testing_email_data.csv 

###CLI
python3 nlp_fib/src/main.py  --input_file_path s3://pujaa-rajan-enron-email-data/model_input_data --cli
