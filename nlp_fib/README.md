# How to run locally

python3 nlp_fib/src/main.py --input_file_path nlp_fib/input --create_train_test_data nlp_fib/input/enron_sample_1000.csv --output_file_path nlp_fib/output

python nlp_fib/src/main.py  --output_file_path nlp_fib/output --run_ngram_train --training_data_file_path nlp_fib/input/training_email_data.csv 

python nlp_fib/src/main.py  --output_file_path nlp_fib/output --run_ngram_test --testing_data_file_path nlp_fib/input/test_fill_in_the_blank.csv 

# How to run on AWS

ssh -i aws_key.pem ubuntu@ec2-54-214-100-221.us-west-2.compute.amazonaws.com
## Create data set
python3 nlp_fib/src/main.py --create_train_test_data s3://pujaa-rajan-enron-email-data/raw_email_data/enron_sample_1k.csv --output_file_path s3://pujaa-rajan-enron-email-data/model_output_data --input_file_path  s3://pujaa-rajan-enron-email-data/model_input_data
## Training model
python3 nlp_fib/src/main.py  --input_file_path s3://pujaa-rajan-enron-email-data/model_input_data --run_ngram_train --training_data_file_path s3://pujaa-rajan-enron-email-data/model_input_data/training_email_data.csv
## Testing model
python3 nlp_fib/src/main.py  --output_file_path s3://pujaa-rajan-enron-email-data/model_output_data --run_ngram_test --testing_data_file_path s3://pujaa-rajan-enron-email-data/model_input_data/testing_email_data.csv 

#C:LI




AWS Run commands
ps -ef | grep python3

To Do:
    
1. Delete trigram
2. Use CLI for Demo
3. Get validation stats on full data set
4. Use word vectors on output and use cosign similarity to calculate most similar words
