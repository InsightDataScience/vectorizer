Commands to remember

python3 nlp_fib/src/main.py --input_file_path nlp_fib/input --create_train_test_data nlp_fib/input/enron_sample_1000.csv --output_file_path nlp_fib/output

python nlp_fib/src/main.py  --output_file_path nlp_fib/output --run_ngram_train --training_data_file_path nlp_fib/input/training_email_data.csv 

python nlp_fib/src/main.py  --output_file_path nlp_fib/output --run_ngram_test --testing_data_file_path nlp_fib/input/test_fill_in_the_blank.csv 

ssh -i aws_key.pem ubuntu@ec2-54-214-100-221.us-west-2.compute.amazonaws.com

python3 nlp_fib/src/main.py  --output_file_path "/home/ubuntu/vectorizer/nlp_fib/output" --run_ngram_train --training_data_file_path "/home/ubuntu/vectorizer/nlp_fib/output/full_training_email_data.csv"
python3 nlp_fib/src/main.py  --output_file_path "/home/ubuntu/vectorizer/nlp_fib/output" --run_ngram_test --testing_data_file_path "/home/ubuntu/vectorizer/nlp_fib/output/"

To Do:
    
3. Write pickle file to s3
4. Add trigram and quadgram to bigram model
5. Run code on AWS
6. Perplexity
7. Use word vectors on output and use cosign similarity to calculate most similar words

Presentation:
2. Create data pipeline slide
3. Add validation chart and graphs
4. Include initial statistics about input data set
5. Give 5 examples of this being used in the professional space
6. Extensibility and Scalability
8. Add slide saying what you learned - google ngrams
