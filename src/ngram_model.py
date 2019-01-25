# TODO: Need to edit/refactor train corpus file

# return: void
# arg:string,string,dict,dict,dict,dict,dict
# Used for testing the Language Model
def trainCorpus(train_file, test_file, bi_dict, tri_dict, quad_dict, vocab_dict, prob_dict):
    score = 0

    # load the training corpus for the dataset
    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    print("---Processing Time for Corpus Loading: %s seconds ---" % (time.time() - start_time))

    start_time1 = time.time()

    # estimate the lambdas for interpolation
    # found earlier usign estimate Function
    param = [0.7, 0.1, 0.1, 0.1]
    # param = estimateParameters(token_len, vocab_dict, bi_dict, tri_dict, quad_dict)
    # print(param)

    # create trigram Probability Dictionary
    findTrigramProbAdd1(vocab_dict, bi_dict, tri_dict, tri_prob_dict)
    # create bigram Probability Dictionary
    findBigramProbAdd1(vocab_dict, bi_dict, bi_prob_dict)
    # create quadgram Probability Dictionary
    findQuadgramProbAdd1(vocab_dict, bi_dict, tri_dict, quad_dict, quad_prob_dict)
    # sort the probability dictionaries
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    gc.collect()
    print("---Preprocessing Time for Creating Probable Word Dict: %s seconds ---" % (time.time() - start_time1))

    ### TESTING WITH TEST CORPUS
    test_data = ''
    # Now load the test corpus
    with open('test_corpus.txt', 'r') as file:
        test_data = file.read()

    # remove punctuations from the test data
    test_data = removePunctuations(test_data)
    test_token = test_data.split()

    # split the test data into 4 words list
    test_token = test_data.split()
    test_quadgrams = list(ngrams(test_token, 4))

    # choose most probable words for prediction
    start_time2 = time.time()
    score = computeTestScore(test_quadgrams, bi_dict, tri_dict, quad_dict, vocab_dict,
                             bi_prob_dict, tri_prob_dict, quad_prob_dict, token_len, param)
    print('Score:', score)
    print("---Processing Time for computing score: %s seconds ---" % (time.time() - start_time2))

    start_time3 = time.time()
    perplexity = computePerplexity(test_token, token_len, tri_dict, quad_dict, vocab_dict, prob_dict)
    print('Perplexity:', perplexity)
    print("---Processing Time for computing Perplexity: %s seconds ---" % (time.time() - start_time3))
