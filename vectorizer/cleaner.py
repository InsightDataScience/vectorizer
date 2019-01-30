import pandas as pd
import numpy as np

def remove_URL(tweet):
	tweet = tweet.replace(r"http\S+", "")
	tweet = tweet.replace(r"http", "")
	return tweet

def remove_special_characters(tweet):
	# tweet = tweet.replace(r"@\S+", "")
	tweet = tweet.replace(r"[^A-Za-z0-9 ]+", "")
	# tweet = tweet.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
	# tweet = tweet.replace(r"@", "at")
	return tweet

def lowercase(tweet):
	tweet = tweet.lower()
	return tweet
