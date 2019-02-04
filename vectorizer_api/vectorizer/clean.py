import pandas as pd
import numpy as np
import re

def clean_str(string):
	# 2/2/19 Adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py

    string = re.sub(r"@\S+", "", string) # removes username handles
    string = re.sub(r"http\S+", "", string) # removes URL
    string = re.sub(r"http", "", string)
    string = re.sub(r"[^A-Za-z0-9]", " ", string) # removes special characters
    string = re.sub(r"\s{2,}", " ", string) # removes consecutive white spaces

    return string.strip().lower()

def remove_URL(string):
	string = string.replace(r"http\S+", "")
	string = string.replace(r"http", "")
	return string

def remove_special_characters(string):
	# string = string.replace(r"@\S+", "")
	string = string.replace(r"[^A-Za-z0-9 ]+", "")
	# string = string.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
	# string = string.replace(r"@", "at")
	return string

def lowercase(string):
	string = string.lower()
	return string
