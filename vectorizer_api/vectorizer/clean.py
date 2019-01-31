import pandas as pd
import numpy as np

def remove_URL(text):
	text = text.replace(r"http\S+", "")
	text = text.replace(r"http", "")
	return text

def remove_special_characters(text):
	# text = text.replace(r"@\S+", "")
	text = text.replace(r"[^A-Za-z0-9 ]+", "")
	# text = text.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
	# text = text.replace(r"@", "at")
	return text

def lowercase(text):
	text = text.lower()
	return text
