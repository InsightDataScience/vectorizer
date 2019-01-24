import click
import pandas as pd
import numpy as np

def remove_URL(df, text_field):
	df[text_field] = df[text_field].str.replace(r"http\S+", "")
	df[text_field] = df[text_field].str.replace(r"http", "")
	return df

def remove_special_characters(df, text_field):
	df[text_field] = df[text_field].str.replace(r"@\S+", "")
	df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
	df[text_field] = df[text_field].str.replace(r"@", "at")
	return df

def lowercase(df, text_field):
	df[text_field] = df[text_field].str.lower()
	return df
