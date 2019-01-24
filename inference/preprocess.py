def main():
	# Initialize logger
	LOG_FILENAME = 'case_automation.log'
	logging.basicConfig(filename=LOG_FILENAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

	pass

from nltk.tokenize import RegexpTokenizer

def tokenize(df, text_field):
	tokenizer = RegexpTokenizer(r'\w+')
	df["tokens"] = df["text"].apply(tokenizer.tokenize)
	return df

if __name__ == '__main__':
    main()
