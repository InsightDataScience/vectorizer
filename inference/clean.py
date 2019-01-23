import pandas as pd
import numpy as np

def main():
	# Initialize logger
	LOG_FILENAME = 'case_automation.log'
	logging.basicConfig(filename=LOG_FILENAME,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

	pass

if __name__ == '__main__':
    main()
