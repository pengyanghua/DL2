import logging

# create logger
def getLogger(name='logger', level='INFO', mode='w', fh=True, ch=True, prefix=""):

	logger = logging.getLogger(name)

	fh = logging.FileHandler(name + '.log', mode)

	ch = logging.StreamHandler()

	if level == "INFO":
		logger.setLevel(logging.INFO)
		fh.setLevel(logging.INFO)
		ch.setLevel(logging.INFO)
	elif level == "DEBUG":
		logger.setLevel(logging.DEBUG)
		fh.setLevel(logging.DEBUG)
		ch.setLevel(logging.DEBUG)
	elif level == "ERROR":
		logger.setLevel(logging.ERROR)
		fh.setLevel(logging.ERROR)
		ch.setLevel(logging.ERROR)

	#formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
	formatter = logging.Formatter(prefix+' '+'%(filename)s:%(lineno)s %(levelname)s: %(message)s')
	fh.setFormatter(formatter)
	ch.setFormatter(formatter)

	if fh:
		logger.addHandler(fh)
	if ch:
		logger.addHandler(ch)

	return logger
