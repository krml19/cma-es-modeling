import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.debug("elo")

# create a file handler
handler = logging.FileHandler('hello.log')
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(formatter)
consoleHandler.setLevel(logging.DEBUG)
logger.addHandler(consoleHandler)

logger.debug("elo")