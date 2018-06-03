import logging


class Logger:
    _logger = None

    def __init__(self, logger_log_level=logging.INFO, console_log_level=logging.DEBUG, file_log_level=logging.INFO, name=__name__):
        self._configure_logger(logger_log_level=logger_log_level, console_log_level=console_log_level, file_log_level=file_log_level, name=name)

    def _configure_logger(self, logger_log_level, console_log_level, file_log_level, name):
        self._logger = logging.getLogger(name=name)
        self._logger.setLevel(logger_log_level)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # create a file handler
        file_handler = logging.FileHandler('console.log')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)

        # add the handlers to the logger
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_log_level)
        self._logger.addHandler(console_handler)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)
