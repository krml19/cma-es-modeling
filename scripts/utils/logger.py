import logging


class Logger:
    _logger = None

    def __init__(self, logger_log_level=logging.INFO, console_log_level=logging.DEBUG, file_log_level=logging.INFO):
        self._configure_logger(logger_log_level=logger_log_level, console_log_level=console_log_level, file_log_level=file_log_level)

    def _configure_logger(self, logger_log_level=logging.NOTSET, console_log_level=logging.DEBUG, file_log_level=logging.INFO):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logger_log_level)

        # create a file handler
        handler = logging.FileHandler('console.log')
        handler.setLevel(file_log_level)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self._logger.addHandler(handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_log_level)
        self._logger.addHandler(console_handler)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)
