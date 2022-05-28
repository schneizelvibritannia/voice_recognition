import logging
import os
from datetime import date
from settings import LOG_ROOT, ERROR_LOGGING


class Logger:
    @staticmethod
    def get_logger():
        if not os.path.exists(LOG_ROOT):
            os.mkdir(LOG_ROOT)
        filename = LOG_ROOT + '/log_' + date.today(). \
            strftime("%Y%m%d") + '.log'
        error_logging = ERROR_LOGGING
        format_ = "%(asctime)s - %(pathname)s:%(funcName)s:%(lineno)d " \
                  "\n%(message)s"

        logging.basicConfig(filename=filename, format=format_, force=True, level=logging.DEBUG)
        if not error_logging:
            logging.disable(logging.CRITICAL)

        return logging
