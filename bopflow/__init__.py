__version__ = "0.0.11"


import logging


def get_logger(name, clear_logfile=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - Line: %(lineno)d of %(funcName)s -- %(message)s"
    )

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    file_path = f"{name}.log"
    ch = logging.FileHandler(file_path)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if clear_logfile:
        with open(file_path, "w"):
            pass

    return logger


LOGGER = get_logger("bopflow")
