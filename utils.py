import logging

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
        logging.Formatter('%(asctime)s:%(levelname)s:%(message)s'))
        logger.addHandler(file_handler)
