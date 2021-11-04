import logging
from multiprocessing import Queue

def log_handler_fn(log_queue: Queue):
    logging.basicConfig()
    while True:
        log_record: logging.LogRecord = log_queue.get()
        if log_record is None: break
        logger = logging.getLogger(log_record.name)
        logger.handle(log_record)
