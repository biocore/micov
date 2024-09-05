import logging

def configure_logging():
    logger = logging.getLogger("micov")
    logger.setLevel(logging.INFO)
    # Check if the logger already has handlers (prevents adding multiple handlers)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger