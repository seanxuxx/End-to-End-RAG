import logging
from typing import List

from langchain.text_splitter import TextSplitter


def get_logger(log_name: str, file_mode='a') -> logging.Logger:
    logger = logging.getLogger('logging')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f'{log_name}.log',
                                       mode=file_mode)
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt=datefmt)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    return logger


class ParagraphTextSplitter(TextSplitter):
    """Custom text splitter to strictly split by paragraphs (\n) without merging."""

    def split_text(self, text: str) -> List[str]:
        return [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
