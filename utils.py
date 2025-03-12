import logging
from typing import List

from langchain.text_splitter import TextSplitter


def set_logger(log_name: str, file_mode='a'):

    file_handler = logging.FileHandler(f'{log_name}.log', mode=file_mode)  # Log to file
    file_fmt = '%(asctime)s [%(levelname)s] %(message)s'
    file_datefmt = '%Y-%m-%d %H:%M:%S'
    file_formatter = logging.Formatter(fmt=file_fmt, datefmt=file_datefmt)
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()  # Log to console
    console_fmt = '%(message)s'
    console_formatter = logging.Formatter(fmt=console_fmt)
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(level=logging.INFO,
                        handlers=[file_handler, console_handler])


class ParagraphTextSplitter(TextSplitter):
    """Custom text splitter to strictly split by paragraphs (\n) without merging."""

    def split_text(self, text: str) -> List[str]:
        return [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
