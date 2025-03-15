import logging
from typing import List

import numpy as np
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document


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


def get_chunk_max_length(chunks: List[Document]) -> int:
    """
    Find the upper bound of the length of the chunks to filter too long chunks.

    Args:
        chunks (List[Document]): A list semantic chunks.

    Returns:
        int: Upper bound of the non-outliers as the max length to filter too long chunks.
    """
    lengths = np.array([len(chunk.page_content) for chunk in chunks])
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    upper_bound = q3 + 1.5 * (q3 - q1)
    return (int(upper_bound // 100) + 1) * 100


class ParagraphTextSplitter(TextSplitter):
    """Custom text splitter to strictly split by paragraphs (\n) without merging."""

    def split_text(self, text: str) -> List[str]:
        return [chunk.strip() for chunk in text.split("\n") if chunk.strip()]
