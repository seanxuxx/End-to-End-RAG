from llama_index.core import SimpleDirectoryReader, ServiceContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import torch
from langchain_community.document_loaders import DirectoryLoader


def load_doc(file_path: str) -> 