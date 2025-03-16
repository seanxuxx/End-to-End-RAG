import argparse
import json
import logging
import os
import re
from typing import List, TypedDict

import torch
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.data.index import Index
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, GenerationConfig,
                          TextGenerationPipeline, pipeline)

from utils import get_chunk_max_length, set_logger

load_dotenv()

# Check if required api keys are written to the environment
for api_key in ['PINECONE_API_KEY', 'HF_TOKEN', 'LANGSMITH_API_KEY', 'LANGSMITH_TRACING']:
    if not os.getenv(api_key):
        raise ValueError(f"{api_key} is missing. Set it as an environment variable.")

# Login in Pinecone
pc = Pinecone()

# Set device
DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')


class Query(TypedDict):
    """
    Using TypedDict keeps track of input question, retrieved context, and generated answer,
    helping maintain a structured, type-safe, and modular workflow.
    The idea is adopted from Build a Retrieval Augmented Generation (RAG) App: Part 1 | 🦜️🔗 LangChain
    https://python.langchain.com/docs/tutorials/rag/
    """
    question: str
    context: List[str]
    answer: str


class DataStore():
    def __init__(self, model_name: str, data_dir: str,
                 chunk_size=1000, chunk_overlap=100,
                 filename_pattern='**/*.txt',
                 is_semantic_chunking=True, is_new_index=True):
        """
        Args:
            model_name (str): Name of HuggingFaceEmbeddings model.
            data_dir (str): Directory storing raw data files.
            chunk_size (int, optional): Defaults to 500.
            chunk_overlap (int, optional): Defaults to 100.
            filename_pattern (str, optional): "glob" parameter for DirectoryLoader. Defaults to '**/*.txt'.
            is_semantic_chunking (bool, optional):
                True for mainly using SemanticChunker and False for RecursiveCharacterTextSplitter.
                Defaults to True.
            is_new_index (bool, optional):
                Whether to load, chunk, and upsert documents to vector store.
                Defaults to True.
        """
        # Data config
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern

        # Chunking config
        self.is_semantic_chunking = is_semantic_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Index config
        self.index_name = re.sub(r'[^a-zA-Z0-9]', '-',
                                 f'{model_name}-{chunk_size}-{chunk_overlap}'.lower())

        # Embedding model config
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                                model_kwargs={'device': DEVICE})
        self.dimension = len(self.embeddings.embed_documents(['test'])[0])

        # Initialize Vector Store
        if is_new_index:  # Create new index and vector store and upsert documents
            logging.info(f"Create Pinecone index: {self.index_name}")
            self.vector_store = self.get_vector_store()
            chunks = self.chunk_documents()
            logging.info(f'Upserting {len(chunks)} chunks to Index "{self.index_name}"...')
            self.vector_store.add_documents(chunks)
        else:  # Load existing vector store
            logging.info(f"Use existing Pinecone index: {self.index_name}")
            pc_index = pc.Index(self.index_name)
            self.vector_store = PineconeVectorStore(index=pc_index, embedding=self.embeddings)

    def get_vector_store(self) -> PineconeVectorStore:
        """
            PineconeVectorStore
        """
        if self.index_name in pc.list_indexes().names():
            pc.delete_index(self.index_name)
        pc.create_index(name=self.index_name, dimension=self.dimension,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        pc_index = pc.Index(self.index_name)
        vector_store = PineconeVectorStore(index=pc_index, embedding=self.embeddings)
        return vector_store

    def chunk_documents(self) -> List[Document]:
        """
        Returns:
            list[Document]
        """
        # Load raw documents
        assert os.path.exists(self.data_dir), f"{self.data_dir} does not exist"
        loader = DirectoryLoader(self.data_dir, glob=self.filename_pattern,
                                 show_progress=True, use_multithreading=True)
        docs = loader.load()

        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        if self.is_semantic_chunking:  # SemanticChunker + RecursiveCharacterTextSplitter
            semantic_splitter = SemanticChunker(self.embeddings)
            semantic_chunks = [chunk for doc in tqdm(docs, desc='Chunk docs')
                               for chunk in semantic_splitter.split_documents([doc])]
            chunks = []
            for chunk in semantic_chunks:
                if len(chunk.page_content) <= self.chunk_size:
                    chunks.append(chunk)
                else:  # Sub-chunk documents with length larger than chunk_size
                    chunks.extend(text_splitter.split_documents([chunk]))
        else:  # Pure RecursiveCharacterTextSplitter
            chunks = text_splitter.split_documents(docs)

        # Add IDs for the documents
        result = []
        doc_text = set()
        for i, doc in tqdm(enumerate(chunks), total=len(chunks), desc='Add doc id'):
            if doc.page_content in doc_text:  # Deduplicate
                continue
            id_text = f"{i}_{doc.metadata['source']}"
            id_text = re.sub(r'[^\w]', '_', id_text).lower()
            doc.id = id_text
            doc_text.add(doc.page_content)
            result.append(doc)
        return result


class RetrivalLM():
    def __init__(self, data_store: DataStore,
                 search_type: str,
                 search_kwargs: dict,
                 task='text-generation',
                 model_name='mistralai/Mistral-7B-Instruct-v0.2'):
        """
        Args:
            data_store (DataStore): DataStore storing chunked documents.
            search_type (str): Type of search that the Retriever should perform.
                Defaults to 'similarity';
                Options: 'similarity', 'similarity_score_threshold', 'mmr'.
            search_kwargs (dict): Keyword arguments to pass to the search function.
                Defaults to {'k': 5} fo 'similarity';
                Example {'score_threshold': 0.5} for 'similarity_score_threshold';
                Example {'fetch_k': 20, 'lambda_mult': 0.5} for 'mmr'.
            task (str, optional): Transformer pipeline task.
                Defaults to 'text-generation'.
                Options: 'text-geneartion', 'text2text-generation'.
            model_name (str, optional): Model name for pipeline(). Should be consistent with the pipeline task.
        """
        # Retriever config
        self.retriever = data_store.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

        # LLM config
        torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=4096)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.task = task
        self.llm = pipeline(
            task=task,
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device=DEVICE
        )
        self.prompt_template = hub.pull("langchain-ai/retrieval-qa-chat")

    def qa(self, query: Query, **kwargs):
        """
        Retrive "context", generate "answer", and update them in the Query dictionary.

        Args:
            query (Query): Query dictionary with "question", "context", and "answer".
        **kwargs: for calling pipeline()
        """
        retrieved_docs = self.retriever.invoke(query['question'])
        query['context'] = [doc.page_content for doc in retrieved_docs]
        prompt = self.prompt_template.format(context='\n'.join(query['context']),
                                             input=query['question'])
        response = self.llm(prompt, **kwargs)
        query['answer'] = response[0]["generated_text"]  # type: ignore
        torch.cuda.empty_cache()
