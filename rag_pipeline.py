import argparse
import json
import logging
import os
import re
from typing import List, TypedDict

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from langchain import hub
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
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


def parse_datastore_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data_dir', type=str, default='raw_data',
                        help='Directory of raw text files')
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--chunk_overlap', type=int, default=100)
    parser.add_argument('--is_semantic_chunking', action='store_true', default=True)
    parser.add_argument('--not_semantic_chunking', action='store_false',
                        dest='is_semantic_chunking')


def parse_retriever_args(parser: argparse.ArgumentParser):
    parser.add_argument('--search_type', type=str, default='similarity',
                        choices=['similarity', 'similarity_score_threshold', 'mmr'])
    parser.add_argument('--search_k', type=int, default=5)
    parser.add_argument('--fetch_k', type=int, default=20)
    parser.add_argument('--lambda_mult', type=float, default=0.5)
    parser.add_argument('--score_threshold', type=float, default=0.5)


def parse_generator_args(parser: argparse.ArgumentParser):
    parser.add_argument('--task', type=str, default='text2text-generation',
                        choices=['text-generation', 'text2text-generation'])
    parser.add_argument('--generator_model', type=str, default='google/flan-t5-large')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--do_sample', action='store_true', default=True,
                        help="Enable do_sample when calling pipeline (default: True)")
    parser.add_argument('--not_do_sample', action='store_false', dest='do_sample',
                        help="Disable do_sample when calling pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parse_datastore_args(parser)
    parse_retriever_args(parser)
    parse_generator_args(parser)
    parser.add_argument('--log_file_mode', type=str, default='a')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    set_logger('rag_pipeline', file_mode=args.log_file_mode)
    logging.info(f'Configuration:\n{vars(args)}')

    # Set up DataStore
    data_store = DataStore(
        data_dir=args.data_dir,
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        is_semantic_chunking=args.is_semantic_chunking,
        is_new_index=False,
    )

    # Set up Retriver
    search_config = {'k': args.search_k}
    if args.search_type == 'mmr':
        search_config['fetch_k'] = args.fetch_k
        search_config['lambda_mult'] = args.lambda_mult
    elif args.search_type == 'similarity_score_threshold':
        search_config['score_threshold'] = args.score_threshold
    rag_model = RetrivalLM(
        data_store=data_store,
        search_type=args.search_type,
        search_kwargs=search_config,
        task=args.task,
        model_name=args.generator_model
    )

    # Set up generator
    generation_config = {
        'max_new_tokens': args.max_new_tokens,
        'do_sample': args.do_sample,
        'use_cache': True,
    }
    if args.do_sample:
        generation_config['temperature'] = args.temperature
        generation_config['top_p'] = args.top_p
        generation_config['repetition_penalty'] = args.repetition_penalty
    if args.task == 'text-generation':
        generation_config['return_full_text'] = False

    # Run RAG
    question = [
        "What kind of food is featured in the Lenten Fish Fry?",
        "What food is Picklesburgh centered around?",
        "What year was the Pittsburgh Penguins team founded?",
        "Who is the Bench Coach for the Pittsburgh Pirates?",
        "Which food festival is held at Stacks at 3 Crossings?",
    ]
    queries = []
    for question in tqdm(question):
        query = Query(question=question, context=[], answer="")
        rag_model.qa(query, **generation_config)
        queries.append(query)

    with open(f'scratch_{args.search_type}.json', 'w') as f:
        json.dump(queries, f, indent=4)
