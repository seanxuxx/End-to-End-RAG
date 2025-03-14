import argparse
import logging
import os
import re
from typing import List, TypedDict

import torch
from dotenv import load_dotenv
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
                          BitsAndBytesConfig, pipeline)

from utils import get_chunk_max_length, set_logger

load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing. Set it as an environment variable.")
pc = Pinecone(api_key=pinecone_api_key)


DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')


PROMPT_IN_CHAT_FORMAT = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]


PROMPT_IN_CHAT_FORMAT = """\
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
If the answer cannot be deduced from the context, do not give an answer.

Context:
{context}
---
Now answer this question: {question}
"""


class Query(TypedDict):
    """
    _summary_
    Using TypedDict keeps track of input question, retrieved context, and generated answer,
    helping maintain a structured, type-safe, and modular workflow.
    The idea is adopted from Build a Retrieval Augmented Generation (RAG) App: Part 1 | 🦜️🔗 LangChain
    https://python.langchain.com/docs/tutorials/rag/
    """
    question: str
    context: List[Document]
    answer: str


class DataStore():
    def __init__(self, model_name: str, chunk_size=500, chunk_overlap=100,
                 data_dir='raw_data', filename_pattern='**/*.txt',
                 is_semantic_chunking=False, is_upsert_data=False):
        """
        Args:
            model_name (str): Name of HuggingFaceEmbeddings model.
            chunk_size (int, optional): Defaults to 500.
            chunk_overlap (int, optional): Defaults to 100.
            data_dir (str): Directory storing raw data files. Defaults to 'raw_data'.
            filename_pattern (str, optional): "glob" parameter for DirectoryLoader. Defaults to '**/*.txt'.
            is_semantic_chunking (bool, optional):
                True for using SemanticChunker, otherwise RecursiveCharacterTextSplitter. Defaults to False.
            is_upsert_data (bool, optional):
                Whether to upsert documents to vector store. Defaults to False.
        """
        # Data config
        self.data_dir = data_dir
        self.filename_pattern = filename_pattern

        # Chunking config
        chunker_name = 'semantic' if is_semantic_chunking else 'recursive'
        self.is_semantic_chunking = is_semantic_chunking
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Index config
        index_name = f'{model_name}-{chunker_name}-{chunk_size}-{chunk_overlap}'.lower()
        self.index_name = re.sub(r'[^a-zA-Z0-9]', '-',
                                 index_name)  # Rename for Pinecone index name requirement
        self.is_upsert_data = is_upsert_data

        # Embedding model config
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                                model_kwargs={'device': DEVICE})
        self.dimension = len(self.embeddings.embed_documents(['test'])[0])

        # Initialize Pinecone Index and Vector Store
        self.pc_index = self.get_pinecone_index()
        self.vector_store = PineconeVectorStore(index=self.pc_index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()

        if self.is_upsert_data:
            self.upsert_vector_store()

    def get_pinecone_index(self) -> Index:
        """
        Set up Pinecone Index with mathing dimension and similarity score.

        Returns:
            Index
        """
        if self.index_name not in pc.list_indexes().names():
            logging.info(f"Create new Pinecone Index")
            self.create_new_index()
        else:
            index_dict = pc.describe_index(self.index_name)
            if index_dict['dimension'] != self.dimension:
                logging.info(f"Recreate Pinecone Index due to mismatch in model dimension")
                pc.delete_index(self.index_name)
                self.create_new_index()
        pc_index = pc.Index(self.index_name)
        return pc_index

    def create_new_index(self):
        pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        self.is_upsert_data = True  # Upsert data as long as creating new index

    def upsert_vector_store(self):
        """
        Update and insert chunked documents to vector store.

        Returns:
            list[Document]
        """
        # Load raw documents
        assert os.path.exists(self.data_dir), f"{self.data_dir} does not exist"
        loader = DirectoryLoader(self.data_dir, glob=self.filename_pattern,
                                 show_progress=True, use_multithreading=True)
        docs = loader.load()

        # Chunk documents
        if self.is_semantic_chunking:  # SemanticChunker + potentail RecursiveCharacterTextSplitter
            semantic_splitter = SemanticChunker(self.embeddings)
            semantic_chunks = [chunk for doc in tqdm(docs, desc='Chunk docs')
                               for chunk in semantic_splitter.split_documents([doc])]
            # Sub-chunk long documents where the document length falls in the upper outlier range
            max_length = min(get_chunk_max_length(semantic_chunks), 40000)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_length,
                                                           chunk_overlap=int(max_length*0.1))
            chunks = []
            for chunk in semantic_chunks:
                if len(chunk.page_content) < max_length:
                    chunks.append(chunk)
                else:  # Document length outliers
                    chunks.extend(text_splitter.split_documents([chunk]))
        else:  # Pure RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                           chunk_overlap=self.chunk_overlap)
            chunks = text_splitter.split_documents(docs)

        # Add IDs for the documents
        for i, doc in tqdm(enumerate(chunks), total=len(chunks), desc='Add doc id'):
            id_text = f"{i}_{doc.metadata['source']}"
            id_text = re.sub(r'[^\w]', '_', id_text).lower()
            doc.id = id_text

        # Index documents
        logging.info(f'Upserting {len(chunks)} chunks to Index "{self.index_name}"...')
        self.vector_store.add_documents(chunks)
        logging.info("Done")


class RetrivalLM():
    def __init__(self, task: str, model_name: str, data_store: DataStore, **kwargs):
        """
        Args:
            model_name (str): Model name for pipeline(). Should be available for AutoModelForCausalLM.
            data_store (DataStore): DataStore storing chunked documents.
        **kwargs: for as_retriever()
        """
        # Retriver config
        if kwargs:
            self.retriever = data_store.vector_store.as_retriever(**kwargs)
        else:
            self.retriever = data_store.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 5})

        # LLM config
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.task = task
        self.llm = pipeline(task=task,
                            model=model_name,
                            tokenizer=tokenizer,
                            max_length=20000,
                            truncation=True,
                            torch_dtype=torch.bfloat16,
                            device=DEVICE)

        # Prompt config
        # self.prompt_template = self.llm.tokenizer.apply_chat_template(
        #     PROMPT_IN_CHAT_FORMAT, tokenize=False, add_generation_prompt=True
        # )

    def qa(self, query: Query, **kwargs):
        """
        Retrive "context", generate "answer", and update them in the Query dictionary.

        Args:
            query (Query): Query dictionary with "question", "context", and "answer".
        **kwargs: for calling pipeline()
        """
        query['context'] = self.retriever.invoke(query['question'])
        context = ''.join(['\n'+doc.page_content for doc in query['context']])

        if self.task == 'text-generation':
            kwargs['return_full_text'] = False

        prompt = PROMPT_IN_CHAT_FORMAT.format(context=context,
                                              question=query['question'])
        response = self.llm(prompt, **kwargs)
        query['answer'] = response[0]["generated_text"].strip()  # type: ignore
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # DataStore paramters
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--chunk_overlap', type=int, default=100)
    parser.add_argument('--is_semantic_chunking', type=bool, default=False)
    # Retriver parameters
    parser.add_argument('--llm_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--task', type=str, default='text-generation')
    # Logging paramters
    parser.add_argument('--log_file_mode', type=str, default='a')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    set_logger('rag_pipeline', file_mode=args.log_file_mode)

    logging.info(f'Configuration:\n{vars(args)}')
    logging.info(f'Device: {DEVICE}')

    data_store = DataStore(model_name=args.embedding_model, data_dir='raw_data',
                           chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                           is_semantic_chunking=args.is_semantic_chunking)
    rag_model = RetrivalLM(task=args.task, model_name=args.llm_model,
                           data_store=data_store)

    question = 'What type of artworks can one explore at The Andy Warhol Museum in Pittsburgh?'
    query = Query(question=question, context=[], answer="")
    rag_model.qa(query)

    logging.info(f"\n{question}\n{query['answer']}\n")
