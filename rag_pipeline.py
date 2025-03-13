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
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.data.index import Index
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, pipeline)

from utils import ParagraphTextSplitter, set_logger

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # DataStore paramters
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--chunker_name', type=str, default='character_chunker')
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--chunk_overlap', type=int, default=100)
    parser.add_argument('--similarity_score', type=str, default='cosine')
    # Retriver parameters
    parser.add_argument('--llm_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    return parser.parse_args()


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
    def __init__(self, model_name: str,
                 chunker_name: str, chunk_size=500, chunk_overlap=100,
                 dir_to_chunk='raw_data', dir_preformatted='',
                 filename_pattern='**/*.txt', similarity_score='cosine',
                 is_upsert_data=False):
        """
        Args:
            model_name (str): Name of HuggingFaceEmbeddings model.
            chunker_name (str): Chunking method.
                'character_chunker' for RecursiveCharacterTextSplitter;
                'semantic_chunker' for SemanticChunker;
                other string will raise an error.
            chunk_size (int, optional): Defaults to 500.
            chunk_overlap (int, optional): Defaults to 100.
            dir_to_chunk (str, optional):
                Directory storing raw data files. Defaults to 'raw_data'.
            dir_preformatted (str, optional):
                Directory storing pre-formatted data files. Defaults to ''.
            filename_pattern (str, optional):
                "glob" parameter for DirectoryLoader. Defaults to '**/*.txt'.
            similarity_score (str, optional):
                "metric" parameter for Pinecone Index. Defaults to 'cosine'.
            is_upsert_data (bool, optional):
                Whether to upsert documents to vector store. Defaults to False.
        """
        # Data config
        assert os.path.exists(dir_to_chunk), f"{dir_to_chunk} does not exist"
        if dir_preformatted:
            assert os.path.exists(dir_preformatted), f"{dir_preformatted} does not exist"
        self.dir_to_chunk = dir_to_chunk
        self.dir_preformatted = dir_preformatted
        self.filename_pattern = filename_pattern

        # Chunking config
        chunker_options = ['character_chunker', 'semantic_chunker']
        assert chunker_name in chunker_options, f"{chunker_name} is invalid chunker"
        self.chunker_name = chunker_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Index config
        index_name = f'{model_name}-{chunker_name}-{chunk_size}-{chunk_overlap}'.lower()
        self.index_name = re.sub(r'[^a-zA-Z0-9]', '-',
                                 index_name)  # Rename for Pinecone index name requirement
        self.similarity_score = similarity_score
        self.is_upsert_data = is_upsert_data

        # Embedding model config
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                                model_kwargs={'device': DEVICE})
        self.dimension = len(self.embeddings.embed_documents(['test'])[0])

        # Initialize Pinecone Index and Vector Store
        self.pc_index = self.get_pinecone_index()
        self.vector_store = PineconeVectorStore(index=self.pc_index, embedding=self.embeddings)
        self.retriever = self.vector_store.as_retriever()

        logging.info(f"Initialized Retriver model:\n{self.__dict__}\n")

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
            elif index_dict['metric'] != self.similarity_score:
                logging.info(f"Recreate Pinecone Index due to mismatch in metric")
                pc.delete_index(self.index_name)
                self.create_new_index()
        pc_index = pc.Index(self.index_name)
        return pc_index

    def create_new_index(self):
        pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.similarity_score,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        self.is_upsert_data = True  # Upsert data as long as creating new index

    def upsert_vector_store(self):
        """
        Update and insert chunked documents to vector store.

        Returns:
            list[Document]
        """
        # Load raw documents and split them into chunks
        loader = DirectoryLoader(self.dir_to_chunk,
                                 glob=self.filename_pattern,
                                 show_progress=True,
                                 use_multithreading=True)
        docs = loader.load()
        if self.chunker_name == 'semantic_chunker':
            text_splitter = SemanticChunker(self.embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                           chunk_overlap=self.chunk_overlap)
        chunks = [chunk for doc in tqdm(docs, desc='Chunk docs')
                  for chunk in text_splitter.split_documents([doc])]

        # Load pre-formatted documents and add to chunks
        if self.dir_preformatted:
            loader = DirectoryLoader(self.dir_preformatted,
                                     glob=self.filename_pattern,
                                     show_progress=True,
                                     use_multithreading=True)
            docs = loader.load()
            text_splitter = ParagraphTextSplitter()
            chunks.extend(text_splitter.split_documents(docs))

        # Add IDs for the documents
        for i, doc in tqdm(enumerate(chunks), total=len(chunks), desc='Add doc id'):
            id_text = f"{self.chunker_name}_{i}_{doc.metadata['source']}"
            id_text = re.sub(r'[^\w]', '_', id_text).lower()
            doc.id = id_text

        # Index documents
        logging.info(f"Upserting {len(chunks)} chunks to Index {self.index_name}...")
        self.vector_store.add_documents(chunks)
        logging.info("Done\n")

    def query_vector_store(self, query: Query):
        query['context'] = self.retriever.invoke(query['question'])


class RetrivalLLM():
    def __init__(self, model_name: str, data_store: DataStore):
        # Retriver config
        self.retriever = data_store.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        # LLM config
        torch.cuda.empty_cache()
        # model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        self.llm = pipeline(task="text-generation",
                            model=model_name,
                            tokenizer=tokenizer,
                            max_new_tokens=512,
                            torch_dtype=torch.bfloat16,
                            device=DEVICE)

        # Prompt config
        self.prompt_template = self.llm.tokenizer.apply_chat_template(
            PROMPT_IN_CHAT_FORMAT, tokenize=False, add_generation_prompt=True
        )

    def query_answer(self, query: Query):
        query['context'] = self.retriever.invoke(query['question'])
        context = ['\n'+doc.page_content for doc in query['context']]
        prompt = self.prompt_template.format(context=context, question=query['question'])
        response = self.llm(prompt, return_full_text=False)
        query['answer'] = response[0]["generated_text"].strip()  # type: ignore
        torch.cuda.empty_cache()


if __name__ == '__main__':

    set_logger('rag_pipeline')
    logging.info(f'Device: {DEVICE}')

    args = parse_args()

    data_store = DataStore(model_name=args.embedding_model, chunker_name=args.chunker_name,
                           chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                           similarity_score=args.similarity_score, is_upsert_data=False)
    rag_model = RetrivalLLM(model_name=args.llm_model, data_store=data_store)

    question = "When is the Vintage Pittsburgh retro fair taking place?"
    query = Query(question=question, context=[], answer="")
    rag_model.query_answer(query)

    logging.info(f"Question: {question}\nAnswer: {query['answer']}")
