import os
import re
from typing import List

import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.data.index import Index
from tqdm import tqdm

from utils import ParagraphTextSplitter, get_logger

load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing. Set it as an environment variable.")
pc = Pinecone(api_key=pinecone_api_key)


DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')


class RetrieverModel():
    def __init__(self, model_name: str, chunker_name: str,
                 dir_to_chunk: str, dir_preformatted='',
                 filename_pattern='**/*.txt', similarity_score='cosine',
                 is_upsert_data=True, **kwargs):
        """
        Args:
            model_name (str): Name of HuggingFaceEmbeddings model
            chunker_name (str): _description_
                'semantic_chunker' for SemanticChunker;
                'character_chunker' for RecursiveCharacterTextSplitter;
                other string will raise an error.
            dir_to_chunk (str): Directory storing raw data files
            dir_preformatted (str, optional): Directory storing pre-formatted data files. Defaults to ''.
            filename_pattern (str, optional): "glob" parameter for DirectoryLoader. Defaults to '**/*.txt'.
            similarity_score (str, optional): "metric" parameter for Pinecone Index. Defaults to 'cosine'.
            is_upsert_data (bool, optional): Whether to upsert documents to vector store. Defaults to True.
        """

        # Data config
        assert os.path.exists(dir_to_chunk), f'{dir_to_chunk} does not exist'
        if dir_preformatted:
            assert os.path.exists(dir_preformatted), f'{dir_preformatted} does not exist'
        self.dir_to_chunk = dir_to_chunk
        self.dir_preformatted = dir_preformatted
        self.filename_pattern = filename_pattern

        # Chunking config
        chunker_options = ['character_chunker', 'semantic_chunker']
        assert chunker_name in chunker_options, f'{chunker_name} is invalid chunker'
        self.model_name = model_name
        self.chunker_name = chunker_name
        self.chunk_size = kwargs['chunk_size'] if 'chunk_size' in kwargs else 1000
        self.chunk_overlap = kwargs['chunk_overlap'] if 'chunk_overlap' in kwargs else 100

        # Index config
        index_name = f'{model_name}_{chunker_name}'.lower()
        self.index_name = re.sub(r'[^a-zA-Z0-9]', '-',
                                 index_name)  # Rename for Pinecone index name requirement
        self.similarity_score = similarity_score
        self.is_upsert_data = is_upsert_data

        logger.info('Loading embedding model...')
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                                model_kwargs={'device': DEVICE})
        self.dimension = len(self.embeddings.embed_documents(['test'])[0])
        self.pc_index = self.get_pinecone_index()
        self.vector_store = PineconeVectorStore(index=self.pc_index, embedding=self.embeddings)
        logger.info('Initialized Retriver model\n')

        if self.is_upsert_data:
            self.upsert_vector_store()

    def get_pinecone_index(self) -> Index:
        """
        Set up Pinecone Index with mathing dimension and similarity score.

        Returns:
            Index
        """
        if self.index_name not in pc.list_indexes().names():
            logger.info(f'Creating new Pinecone Index')
            self.create_new_index()
        else:
            if pc.Index(self.index_name).describe_index_stats().dimension != self.dimension:
                logger.info(f'Recreating Pinecone Index due to mismatch in model dimension')
                pc.delete_index(self.index_name)
                self.create_new_index()
            elif pc.describe_index(self.index_name)['metric'] != self.similarity_score:
                logger.info(f'Recreating Pinecone Index due to mismatch in metric')
                pc.delete_index(self.index_name)
                self.create_new_index()

        pc_index = pc.Index(self.index_name)
        logger.info(f'Index "{self.index_name}"')
        logger.info(pc.describe_index(self.index_name))
        logger.info(pc_index.describe_index_stats())
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
        logger.info(f'Upserting {len(chunks)} chunks to "{self.index_name}" index...')
        self.vector_store.add_documents(chunks)
        logger.info('Done\n')


if __name__ == '__main__':
    logger = get_logger('rag_pipeline')
    logger.info(f'Device: {DEVICE}')
    retriver = RetrieverModel(model_name='all-mpnet-base-v2',
                              chunker_name='character_chunker',
                              dir_to_chunk='raw_data',
                              dir_preformatted='formatted_data',
                              chunk_size=400, chunk_overlap=100)
