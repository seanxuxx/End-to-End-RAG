import os
import re

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

load_dotenv()
pinecone_api_key = os.getenv('PINECONE_API_KEY')
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is missing. Set it as an environment variable.")
pc = Pinecone(api_key=pinecone_api_key)


DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')


class RetrieverModel():
    def __init__(self, model_name: str, chunker_name: str,
                 data_dir_to_chunk: str, data_dir_preformatted='',
                 filename_pattern='**/*.txt', similarity_score='cosine',
                 is_upsert_data=True, **kwargs):
        """
        Args:
            model_name (str): Name of HuggingFaceEmbeddings model
            chunker_name (str): _description_
                'semantic_chunker' for SemanticChunker;
                'character_chunker' for RecursiveCharacterTextSplitter;
                other string will raise an error.
            data_dir_to_chunk (str): Directory storing raw data files
            data_dir_preformatted (str, optional): Directory storing pre-formatted data files. Defaults to ''.
            filename_pattern (str, optional): "glob" for DirectoryLoader. Defaults to '**/*.txt'.
            similarity_score (str, optional): "metric" for Pinecone Index. Defaults to 'cosine'.
            is_upsert_data (bool, optional): Whether to upsert documents to vector store. Defaults to True.
        """

        # Data config
        assert os.path.exists(data_dir_to_chunk), f'{data_dir_to_chunk} does not exist'
        if data_dir_preformatted:
            assert os.path.exists(data_dir_preformatted), f'{data_dir_preformatted} does not exist'
        self.data_dir_to_chunk = data_dir_to_chunk
        self.data_dir_preformatted = data_dir_preformatted
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

        print('Loading embedding model...')
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                                model_kwargs={'device': DEVICE})
        self.dimension = len(self.embeddings.embed_documents(['test'])[0])
        self.pc_index = self.get_pinecone_index()
        self.vector_store = PineconeVectorStore(index=self.pc_index, embedding=self.embeddings)
        print('Initialized Retriver model\n')

        if is_upsert_data:
            self.upsert_vector_store()

    def get_pinecone_index(self) -> Index:
        """
        Set up Pinecone Index with mathing dimension and similarity score.

        Returns:
            Index
        """
        if self.index_name not in pc.list_indexes().names():
            print(f'Creating new Pinecone Index')
            self.create_new_index()
        else:
            if pc.Index(self.index_name).describe_index_stats().dimension != self.dimension:
                print(f'Recreating Pinecone Index due to mismatch in model dimension')
                pc.delete_index(self.index_name)
                self.create_new_index()
            elif pc.describe_index(self.index_name)['metric'] != self.similarity_score:
                print(f'Recreating Pinecone Index due to mismatch in metric')
                pc.delete_index(self.index_name)
                self.create_new_index()

        pc_index = pc.Index(self.index_name)
        print(f'Index "{self.index_name}"')
        print(pc.describe_index(self.index_name))
        print(pc_index.describe_index_stats())
        return pc_index

    def create_new_index(self):
        pc.create_index(
            name=self.index_name,
            dimension=self.dimension,
            metric=self.similarity_score,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    def upsert_vector_store(self):
        """
        Update and insert chunked documents to vector store.

        Returns:
            list[Document]
        """
        # Load raw documents and split them into chunks
        loader = DirectoryLoader(self.data_dir_to_chunk,
                                 glob=self.filename_pattern,
                                 show_progress=True,
                                 use_multithreading=True)
        docs = loader.load()
        print('Chunking documents...')
        if self.chunker_name == 'semantic_chunker':
            text_splitter = SemanticChunker(self.embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                           chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_documents(docs)

        # Load pre-formatted documents and add to chunks
        if self.data_dir_preformatted:
            loader = DirectoryLoader(self.data_dir_preformatted,
                                     glob=self.filename_pattern,
                                     show_progress=True,
                                     use_multithreading=True)
            docs = loader.load()
            chunks.extend(docs)

        # Add IDs for the documents
        for i, doc in tqdm(enumerate(chunks), total=len(chunks), desc='Adding doc id'):
            id_text = f"{self.chunker_name}_{i}_{doc.metadata['source']}"
            id_text = re.sub(r'[^\w]', '_', id_text).lower()
            doc.id = id_text

        # Index documents
        print(f'Upserting {len(chunks)} chunks to "{self.index_name}" index...')
        self.vector_store.add_documents(chunks)
        print('Done\n')


if __name__ == '__main__':

    retriver = RetrieverModel(model_name='all-mpnet-base-v2',
                              chunker_name='semantic_chunker',
                              data_dir_to_chunk='formatted_data')
