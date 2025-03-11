import os
from dotenv import load_dotenv
import torch
from langchain_community.document_loaders import DirectoryLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.data.index import Index

DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if torch.backends.mps.is_available() else 'cpu')

load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing. Set it as an environment variable.")
pc = Pinecone(api_key=PINECONE_API_KEY)


TEXT_SPLITTERS_OPTIONS = ['character_chunker', 'semantic_chunker']


def chunk_documents(embeddings: HuggingFaceEmbeddings, chunker: str,
                    raw_data_dir: str, formatted_data_dir='',
                    filename_pattern='**/*.txt', **kwargs) -> list:
    """
    Load documents and split them into chunks.

    Args:
        embeddings (HuggingFaceEmbeddings)
        chunker (str):
            'semantic_chunker' for SemanticChunker;
            'character_chunker' for RecursiveCharacterTextSplitter;
            other string will raise an error.
        raw_data_dir (str): Directory storing raw data files.
        formatted_data_dir (str, optional): Directory storing pre-formatted data files. Defaults to ''.
        filename_pattern (str, optional): "glob" for DirectoryLoader. Defaults to '**/*.txt'.

    Returns:
        list[langchain_core.documents.base.Document]
    """
    assert chunker in TEXT_SPLITTERS_OPTIONS, f'{chunker} is invalid chunker'

    # Load raw documents and split them into chunks
    loader = DirectoryLoader(raw_data_dir, glob=filename_pattern,
                             show_progress=True, use_multithreading=True)
    docs = loader.load()

    print('Chunking documents...')
    if chunker == 'semantic_chunker':
        text_splitter = SemanticChunker(embeddings)
    else:
        text_splitter = RecursiveCharacterTextSplitter(**kwargs)
    chunks = text_splitter.split_documents(docs)

    # Load pre-formatted documents and add to chunks
    if formatted_data_dir:
        loader = DirectoryLoader(raw_data_dir, glob=filename_pattern,
                                 show_progress=True, use_multithreading=True)
        docs = loader.load()
        chunks.extend(docs)

    return chunks


def get_pinecone_index(index_name: str, dimension: int,
                       similarity_score='cosine') -> Index:
    """
    Set up Pinecone Index with mathing dimension and similarity score.

    Args:
        index_name (str): Name of Pinecone Index.
        dimension (int): Dimension of the embedding model.
        similarity_score (str, optional): "metric" for Pinecone Index.. Defaults to 'cosine'.

    Returns:
        Index
    """
    if index_name not in pc.list_indexes().names():
        print(f'Create new Pinecone Index')
        index = create_new_index(index_name, dimension, similarity_score)
    else:
        index = pc.Index(index_name)
        if (index.describe_index_stats().dimension != dimension or
                pc.describe_index(index_name)['metric'] != similarity_score):
            print(f'Recreate Pinecone Index due to mismatch in model dimension or similarity score')
            pc.delete_index(index_name)
            index = create_new_index(index_name, dimension, similarity_score)
    if index.describe_index_stats().total_vector_count > 0:
        index.delete(delete_all=True)
        print(f'Empty "{index_name}" index')
    print(f'Describe "{index_name}" index')
    print(pc.describe_index(index_name))
    return index


def create_new_index(index_name: str, dimension: int, similarity_score='cosine') -> Index:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=similarity_score,
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    return pc.Index(index_name)


def build_datastore(model_name: str, index_name: str,
                    chunker: str, raw_data_dir: str,
                    formatted_data_dir='', filename_pattern='**/*.txt',
                    similarity_score='cosine') -> PineconeVectorStore:
    """
    Args:
        model_name (str): Name of HuggingFaceEmbeddings model.
        index_name (str): Name of Pinecone Index.
        chunker (str):
            'semantic_chunker' for SemanticChunker;
            'character_chunker' for RecursiveCharacterTextSplitter;
            other string will raise an error.
        raw_data_dir (str): Directory storing raw data files.
        formatted_data_dir (str, optional): Directory storing pre-formatted data files. Defaults to ''.
        filename_pattern (str, optional): "glob" for DirectoryLoader. Defaults to '**/*.txt'.
        similarity_score (str, optional): "metric" for Pinecone Index. Defaults to 'cosine'.

    Returns:
        PineconeVectorStore
    """
    print('Loading embeddings model...')
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                       model_kwargs={'device': DEVICE})
    dimension = len(embeddings.embed_documents(['test'])[0])
    chunks = chunk_documents(embeddings, chunker,
                             raw_data_dir, formatted_data_dir, filename_pattern)
    index = get_pinecone_index(index_name, dimension, similarity_score)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    print(f'Storing {len(chunks)} chunks into "{index_name}" index...')
    _ = vector_store.add_documents(chunks)  # Index chunks
    return vector_store


if __name__ == '__main__':
    build_datastore(model_name='all-mpnet-base-v2',
                    index_name='11711-hw2',
                    chunker='semantic_chunker',
                    raw_data_dir='formatted_data')
