import json
from rag_pipeline import *
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
                          BitsAndBytesConfig, GenerationConfig,
                          TextGenerationPipeline, pipeline)
from utils import get_chunk_max_length, set_logger

# Experiment hyperparameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_type', type=str, help='train/test')
    parser.add_argument('experiment_folder',type=str, default='Annotation/train_testdata')
    parser.add_argument('result_folder',type=str, default='Annotation/experiment_result')
    # Retriver parameters
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2')
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--chunk_overlap', type=int, default=100)
    parser.add_argument('--is_semantic_chunking', type=bool, default=False)
    parser.add_argument('--search_type', type=str, default='similarity')
    parser.add_argument('--search_num',type=int, default=3)
    # LLM parameters
    parser.add_argument('--llm_model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('--task', type=str, default='text-generation')
    parser.add_argument('--max_new_token_length', type=int, default=100)
    parser.add_argument('--temperature',type=float,default=0.01)
    parser.add_argument('--top_p',type=float,default=0.95)
    parser.add_argument('--repetition_penalty',type=float,default=1.2)

    return parser.parse_args()


if __name__ == '__main__':
    args =parse_args()
    search_config = {'k': args.search_num}
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_token_length,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Set up models
    data_store = DataStore(model_name=args.embedding_model, data_dir=args.data_dir,
                            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                            is_semantic_chunking=args.is_semantic_chunking)
    rag_model = RetrivalLM(data_store=data_store,
                            search_type=args.search_type,
                            search_kwargs=search_config,
                            task=args.task, model_name=args.llm_model)

    # Run RAG
    with open(os.path.join(args.experiment_folder,args.experiment_type+'.json'),'r') as f:
        content = json.load(f)
    questions = [item['Question'] for item in content]
    reference_answer = [item['Answer'] for item in content]
    generated_answer = []
    for question in questions:
        query = Query(question=question, context=[], answer="")
        rag_model.qa(query, **generation_config.to_dict())
        generated_answer.append(query['answer'])
    result = [{'Question': questions[i],'<Generated>Answer': generated_answer[i],'<Reference>Answer':reference_answer[i]} for i in range(len(questions))]
    with open(os.path.join(args.result_folder,args.experiment_type+'.json'),'w') as f:
        json.dump(result,f)