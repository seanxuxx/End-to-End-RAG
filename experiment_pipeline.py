import argparse
import json
import logging
import os
import random
import re
from datetime import datetime
from typing import List, Tuple, TypedDict

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

from evaluation_pipeline import *
from rag_pipeline import *
from utils import set_logger


def load_data(question_filepath: str, no_reference_answers: bool)-> tuple[list[str], dict]:
    # Load questions
    with open(question_filepath, 'r') as f:
        questions = [line.strip() for line in f.readlines()]
    logging.info(f'Load {len(questions)} questions from {question_filepath}')

    # Load reference answers if available
    if not no_reference_answers:
        reference_answer_file = os.path.join(os.path.dirname(question_filepath),
                                             'reference_answers.json')
        with open(reference_answer_file, 'r') as f:
            reference_answers = json.load(f)
    else:
        reference_answers = {}
    logging.info(f'Load {len(reference_answers)} reference answers from {reference_answer_file}')

    return questions, reference_answers


def convert_query_responses(queries: List[Query], reference_answers: dict) -> List[Dict]:
    result = [dict(query) for query in queries]
    if reference_answers:
        for i, q_dict in enumerate(result):  # Write reference answers to result dictionary
            q_dict['reference_answer'] = reference_answers[f'{i+1}']
    return result


def get_output_filepaths(embedding_model: str, generator_model: str, search_type: str) -> Tuple[str, str, str]:
    variant_name_component = [embedding_model, generator_model, search_type,
                              datetime.now().strftime("%m%d%H%M")]
    variant_name = '-'.join([str(item) for item in variant_name_component])
    variant_folder = os.path.join(args.output_folder, variant_name)
    os.makedirs(variant_folder, exist_ok=True)
    result_filepath = os.path.join(variant_folder, 'results.json')
    config_filepath = os.path.join(variant_folder, 'config.json')
    eval_filepath = os.path.join(variant_folder, 'evaluation.json')
    return result_filepath, config_filepath, eval_filepath


def save_outputs(result_filepath: str, config_filepath: str, eval_filepath: str):
    with open(result_filepath, 'w') as f:
        json.dump(result, f, indent=2)
        logging.info(f'Save {len(result)} results to {result_filepath}')

    with open(config_filepath, 'w') as f:
        json.dump(vars(args), f, indent=2)
        logging.info(f'Save model configuration to {config_filepath}')

    if eval_filepath:  # Evaluation
        model_outputs = [{'Question': item['question'],
                          'Answer': item['answer']} for item in result]
        annotated_data = [{'Question': item['question'],
                           'Answer': item['reference_answer']} for item in result]
        evaluation = QAEvaluator(model_outputs, annotated_data)
        evaluation.evaluate()
        evaluation.save_logs_to_json(eval_filepath)
        logging.info(f'Save evaluation metrics to {eval_filepath}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', type=str,
                        default='data/test/questions.txt')
    parser.add_argument('--output_folder', type=str,
                        default='system_outputs')
    parser.add_argument('--no_reference_answers', action='store_true', default=False,
                        help='Include this flag if only running on the final test set')
    parse_datastore_args(parser)
    parse_retriever_args(parser)
    parse_generator_args(parser)

    args = parser.parse_args()
    assert args.experiment_file.endswith('.txt'), '"--experiment_file" must be a txt file'
    return args


if __name__ == '__main__':

    args = parse_args()
    set_logger('rag_pipeline', file_mode='w')
    logging.info(f'Configuration:\n{vars(args)}')

    questions, reference_answers = load_data(args.experiment_file,
                                             args.no_reference_answers)

    # Set up DataStore
    data_store = DataStore(
        data_dir=args.data_dir,
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        is_semantic_chunking=args.is_semantic_chunking,
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
    queries = [Query(question=question, context=[], answer='') for question in questions]
    for query in tqdm(queries, desc='RAG Q&Aing'):
        rag_model.qa(query, **generation_config)
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    result = convert_query_responses(queries, reference_answers)

    result_filepath, config_filepath, eval_filepath = get_output_filepaths(
        args.embedding_model, args.generator_model, args.search_type)
    eval_filepath = '' if args.no_reference_answers else eval_filepath

    save_outputs(result_filepath, config_filepath, eval_filepath)
