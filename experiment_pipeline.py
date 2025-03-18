import argparse
import json
import logging
import os
import re
import time
from ast import arg
from datetime import datetime
from typing import List

import torch
from tqdm import tqdm

from evaluation_pipeline import *
from rag_pipeline import *
from utils import set_logger


def load_data(question_filepath: str, no_reference_answers: bool)-> tuple[list[str], dict]:
    # Load questions
    with open(question_filepath, 'r') as f:
        questions = [line.strip() for line in f.readlines()]
    logging.info(f'Load {len(questions)} questions from: {question_filepath}')

    # Load reference answers if available
    if not no_reference_answers:
        reference_answer_file = os.path.join(os.path.dirname(question_filepath),
                                             'reference_answers.json')
        with open(reference_answer_file, 'r') as f:
            reference_answers = json.load(f)
        logging.info(
            f'Load {len(reference_answers)} reference answers from: {reference_answer_file}')
    else:
        reference_answers = {}

    return questions, reference_answers


def convert_query_responses(queries: List[Query], reference_answers: dict) -> List[Dict]:
    result = [dict(query) for query in queries]
    if reference_answers:
        for i, q_dict in enumerate(result):  # Write reference answers to result dictionary
            q_dict['reference_answer'] = reference_answers[f'{i+1}']
    return result


def format_variant_name(*args) -> str:
    variant_name_component = []
    for item in args:
        s = str(item).split('/')[-1]  # Remove potential path
        s = re.sub(r'[^\w]', '_', s)  # Remove special characters
        variant_name_component.append(s)
    variant_name_component.append(datetime.now().strftime("%m%d%H%M"))
    variant_name = '-'.join(variant_name_component)
    return variant_name


def save_outputs(result: List[dict], configuration: dict, evaluate: bool,
                 submit: bool, output_dir: str, sub_dir: str):
    dir_path = os.path.join(output_dir, sub_dir)
    os.makedirs(dir_path, exist_ok=True)

    qa_filepath = os.path.join(dir_path, 'results.json')
    if submit:
        submission = {}
        for i in range(len(result)):
            curr_a = result[i]['answer']
            submission[f"{i+1}"] = curr_a
        with open(qa_filepath, 'w') as f:
            json.dump(submission, f, indent=2)
            logging.info(f'Save {len(result)} results to: {qa_filepath}')
    else:
        with open(qa_filepath, 'w') as f:
            json.dump(result, f, indent=2)
            logging.info(f'Save {len(result)} results to: {qa_filepath}')

    config_filepath = os.path.join(dir_path, 'config.json')
    with open(config_filepath, 'w') as f:
        json.dump(configuration, f, indent=2)
        logging.info(f'Save model configuration to: {config_filepath}')

    if evaluate:  # Evaluation
        eval_filepath = os.path.join(dir_path, 'evaluation.json')
        model_outputs = [{'Question': item['question'],
                          'Answer': item['answer']} for item in result]
        annotated_data = [{'Question': item['question'],
                           'Answer': item['reference_answer']} for item in result]
        evaluation = QAEvaluator(model_outputs, annotated_data)
        evaluation.evaluate()
        evaluation.save_logs_to_json(eval_filepath)
        logging.info(f'Save evaluation metrics to: {eval_filepath}')


def parse_datastore_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data_dir', type=str, default='raw_data',
                        help='Relative filepath of the raw text data directory')
    parser.add_argument('--embedding_model', type=str, default='all-mpnet-base-v2',
                        help='sentence-transformers model for embeddings')
    parser.add_argument('--chunk_size', type=int, default=1000)
    parser.add_argument('--chunk_overlap', type=int, default=100)
    parser.add_argument('--no_semantic_chunk', action='store_true', default=False,
                        help='Include this flag to enable RecursiveCharacterTextSplitter only')


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
    parser.add_argument('--generator_model', type=str, default='google/flan-t5-large',
                        help='transformer model supporting the specified task')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--repetition_penalty', type=float, default=1.2)
    parser.add_argument('--do_sample', action='store_true', default=True)
    parser.add_argument('--not_do_sample', action='store_false', dest='do_sample',
                        help='Include this flag to disable do_sampling for pipeline generation')
    parser.add_argument('--few_shot', action='store_false', default=False)
    parser.add_argument('--add_few_shot', action='store_true', dest='few_shot',
                        help='Include this flag to enable few-shot learning')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', type=str, default='data/test/questions.txt',
                        help='Relative filepath of the txt file for inference')
    parser.add_argument('--output_folder', type=str, default='system_outputs')
    parser.add_argument('--output_name', type=str, default='',
                        help='Name of a subfolder under the output_folder to store result files')
    parser.add_argument('--no_reference_answers', action='store_true', default=False,
                        help='Include this flag if only running on the final test set')
    parse_datastore_args(parser)
    parse_retriever_args(parser)
    parse_generator_args(parser)
    return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    args = parse_args()

    if args.output_name:
        variant_name = args.output_name
    else:
        variant_name = format_variant_name(
            args.embedding_model,
            args.chunk_size,
            args.chunk_overlap,
            args.search_type,
            args.generator_model
        )

    set_logger('experiments', file_mode='a')
    logging.info(f'Configuration:\n{vars(args)}')

    # Load dataset
    questions, reference_answers = load_data(args.experiment_file,
                                             args.no_reference_answers)

    # Set up DataStore
    data_store = DataStore(
        data_dir=args.data_dir,
        model_name=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        no_semantic_chunk=args.no_semantic_chunk,
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
        model_name=args.generator_model,
        few_shot=args.few_shot,
        training_path='data/train'
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

    setup_time = time.time()
    logging.info(f'Configure time elapsed: {setup_time - start_time:.2f}s')

    # Run RAG
    queries = [Query(question=question, context=[], answer='') for question in questions]
    for query in tqdm(queries, desc='RAG Q&Aing'):
        rag_model.qa(query, **generation_config)
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    infer_time = time.time()
    logging.info(f'Inference time elapsed: {infer_time - setup_time:.2f}s')

    # Save result
    result = convert_query_responses(queries, reference_answers)
    configuration = vars(args)
    save_outputs(result, configuration, evaluate=not args.no_reference_answers,
                 submit=args.no_reference_answers, output_dir=args.output_folder, sub_dir=variant_name)

    logging.info('\n')  # Done!!!
