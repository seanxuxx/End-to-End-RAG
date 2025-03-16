import argparse
import json
import logging
import os
import re
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


def format_variant_name(*args) -> str:
    variant_name_component = list(args)
    variant_name_component.append(datetime.now().strftime("%m%d%H%M"))
    variant_name = '-'.join([re.sub(r'[^\w]', '_', str(item))
                             for item in variant_name_component])
    return variant_name


def save_outputs(result: List[dict], configuration: dict, evaluate: bool,
                 output_dir: str, sub_dir: str):
    dir_path = os.path.join(output_dir, sub_dir)
    os.makedirs(dir_path, exist_ok=True)

    qa_filepath = os.path.join(dir_path, 'results.json')
    with open(qa_filepath, 'w') as f:
        json.dump(result, f, indent=2)
        logging.info(f'Save {len(result)} results to {qa_filepath}')

    config_filepath = os.path.join(dir_path, 'config.json')
    with open(config_filepath, 'w') as f:
        json.dump(configuration, f, indent=2)
        logging.info(f'Save model configuration to {config_filepath}')

    if evaluate:  # Evaluation
        eval_filepath = os.path.join(dir_path, 'evaluation.json')
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
    configuration = vars(args)
    variant_name = format_variant_name(
        args.embedding_model,
        args.chunk_size,
        args.chunk_overlap,
        args.search_type,
        args.generator_model
    )
    save_outputs(result, configuration, evaluate=not args.no_reference_answers,
                 output_dir=args.output_folder, sub_dir=variant_name)
