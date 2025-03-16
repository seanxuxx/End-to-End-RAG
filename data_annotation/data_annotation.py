import argparse
import json
import logging
import os
import random

import torch
from huggingface_hub import login
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import (ServiceContext, SimpleDirectoryReader,
                              VectorStoreIndex)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from tqdm import tqdm
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument('--topic_folder', type=str, help='Path to the topic folder',
                        default='Annotation/Annotated_rawdata/musicculture')
    parser.add_argument('--huggingfacetoken', type=str,
                        help='huggingface token to log in', required=True)
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument('--output_path', type=str, default="Annotation/Annotated_data")
    args = parser.parse_args()

    return args


def load_docs(folder_path, chunk_size=1024, chunk_overlap=100) -> list:
    '''
    folder_path: folder to annotate
    return: list of chunked content

    '''
    try:
        loader = DirectoryLoader(folder_path, glob='*.txt',
                                 show_progress=True, use_multithreading=True)
        documents = loader.load()
        if not documents:
            logging.warning(f"No documents found in {folder_path}.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = []
        for doc in documents:
            try:
                chunks.extend(text_splitter.split_documents([doc]))
            except Exception as e:
                logging.error(f"Error processing document: {doc.page_content[:100]} - {e}")

        return [chunk.page_content for chunk in chunks]

    except Exception as e:
        logging.error(f"Failed to load documents from {folder_path}: {e}")
        return []


def prompt_generation(examples, context, num_questions) -> str:
    prompt = "You are an AI assistant trained for data annotation.\n"
    prompt += "Your task is to generate **question-answer pairs** based on the given factual context. You will be given a context passage, and you will select a fact from the context, then ask a question from it, and then provide the answer to the asked question based on the selected fact. "
    prompt += "Ensure the questions are well-formed, unambiguous, and directly answerable using the provided context. Avoid speculative, open-ended questions, or generate any new context. "
    prompt += "These are some examples with questions and answers as well as the fact that help answer that question: \n"
    for i, example in enumerate(examples):
        prompt += f"Question: {example['question']}\n"
        prompt += f"Answer: {example['answer']}\n"
        prompt += "\n"
    prompt += f"Here is the context for the question and answer generation task: {context}\n\n"
    prompt += f"Extracted {num_questions} question-answer pairs based on the given context using the following format: \n"
    prompt += "Question: \n"
    prompt += "Answer: \n\n"
    prompt += "Only return the question and answer you generated. Do not include any additional information."
    return prompt


def result_formated(result) -> str:
    answers = result.split('Do not include any additional information.\n\n')[-1]
    answers = answers.replace('Question', '<Generated>Question')
    return answers


def main():
    args = parse_args()
    if not args.huggingfacetoken:
        logging.error("No Hugging Face token provided. Please provide a valid token.")
        exit(1)
    login(args.huggingfacetoken)
    try:
        with open('Annotation/example.json', 'r') as f:
            examples = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load example file: {e}")
        exit(1)
    chunks_content = load_docs(args.topic_folder)
    #sample context to generate qa pairs
    sample_size = 100 if args.topic_folder.split(
        '/')[-1] in ['generalinfo_cmu', 'generalinfo_pittsburgh', 'eventspittsburgh'] else 50
    context_sample = random.sample(chunks_content, min(sample_size, len(chunks_content)))
    augmented_data = ''
    torch.cuda.empty_cache()
    pipe = pipeline("text-generation", model=args.model_name, max_new_tokens=512,
                    torch_dtype=torch.bfloat16, device_map="cuda")
    for context in tqdm(context_sample, total=len(context_sample), desc="Generating QA pairs", mininterval=5):
        prompt = prompt_generation(examples, context, 1)
        results = pipe(prompt)
        answers = result_formated(results[0]['generated_text'])
        augmented_data += answers
        augmented_data += '\n\n'
    output_file = f"{args.output_path}{args.topic_folder.split('/')[-1]}.txt"
    with open(output_file, 'w') as f:
        f.write(augmented_data)
    logging.info(f"Augmented data saved successfully to {output_file}")


if __name__ == '__main__':
    main()
