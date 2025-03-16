import argparse
import json
import os
import random
from typing import List


def write_data_files(data: List[str], output_dir: str, sub_dir: str):
    data_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(data_dir, exist_ok=True)

    questions = []
    answers = {}
    for i, qa_text in enumerate(data):
        lines = qa_text.split('\n')
        curr_q = lines[0].removeprefix('<Generated>Question: ').strip()
        curr_a_list = ' '.join(lines[1:]).split('Answer:')
        curr_a = '; '.join([a.strip() for a in curr_a_list if a.strip()])
        questions.append(curr_q)
        answers[f"{i+1}"] = curr_a

    question_filepath = os.path.join(data_dir, 'questions.txt')
    answer_filepath = os.path.join(data_dir, 'reference_answers.json')

    with open(question_filepath, 'w') as f:
        f.write('\n'.join(questions))
        print(f"{sub_dir} questions written to {question_filepath}")

    with open(answer_filepath, 'w') as f:
        json.dump(answers, f, indent=4)
        print(f"{sub_dir} answers written to {answer_filepath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    args = parser.parse_args()

    os.chdir(os.path.dirname(__file__))

    train = []
    test = []
    for file in os.listdir(args.input_folder):
        if file.endswith('txt'):
            # Every file in the folder is a different topic, thus ensureing the diversity of the question
            with open(os.path.join(args.input_folder, file), 'r') as f:
                content = f.read()
                content_list = content.strip().split('\n\n')
                train_list = random.sample(content_list, 10)
                test_list = [x for x in content_list if x not in train_list]
                train.extend(train_list)
                test.extend(test_list)

    output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    write_data_files(train, output_folder, 'train')
    write_data_files(test, output_folder, 'test')
