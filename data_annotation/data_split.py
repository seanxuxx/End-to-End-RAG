import argparse
import json
import os
import random
from typing import List


def write_data_files(data: List[str], output_dir: str, sub_dir: str):
    data_dir = os.path.join(output_dir, sub_dir)
    os.makedirs(data_dir, exist_ok=True)

    questions = []
    answers = []
    for i, qa_text in enumerate(data):
        lines = qa_text.split('\n')
        curr_q = lines[0].removeprefix('<Generated>Question: ').strip()
        curr_a_list = ' '.join(lines[1:]).split('Answer:')
        curr_a = '; '.join([a.strip() for a in curr_a_list if a.strip()])
        questions.append(curr_q)
        answers.append({f"{i+1}": curr_a})

    with open(os.path.join(data_dir, 'questions.txt'), 'w') as f:
        f.write('\n'.join(questions))

    with open(os.path.join(data_dir, 'reference_answers.json'), 'w') as f:
        json.dump(answers, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True)
    parser.add_argument('--output_folder', type=str, required=True)
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

    write_data_files(train, args.output_folder, 'train')
    write_data_files(test, args.output_folder, 'test')
    # os.path.dirname(os.getcwd())

    # os.path.dirname(__file__)

    # #write the data into json
    # output_folder = '/home/ubuntu/11711-anlp-spring2025-hw2/data'

    # train_content = [{'Question': x.split('\n')[0].removeprefix('<Generated>Question: ').strip(),
    #                   'Answer': x.split('\n')[1].removeprefix('Answer: ').strip()} for x in train]
    # test_content = [{'Question': x.split('\n')[0].removeprefix('<Generated>Question: ').strip(),
    #                 'Answer': x.split('\n')[1].removeprefix('Answer: ').strip()} for x in test]
    # folder = 'Annotation/train_testdata'
    # with open(os.path.join(folder, 'train.json'), 'w') as f:
    #     json.dump(train_content, f)
    # with open(os.path.join(folder, 'test.json'), 'w') as f:
    #     json.dump(test_content, f)

    # qualitycheck_test = random.sample(test_content, 50)

    # with open(os.path.join(folder, 'qualitycheck_test.json'), 'w') as f:
    #     json.dump(qualitycheck_test, f)
