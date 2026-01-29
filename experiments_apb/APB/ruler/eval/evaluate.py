
"""
Get summary.csv with score and null predictions amount.

Running
```
python evaluate.py \
    --data_dir /path/to/your/prediction_jsonl_folder \
    --benchmark synthetic
```
"""

import argparse
import json
import os
import re

import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

import importlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True, help='path to the prediction jsonl files')
parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
parser.add_argument("--verbose", type=int, default=0, help='how many lines you want to display.')
args = parser.parse_args()


def read_jsonl(filename, num_lines=-1):
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i == num_lines:
                break
    return lines


def write_jsonl(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')


def postprocess_pred(predict_str: str, task_config: dict):
    predict_str = predict_str.strip()

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def get_pred_and_ref(
    predictions_file: str,
    task_config: dict,
    input_context_field: str = 'input_context',
    input_query_field: str = 'input_query',
    references_field: str = 'outputs',
    prediction_field: str = 'pred',
    metadata_field: str = 'others',
):
    lines = read_jsonl(predictions_file)

    inputs = []
    predicts = []
    references = []
    indices = []

    for line in tqdm(lines):
        input = line[input_context_field] + line[input_query_field]
        predict = line[prediction_field]
        predict = postprocess_pred(predict, task_config)
        reference = line.get(references_field, [line.get('output', '')])
        index = line[metadata_field].get('id', line['index'])

        inputs.append(input)
        predicts.append(predict)
        references.append(reference)
        indices.append(index)

    return inputs, predicts, references, indices


def run_evaluation_per_task(task_config: dict, predictions_file: str, verbose: int = 0):
    inputs, predicts, references, indices = get_pred_and_ref(
        predictions_file=predictions_file,
        task_config=task_config,
    )

    task_nulls = f'{sum([len(x)==0 for x in predicts])}/{len(predicts)}'

    if len(references) > 0 and references[0][0] is not None:
        task_score = task_config['metric_fn'](predicts, references)
    else:
        task_score = 0.0

    if verbose != 0:
        print('=' * 40)
        for i, (input, reference, predict) in enumerate(zip(inputs, references, predicts)):
            print(f'Input     : {input}')
            print(f'Reference : {reference}')
            print(f'Prediction: {predict}')
            print('=' * 40)
            if i > verbose:
                break

    return task_score, task_nulls, predicts, indices


def write_evaluation(results: dict):
    tasks = list(results.keys())
    score = [results[task]['score'] for task in tasks]
    nulls = [results[task]['nulls'] for task in tasks]
    dfs = [
        ['Tasks'] + tasks,
        ['Score'] + score,
        ['Nulls'] + nulls,
    ]

    output_file = os.path.join(args.data_dir, 'summary.csv' if len(tasks) > 1 else f'summary-{tasks[0]}.csv')
    df = pd.DataFrame(dfs)
    df.to_csv(output_file, index=False)
    print('\n=============================================\n')
    print(df)
    print(f'\nSaved eval results to {output_file}')


def write_submission(results: dict):
    COLUMNS = ["Task", "ID", "Prediction"]
    dfs = pd.DataFrame(columns=COLUMNS, data=[])

    for task, result in results.items():
        df = pd.DataFrame({'Task': task, 'ID': result['indices'], 'Prediction': result['predicts']})
        dfs = pd.concat((dfs, df[COLUMNS]))

    output_file = os.path.join(args.data_dir, 'submission.csv')
    dfs = dfs.reset_index(drop=True)
    dfs.to_csv(output_file, index=False)
    print(f'\nSaved submission results to {output_file}')


def aggregate_chunk(folder):
    jsonl_files = [file for file in os.listdir(folder) if Path(file).suffix == '.jsonl']
    chunk_files = sorted([file for file in jsonl_files if re.match(r'.*[^_]+-\d+\.jsonl', file)])
    chunk_files_dict = defaultdict(list)
    for file in chunk_files:
        task = '-'.join(file.split('-')[:-1])
        chunk_files_dict[task].append(file)

    for task, files in chunk_files_dict.items():
        lines = []
        for file in sorted(files):
            file = os.path.join(folder, file)
            lines += read_jsonl(file)
            os.remove(file)  # Remove chunk files
        write_jsonl(lines, os.path.join(folder, f'{task}.jsonl'))


def main():
    curr_folder = os.path.dirname(os.path.abspath(__file__))

    try:
        module = importlib.import_module(f"{args.benchmark}.constants")
    except ImportError:
        print(f"Module eval.{args.benchmark}.constants not found.")

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}_task_config.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    TASKS = tasks_customized
    for _, config in TASKS.items():
        config.update(tasks_base[config['task']])

    print(f"Total tasks: {list(TASKS.keys())}")

    # Aggregate all prediction files
    aggregate_chunk(args.data_dir)

    # Get scores and nulls
    jsonl_files = [file for file in os.listdir(args.data_dir) if Path(file).suffix == '.jsonl']
    eval_results = {}
    subm_results = {}

    for task, config in TASKS.items():

        if f'{task}.jsonl' not in jsonl_files:
            print(f'Prediction file {task}.jsonl is not found.')
            continue

        print(f'Evaluate task {task}...')
        task_score, task_nulls, predicts, indices = run_evaluation_per_task(
            predictions_file=os.path.join(args.data_dir, f'{task}.jsonl'),
            task_config=config,
        )
        eval_results[task] = {
            'score': task_score,
            'nulls': task_nulls,
        }
        subm_results[task] = {
            'predicts': predicts,
            'indices': indices,
        }

    # Write to csv
    write_evaluation(eval_results)
    write_submission(subm_results)


if __name__ == '__main__':
    main()
