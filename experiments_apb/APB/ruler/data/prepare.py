
"""
Prepare jsonl with field `input` and `outputs`.
{
    "index" int,
    "input": str,
    "outputs": [str],
}

Usage:
python ruler/data/prepare.py \
    --save_dir <path_to_save_generated_dataset> \
    --task <task_name> \
    --tokenizer_path <path_to_tokenizer> \
    --tokenizer_type hf \
    --max_seq_length <dataset_sequence_length> \
    --model_template_type <prompt_template> \
    --num_samples <number_of_samples_to_generate> \
"""
import argparse
import os
import subprocess
import time
from pathlib import Path

import yaml

from synthetic.constants import TASKS
from template import PROMPT_TEMPLATES

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument('--task', type=str, required=True, help='tasks in benchmark')
parser.add_argument('--tokenizer_path', type=str, required=True, help='path to the tokenizer model')
parser.add_argument('--tokenizer_type', type=str, default='nemo', help='[Options] nemo, hf, openai.')
parser.add_argument(
    '--max_seq_length',
    type=int,
    required=True,
    help='max sequence length including all input tokens and generated tokens.',
)
parser.add_argument('--num_samples', type=int, default=500, help='maximum number of samples we want to test')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--model_template_type', type=str, default='base', help='Options in `template.py`')
parser.add_argument('--remove_newline_tab', action='store_true', help='remove `\n` and `\t` in all strings.')
parser.add_argument('--chunk_idx', type=int, default=0, help='index of current split chunk')
parser.add_argument('--chunk_amount', type=int, default=1, help='size of split chunk')

args = parser.parse_args()


def main():
    start_time = time.time()

    with open(os.path.join(os.path.dirname(CURRENT_DIR), 'synthetic_task_config.yaml')) as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f'{args.task} is not found in synthetic_task_config.yaml')

    config = tasks_customized.get(args.task)
    config.update(TASKS[config['task']])

    # Add templates
    assert args.model_template_type in PROMPT_TEMPLATES, print(
        f'{args.model_template_type} is not found in {PROMPT_TEMPLATES.keys()}'
    )
    model_template = PROMPT_TEMPLATES[args.model_template_type]['template']

    # Add answer prefix for all models
    answer_prefix = config['answer_prefix'] if 'answer_prefix' in config else ''
    model_template_context, model_template_query = model_template.split('{task_template}')
    config['context_template'] = model_template_context + config['context_template']
    config['query_template'] = config['query_template'] + model_template_query + answer_prefix

    # Split task into multiple chunks
    chunks = [
        (args.num_samples // args.chunk_amount) + (1 if i < args.num_samples % args.chunk_amount else 0)
        for i in range(args.chunk_amount)
    ]
    num_samples = chunks[args.chunk_idx]
    pre_samples = sum(chunks[: args.chunk_idx])

    random_seed = 42 + args.chunk_idx

    try:
        script = os.path.join(CURRENT_DIR, 'synthetic', f"{config['task']}.py")
        additional_args = " ".join([f"--{k} {v}" for k, v in config['args'].items()])
        command = f"""python {script} \
        --save_dir  {args.save_dir} \
        --save_name {args.task} \
        --subset validation \
        --tokenizer_path {args.tokenizer_path} \
        --tokenizer_type {args.tokenizer_type} \
        --max_seq_length {args.max_seq_length} \
        --tokens_to_generate {config['tokens_to_generate']} \
        --num_samples {num_samples} \
        --random_seed {random_seed} \
        {additional_args} \
        {f"--remove_newline_tab" if args.remove_newline_tab else ""} \
        {f"--pre_samples {pre_samples}" if config['task'] == 'qa' else ""} \
        --context_template "{config['context_template']}" \
        --query_template "{config['query_template']}"
        """
        print(command)
        result = subprocess.run(
            command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        if result.returncode == 0:
            print("Output:")
            print(result.stdout)
        else:
            print("Error:")
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error output:", e.stderr)

    save_file = args.save_dir / args.task / 'validation.jsonl'
    print(f"Prepare {args.task} with lines: {args.num_samples} to {save_file}")
    print(f"Used time: {round((time.time() - start_time) / 60, 1)} minutes")


if __name__ == '__main__':
    main()
