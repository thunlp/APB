
import argparse
import os, yaml
from typing import List, Optional

from ruler import PROMPT_TEMPLATES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    'niah_single_1',
    'niah_single_2',
    'niah_single_3',
    'niah_multikey_1',
    'niah_multikey_2',
    'niah_multikey_3',
    'niah_multivalue',
    'niah_multiquery',
    'vt',
    'cwe',
    'fwe',
    'qa_1',
    'qa_2',
]

def define_cmd_arguments():
    parser = argparse.ArgumentParser()

    # Model Parameters
    parser.add_argument('-n', '--model_name', required=True, help='experiment name prefix')
    parser.add_argument('-p', '--model_path', required=True, help='model path')
    parser.add_argument(
        '-pc',
        '--prompt_config',
        required=True,
        choices=PROMPT_TEMPLATES.keys(),
        help='prompt template config name. options from `ruler/data/template.py`',
    )

    # Attention Configuration
    parser.add_argument('-a', '--attn_type', default='star', help='attention type')
    parser.add_argument('-bs', '--block_size', type=int, default=-1, help='context block size')
    parser.add_argument('-as', '--anchor_block_size', type=int, default=-1, help='anchor block size')

    # Sequence Lengths and Tasks
    parser.add_argument(
        '-l',
        '--seq_lengths',
        type=int,
        required=True,
        nargs='+',
        help='sequence lengths',
    )
    parser.add_argument('-t', '--tasks', default=TASKS, nargs='+', choices=TASKS, help='tasks')
    parser.add_argument(
        '-d', '--pregen_data_dir', default=None, help='name pre-generated data directory in the `dataset` folder'
    )
    parser.add_argument(
        '--num_samples_per_task', type=int, default=20, help='number of samples to generate for each task'
    )

    # Distributed Inference Parameters
    parser.add_argument(
        '-nn', '--num_nodes', type=int, default=1, help='number of nodes. For dense attention, default is set to 1.'
    )
    parser.add_argument('-np', '--nproc_per_node', type=int, default=None, help='number of processes per node')

    # Logging
    parser.add_argument(
        '--output_dir',
        default=os.path.join(BASE_DIR, 'results'),
        help='results directory',
    )
    
    parser.add_argument(
        '--time',
        action='store_true',
    )
    
    parser.add_argument(
        '--breakdown',
        action='store_true',
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
    )

    return parser.parse_args()


def submit_job(cmd, log_dir, filename):
    # Save the command to a file
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, filename), 'w') as f:
        f.write(cmd)

    # Submit the job
    os.system(f'cd {BASE_DIR}; {cmd}')


def main(
    model_path: str,
    attn_type: str,
    block_size: int,
    anchor_block_size: int,
    prompt_config: str,
    seq_lengths: List[int],
    tasks: List[str],
    nproc_per_node: int,
    output_dir: str,
    num_samples_per_task: int = 500,
    num_nodes: int = 1,
    pregen_data_dir: Optional[str] = None,
    test_time=False,
    breakdown=False,
    batch=False,
):
    if test_time:
        num_samples_per_task = 20
    if 'star' in attn_type:
        assert (
            block_size >= anchor_block_size
        ), f'block_size ({block_size}) must be greater than anchor_block_size ({anchor_block_size})'

    # Path to any pre-generated data, if exists
    if pregen_data_dir is not None:
        pregen_data_dir = os.path.join(BASE_DIR, 'dataset', pregen_data_dir)

    # Inference Parameters
    stop_words = ','.join(PROMPT_TEMPLATES[prompt_config]['stop_words'])
    with open(os.path.join(BASE_DIR, 'ruler', 'synthetic_inference_config.yaml')) as f:
        tokens_to_generate = yaml.safe_load(f)['tokens_to_generate']

    # Schedule jobs for each sequence length
    for seq_length in seq_lengths:
        if 'star' in attn_type and block_size + anchor_block_size > seq_length:
            print(
                f'block_size + anchor_block_size ({block_size + anchor_block_size}) '
                f'must be less than or equal to seq_length ({seq_length}). '
                'Skipping...'
            )
            continue

        # Depending on the sequence length and the block size, adjust the number of processes
        if attn_type != 'dense' and attn_type != 'minf' and attn_type != 'flash' and attn_type != 'locret' and attn_type != 'h2o' and attn_type != 'snapkv' and attn_type != 'sirllm':
            nproc_per_node_seq_len = min(
                nproc_per_node, (seq_length // block_size) if block_size > 0 else nproc_per_node
            )
            inference_executor = f'torchrun --nnodes={num_nodes} --nproc_per_node={nproc_per_node_seq_len}'
        else:
            inference_executor = 'python'

        results_dir = os.path.join(output_dir, f'{seq_length}')
        log_dir = os.path.join(results_dir, 'logs')
        data_dir = os.path.join(results_dir, 'data')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)

        # Evaluate each task
        for tidx, task in enumerate(tasks):
            task_log_dir = os.path.join(log_dir, task)

            # Prepare dataset
            task_data_file = (
                os.path.join(pregen_data_dir, str(seq_length), task, 'validation.jsonl') if pregen_data_dir else None
            )
            if task_data_file is None or not os.path.exists(task_data_file):
                data_gen_cmd = (
                    f'python ruler/data/prepare.py '
                    f'--save_dir {data_dir} '
                    f'--task {task} '
                    f'--tokenizer_path {model_path} '
                    f'--tokenizer_type hf '
                    f'--max_seq_length {seq_length} '
                    f'--model_template_type {prompt_config} '
                    f'--num_samples {num_samples_per_task}'
                )
                if attn_type == "minf" or attn_type == 'flash' or attn_type == 'locret' or attn_type == 'h2o' or attn_type == 'snapkv' or attn_type == "sirllm":
                    data_gen_cmd = f"CUDA_VISIBLE_DEVICES={tidx} " + data_gen_cmd
                submit_job(data_gen_cmd, task_log_dir, f'data_generation.sh')
                task_data_file = os.path.join(data_dir, task, 'validation.jsonl')

            if attn_type == "lring":
                sc = "run_lring_attn"
            elif attn_type == "ulysses":
                sc = "run_ulysses_attn"
            elif attn_type == "ring":
                sc = "run_ring_attn"
            elif attn_type == "star":
                sc = "run_star_attn"
            elif attn_type == "minf":
                sc = "run_minf_attn"
            elif attn_type == "flash":
                sc = "run_flash_attn"
            elif attn_type == "locret":
                sc = "run_locret_attn"
            elif attn_type == "h2o":
                sc = "run_h2o_attn"
            elif attn_type == "snapkv":
                sc = "run_snapkv_attn"
            elif attn_type == "sirllm":
                sc = "run_sirllm_attn"
            else:
                sc = "run_star_attn_inference"
                
            if test_time:
                sc += "_time"
            
            if breakdown:
                sc += "_b"
            
            if batch:
                sc += "_batch"
                
                
            # Run response generation
            task_gen_cmd = (
                f'{inference_executor} {sc}.py '
                # f'{inference_executor} run_star_attn_inference.py '
                f'--model_path {model_path} '
                f'--attn_type {attn_type} '
                f'--block_size {block_size} '
                f'--anchor_block_size {anchor_block_size} '
                f'--tokens_to_generate {tokens_to_generate[task]} '
                f'--input_path {task_data_file} '
                f'--output_path {os.path.join(results_dir, task)}.jsonl '
                f'--stop_words "{stop_words}"'
            )
            if attn_type == "minf" or attn_type == "flash" or attn_type == "locret" or attn_type == "h2o" or attn_type == "snapkv" or attn_type == "sirllm":
                task_gen_cmd = f"CUDA_VISIBLE_DEVICES={tidx+1} " + task_gen_cmd + " & "
                # task_gen_cmd = f"CUDA_VISIBLE_DEVICES=4,5,6,7 " + task_gen_cmd
            print(f'\nRunning:\n{task_gen_cmd}')
            submit_job(task_gen_cmd, task_log_dir, 'generate_predictions.sh')

        if attn_type != "minf" and not test_time and attn_type != "flash" and attn_type != "locret" and attn_type != "h2o" and attn_type != "snapkv" and attn_type != "sirllm":
            # Run response scoring
            eval_cmd = 'python ruler/eval/evaluate.py ' f'--data_dir {results_dir} ' '--benchmark synthetic'
            print(f'\nRunning:\n{eval_cmd}')
            submit_job(eval_cmd, log_dir, 'evaluate_responses.sh')
            


if __name__ == '__main__':
    # Parse command line arguments
    args = define_cmd_arguments()

    # Validate star attention parameters
    if 'star' in args.attn_type:
        assert args.block_size > 0, 'block_size must be greater than 0'

    # Validate star and ring attention parameters
    if args.attn_type != 'dense' and args.attn_type != 'minf' and args.attn_type != "flash" and args.attn_type != "locret" and args.attn_type != "h2o" and args.attn_type != "snapkv" and args.attn_type != "sirllm":
        assert args.nproc_per_node is not None and args.nproc_per_node > 0, 'nproc_per_node must be greater than 0'
        assert args.num_nodes > 0, 'num_nodes must be greater than 0'

    # Validate model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'{args.model_path} not found')

    # Setup the model name and output directory
    model_name_suffix = ''
    if 'star' in args.attn_type:
        anchor_block_size = args.anchor_block_size if args.anchor_block_size > 0 else args.block_size
        anchor_block_size_repr = (
            f'{anchor_block_size}' if anchor_block_size < 1024 else f'{anchor_block_size // 1024}k'
        )
        block_size_repr = f'{args.block_size}' if args.block_size < 1024 else f'{args.block_size // 1024}k'
        model_name_suffix = f'_b{block_size_repr}a{anchor_block_size_repr}'
    args.output_dir = os.path.join(args.output_dir, f'{args.model_name}_{args.attn_type}{model_name_suffix}')

    main(
        args.model_path,
        args.attn_type,
        args.block_size,
        args.anchor_block_size,
        args.prompt_config,
        args.seq_lengths,
        args.tasks,
        args.nproc_per_node,
        args.output_dir,
        num_samples_per_task=args.num_samples_per_task,
        num_nodes=args.num_nodes,
        pregen_data_dir=args.pregen_data_dir,
        test_time=args.time,
        breakdown=args.breakdown,
        batch=args.batch,
    )
