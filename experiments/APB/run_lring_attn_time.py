
import argparse
import json
import os
os.environ['PYTHONUNBUFFERED'] = '1'
import time
from typing import List, Optional
import torch
import torch.distributed as dist
from tqdm import tqdm

from model import DenseAttentionModel, RingAttentionModel, StarAttentionModel
from lring.utils import load_lring_model_tokenizer, lring_prefill_and_decode
import torch.distributed as dist

def read_jsonl(filename, num_lines=-1):
    lines = []
    with open(filename) as f:
        for i, line in enumerate(f):
            lines.append(json.loads(line))
            if i == num_lines:
                break
    return lines


def init_distributed():
    """Initialize the distributed environment."""

    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f'[run_star_attn_inference.init_distributed] Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def get_resume_point(input_data, output_file):
    """Get the total number of samples saved in the output file."""
    if os.path.exists(output_file):
        output_data = read_jsonl(output_file)
        if 'index' in input_data[0]:
            pred_index = [x['index'] for x in output_data]
            input_data = [x for x in input_data if x['index'] not in pred_index]
        else:
            input_data = input_data[len(output_data) :]

    return input_data


def load_model(
    model_path,
    attn_type,
    tokens_to_generate,
    block_size=-1,
    anchor_block_size=-1,
    stop_words=None,
):
    locret_path = '/yourpath'
    model, tokenizer = load_lring_model_tokenizer(model_path, locret_path=locret_path)
    # if dist.get_rank() == 0:
    #     breakpoint()
    # dist.barrier()
    stop_ids = [tokenizer.encode(s)[-1] for s in stop_words]
    return model, tokenizer, stop_ids


def main(
    model_path: str,
    attn_type: str,
    tokens_to_generate: int,
    input_file: str,
    output_file: str,
    num_samples: int = -1,
    stop_words: Optional[List[str]] = None,
    block_size: int = -1,
    anchor_block_size: int = -1,
    use_cache: bool = False,
):
    """Run inference using Star-Attention.

    Args:
        model_path: path to the model checkpoint
        attn_type: type of attention. One of ['dense', 'star', 'starkv']
        tokens_to_generate: number of tokens to generate during generation
        input_file: path to the input jsonl file
        output_file: path to the output jsonl file where the generated predictions will be saved
        stop_words: list of stop words for generation. Default: None
        block_size: block size for star attention. Default: -1 (should be provided for star attention)
        anchor_block_size: anchor block size for star attention. Default: -1 (should be provided for star attention)
        use_cache: resume from last generation if the output file already exists. Default: False
    """
    rank, _ = init_distributed()

    if rank == 0:
        process_start_time = time.time()

    # Load data
    input_data = read_jsonl(input_file)
    if num_samples > 0:
        input_data = input_data[:num_samples]

    # Resume from last generation
    output_file_mode = 'wt'
    if os.path.exists(output_file) and use_cache:
        print(f'Resuming from last generation. Output file already exists: {output_file}')
        input_data = get_resume_point(input_data, output_file)
        output_file_mode = 'at'

    # Load model
    model, tokenizer, stop_ids = load_model(
        model_path,
        attn_type,
        tokens_to_generate,
        block_size,
        anchor_block_size,
        stop_words=stop_words,
    )
    max_new_tokens = tokens_to_generate
    
    torch.cuda.synchronize()
    pre_run_data = input_data[:20]
    test_data = input_data[:20]
    for input_sample in tqdm(pre_run_data, total=len(pre_run_data)):
        input_str = input_sample['input_context'] + input_sample['input_query']
        context_len = len(tokenizer.encode(input_sample['input_context'], add_special_tokens=False))
        input_ids = tokenizer(input_str, return_tensors='pt', add_special_tokens=False).to(f"cuda:{rank}").input_ids
        position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=torch.device(f"cuda:{rank}")).unsqueeze(0)
        generated_ids = lring_prefill_and_decode(model, input_ids, position_ids, context_len, max_new_tokens=max_new_tokens, stop_ids=stop_ids)
    
    tot_tokens = 0
    tot_time = 0
    

    for input_sample in tqdm(test_data, total=len(test_data)):
        input_str = input_sample['input_context'] + input_sample['input_query']
        context_len = len(tokenizer.encode(input_sample['input_context'], add_special_tokens=False))
        input_ids = tokenizer(input_str, return_tensors='pt', add_special_tokens=False).to(f"cuda:{rank}").input_ids
        position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=torch.device(f"cuda:{rank}")).unsqueeze(0)
        
        torch.cuda.synchronize()
        if rank == 0:
            beg_t = time.time()
        
        generated_ids = lring_prefill_and_decode(model, input_ids, position_ids, context_len, max_new_tokens=max_new_tokens, stop_ids=stop_ids)
        
        torch.cuda.synchronize()
        if rank == 0:
            end_t = time.time()
        
            tot_tokens += input_ids.shape[-1] + len(generated_ids)
            tot_time += end_t - beg_t
            
    torch.cuda.synchronize()

    if rank == 0:
        print("=" * 50)
        print(input_file)
        print(f"Processing speed: {(tot_tokens / tot_time):.2f}")
        print("=" * 50)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the model checkpoint')
    parser.add_argument('--attn_type', required=True, help='type of attention')
    parser.add_argument('--block_size', type=int, default=-1, help='block size for star attention')
    parser.add_argument('--anchor_block_size', type=int, default=-1, help='anchor block size for star attention')
    parser.add_argument('--tokens_to_generate', type=int, required=True, help='number of tokens to generate')
    parser.add_argument('--stop_words', default='', help='comma separated stop words for generation')
    parser.add_argument('--input_path', required=True, help='path to the input jsonl file')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples to use from the input file')
    parser.add_argument(
        '--output_path', required=True, help='path to the jsonl file where the generated predictions will be saved'
    )
    parser.add_argument(
        '--use_cache', action='store_true', help='resume from last generation if the output file already exists'
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f'Invalid model path: {args.model_path}')

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f'Invalid input path: {args.input_path}')
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    main(
        args.model_path,
        args.attn_type,
        args.tokens_to_generate,
        args.input_path,
        args.output_path,
        num_samples=args.num_samples,
        stop_words=list(filter(None, args.stop_words.split(','))),
        block_size=args.block_size,
        anchor_block_size=args.anchor_block_size,
        use_cache=args.use_cache,
    )
