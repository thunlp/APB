
import argparse
import json
import os
import time
from typing import List, Optional
import torch
import torch.distributed as dist
from tqdm import tqdm

from apb import load_apb_model_tokenizer_llama, apb_prefill_and_decode_llama

import torch.distributed as dist


def init_distributed():
    """Initialize the distributed environment."""
    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f'[run_apb_inference.init_distributed] Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    return rank, world_size


def load_model(
    model_path,
    locret_path,
):
    model, tokenizer = load_apb_model_tokenizer_llama(model_path, locret_path=locret_path)
    return model, tokenizer, [tokenizer.eos_token_id]

def make_input(digits):
    head = "There is a pass key hidden in the context. Find it and remember it. I will quiz you about it later. "
    before = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * 2000
    needle = f"The pass key is {digits}. Remember it. The pass key is {digits}"
    after = "The sky is blue. The tree is green. The flower is red. The sun is yellow. " * 4000
    return head + before + needle + after
    
    

def main(
    model_path: str,
    locret_path: str,
    tokens_to_generate: int,
    digits: int,
):
    rank, _ = init_distributed()

    # Load model
    model, tokenizer, stop_ids = load_model(
        model_path,
        locret_path,
    )
    max_new_tokens = tokens_to_generate
    
    query_str = "Now, give me the exact number of the pass key. The pass key is "
    input_str = make_input(digits)
    doc = tokenizer.apply_chat_template([{"role": "user", "content": input_str}], tokenize=False)
    context_len = len(tokenizer.encode(doc, add_special_tokens=False))
    input_ids = tokenizer(doc + query_str, return_tensors='pt', add_special_tokens=False).to(f"cuda:{rank}").input_ids
    query_ids = tokenizer(query_str, return_tensors='pt', add_special_tokens=False).to(f"cuda:{rank}").input_ids
    position_ids = torch.arange(input_ids.shape[-1], dtype=torch.long, device=torch.device(f"cuda:{rank}")).unsqueeze(0)
    
    generated_ids = apb_prefill_and_decode_llama(model, input_ids, position_ids, context_len, max_new_tokens=max_new_tokens, stop_ids=stop_ids)
    pred = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    if rank == 0:
        print(f"Ground Truth: {digits}")
        print(f"Prediction: {pred}")


    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the model checkpoint')
    parser.add_argument('--locret_path', required=True, help='path to locret checkpoint')
    parser.add_argument('--tokens_to_generate', type=int, default=12, help='number of tokens to generate')
    parser.add_argument('--digits', type=int, default=41435554, help='ground truth of NIAH')
    args = parser.parse_args()

    main(
        args.model_path,
        args.locret_path,
        args.tokens_to_generate,
        args.digits,
    )
