# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
os.environ['HF_EVALUATE_OFFLINE'] = '1'
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from args import parse_args
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
import torch.distributed as dist
from lring.utils import load_star_model_tokenizer, load_lring_model_tokenizer, star_prefill_and_decode, lring_prefill_and_decode, load_ulysses_model_tokenizer, ulysses_prefill_and_decode
# from vllm import LLM, SamplingParams

def print_on_master(*args, **kwargs):
    if dist.is_initialized() and dist.get_rank() == 0:
        print(*args, **kwargs)

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, noq_tokens: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input, noq_tokens
    if len(input) <= max_length:
        return input, noq_tokens
    if manner == "middle":
        split = max_length // 2
        start_sec = len(input) - split
        input = input[0:split] + input[-split:]
        noq_tokens = noq_tokens[0:split] + noq_tokens[start_sec:]
        return input, noq_tokens
    else:
        return None, None


def truncate_by_tokens(input, noq_str, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input, add_special_tokens=False)
    noq_tokens = tok.encode(noq_str, add_special_tokens=False)
    len_before = len(tokens)
    print_on_master(f"# tokens before: {len_before}")
    tokens, noq_tokens = truncate_input(tokens, noq_tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print_on_master(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens, len(noq_tokens)


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    input_text_noq: str,
    query_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
    attn_type: str = "vllm",
    method: str = None,
    stop_ids: list = [],
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens, noq_len = truncate_by_tokens(input_text, input_text_noq, tok, max_input_length)
    if verbose:
        print_on_master("# tokens:", len(input_tokens))
        print_on_master("=============== Input ===============")
        print_on_master(tok.decode(input_tokens[:200]))
        print_on_master("...")
        print_on_master(tok.decode(input_tokens[-200:]))
        print_on_master("=====================================")
    if attn_type == "vllm":
        if len(input_tokens) != 1:
            input_tokens = [input_tokens]
        outputs = model.generate(
            prompt_token_ids=input_tokens,
            sampling_params=generation_config,
        )
        output = outputs[0].outputs[0].text
        output = output.strip()
    else:
        input_tensors = {
            "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
        }
        
        if method == 'star':
            position_ids = torch.arange(input_tensors['input_ids'].shape[-1], dtype=torch.long, device=model.device).unsqueeze(0)
            output = star_prefill_and_decode(model, input_tensors['input_ids'], position_ids, noq_len, max_new_tokens=generation_config.max_new_tokens, stop_ids=stop_ids)
        elif method == "lring":
            position_ids = torch.arange(input_tensors['input_ids'].shape[-1], dtype=torch.long, device=model.device).unsqueeze(0)
            query_ids = tok.encode(query_text, add_special_tokens=False)
            query_ids = torch.tensor(query_ids).unsqueeze(0).to(model.device)
            # breakpoint()
            output = lring_prefill_and_decode(model, input_tensors['input_ids'], position_ids, noq_len, max_new_tokens=generation_config.max_new_tokens, stop_ids=stop_ids, query_ids=query_ids)
        elif method == 'ulysses':
            position_ids = torch.arange(input_tensors['input_ids'].shape[-1], dtype=torch.long, device=model.device).unsqueeze(0)
            output = ulysses_prefill_and_decode(model, input_tensors['input_ids'], position_ids, max_new_tokens=generation_config.max_new_tokens, stop_ids=stop_ids)
            
        # outputs = model.generate(**input_tensors, generation_config=generation_config)

        # output = outputs[0, len(input_tokens) :]
        output = tok.decode(output, skip_special_tokens=True)
        output = output.strip()
    # print_on_master(input_text[:5000], input_text[-5000:])
    print_on_master("Chunked generation:", output)
    return output



def init_distributed():
    """Initialize the distributed environment."""

    if 'RANK' in os.environ:
        dist.init_process_group('nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print_on_master(f'[run_star_attn_inference.init_distributed] Rank: {rank}, World size: {world_size}')
    else:
        rank = 0
        world_size = 1

    return rank, world_size

if __name__ == "__main__":
    args = parse_args()
    rank, world_size = init_distributed()

    check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task
    method = args.method
    
    stop_ids = []
    if 'llama-3' in model_name.lower():
        stop_ids = [128001, 128008, 128009]
    

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    # Model
    if method == 'lring':
        model, tok = load_lring_model_tokenizer(args.model_name_or_path, args.locret_path)
    elif method == 'star':
        model, tok = load_star_model_tokenizer(args.model_name_or_path)
    elif method == 'ulysses':
        model, tok = load_ulysses_model_tokenizer(args.model_name_or_path)
    else:
        raise NotImplementedError
    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        if args.attn_type == "vllm":
            generation_config = SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                # temperature=0,
                # top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )

        # Data
        result_dir = Path(args.output_dir, f"{real_model_name}_{method}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print_on_master("==== Evaluation ====")
        print_on_master(f"# examples: {len(examples)}")
        print_on_master(f"Num eval examples: {args.num_eval_examples}")
        print_on_master(f"Verbose: {args.verbose}")
        print_on_master(f"Max new tokens: {max_new_tokens}")

        if os.path.exists(output_path) and not args.rewrite:
            print_on_master(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, real_model_name, max_seq_length)
            exit(0)
        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id:
                continue
            input_text, query_text, input_text_noq = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            # print_on_master(input_text.index(ground_truth), len(input_text), input_text.index(ground_truth) / len(input_text))
            # print_on_master(f"====== Example {i} ======")
            pred = get_pred(
                model,
                tok,
                input_text,
                input_text_noq,
                query_text,
                max_input_length=max_seq_length - max_new_tokens,
                verbose=args.verbose,
                generation_config=generation_config,
                attn_type=args.attn_type,
                method=method,
                stop_ids=stop_ids,
            )
            # print_on_master("Ground Truth", get_answer(eg, data_name))
            if args.verbose:
                print_on_master(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{real_model_name}_{args.attn_type}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print_on_master("==== Results ====")
    print_on_master(json.dumps(results, indent=2))
