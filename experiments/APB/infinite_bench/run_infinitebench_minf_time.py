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
from lring.utils import load_star_model_tokenizer, load_lring_model_tokenizer, star_prefill_and_decode, lring_prefill_and_decode
# from vllm import LLM, SamplingParams
from minference import MInference

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
    # print(f"# tokens before: {len_before}")
    tokens, noq_tokens = truncate_input(tokens, noq_tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    # print(f"# tokens after: {len_after}")
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
    test_time: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens, noq_len = truncate_by_tokens(input_text, input_text_noq, tok, max_input_length)
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
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
        elif method == 'minference':
            if test_time:
                torch.cuda.synchronize()
                beg_t = time.time()
            outputs = model.generate(**input_tensors, generation_config=generation_config)
            output = outputs[0, len(input_tokens) :]
            if test_time:
                torch.cuda.synchronize()
                end_t = time.time()
            
        output = tok.decode(output, skip_special_tokens=True)
        output = output.strip()
    # print(input_text[:5000], input_text[-5000:])
    # print("Chunked generation:", output)
    if not test_time:
        return output
    else:
        return output, end_t - beg_t, input_tensors['input_ids'].shape[-1] + len(output)



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

def load_model(
    model_name: str,
    topk: int = -1,
    starting_layer: int = -1,
    topk_dims_file_path: str = "",
    use_sparq: bool = False,
    attn_type: str = "vllm",
    max_seq_length: int = None,
    is_search: bool = False,
    use_snapkv: bool = False,
    trust_remote_code: bool = False,
    kv_cache_cpu: bool = False,
    kv_cache_cpu_device: str = "cpu",
):
    tok = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code,
    )
    tok.pad_token = tok.eos_token
    minference_patch = MInference(
        attn_type,
        model_name,
        config_path=topk_dims_file_path if 'yi' not in model_name.lower() else "your_config_path",
        starting_layer=starting_layer,
        use_snapkv=use_snapkv,
        is_search=is_search,
        kv_cache_cpu=kv_cache_cpu,
        kv_cache_cpu_device=kv_cache_cpu_device,
    )

    if attn_type == "vllm":
        llm = LLM(
            model_name,
            max_num_seqs=1,
            swap_space=64,
            gpu_memory_utilization=0.98,
            max_model_len=max_seq_length,
        )
    else:
        config = LlamaConfig.from_pretrained(
            model_name, resume_download=None, trust_remote_code=trust_remote_code
        )
        if "LWM" in model_name:
            c = {
                "theta": 10000000,
                "max_sequence_length": 131072,
                "scan_attention": True,
                "scan_query_chunk_size": 1024,
                "scan_key_chunk_size": 1024,
                "scan_mlp": True,
                "scan_mlp_chunk_size": 1024,
                "scan_layers": True,
            }
            config.update(c)

        llm = LlamaForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            resume_download=None,
            trust_remote_code=trust_remote_code,
        )
    llm = minference_patch(llm)
    # breakpoint()

    print("Model and tokenizer loaded.")
    return llm, tok

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
        
    model, tok = load_model(
        model_name,
        args.topk,
        args.starting_layer,
        args.topk_dims_file_path,
        args.use_sparq,
        attn_type=args.attn_type,
        max_seq_length=max_seq_length,
        is_search=False,
        use_snapkv=args.use_snapkv,
        trust_remote_code=args.trust_remote_code,
        kv_cache_cpu=args.kv_cache_cpu,
        kv_cache_cpu_device=args.kv_cache_cpu_device,
    )
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
        result_dir = Path(args.output_dir, f"{real_model_name}_{args.attn_type}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")

        if data_name == "longbook_sum_eng":
            examples = examples[:5]
        else:
            examples = examples[:20]

         # warmup
        for i, eg in tqdm(enumerate(examples)):
            input_text, query_text, input_text_noq = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
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
            
        # test time
        tot_tok = 0
        tot_time = 0
        for i, eg in tqdm(enumerate(examples)):
            input_text, query_text, input_text_noq = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)
            pred, run_t, tok_num = get_pred(
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
                test_time=True,
            )
            tot_tok += tok_num
            tot_time += run_t
        
        print("=" * 50)
        print(data_name)
        print(f"Processing speed: {(tot_tok / tot_time):.2f}")