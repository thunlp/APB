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





if __name__ == "__main__":
    args = parse_args()

    check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task
    method = args.method
    
    stop_ids = [151645] # Qwen2.5-14B-Instruct <|im_end|>
    

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]


    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        # Data
        result_dir = Path(args.output_dir, f"{real_model_name}_{method}")
        output_path = result_dir / f"prediction_{data_name}.jsonl"

        result_file_path = f"{real_model_name}_{args.attn_type}"
        
        
        score = compute_scores(output_path, data_name, result_file_path)

