import os, json
import torch
import torch.distributed as dist
from .models import LlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
import time

def load_apb_model_tokenizer_llama(path: str, locret_path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model_class = LlamaForCausalLM 
    model = model_class.from_pretrained(path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map=torch.device(f"cuda:{rank}"))
    ckpt = torch.load(locret_path)
    pruned_ckpt = {}
    for k, v in ckpt.items():
        if 'fc' in k:
            pruned_ckpt[k] = v
    model.load_state_dict(pruned_ckpt, strict=False)
    model.set_dist(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer