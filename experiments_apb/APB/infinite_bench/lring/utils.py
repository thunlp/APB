

import torch, os, json
import torch.distributed as dist
from .modeling_llama_ring import LlamaForCausalLM
from .modeling_qwen_ring import Qwen2ForCausalLM
from .modeling_llama_ring_dist import LlamaForCausalLM as DistRingLlamaForCausalLM
from .modeling_llama_ulysses import LlamaForCausalLM as UlyssesLlamaForCausalLM
from .modeling_qwen_ulysses import Qwen2ForCausalLM as UlyssesQwenForCausalLM
from .modeling_llama_ulysses_dist import LlamaForCausalLM as DistUlyssesLlamaForCausalLM
# from .modeling_llama_lring_batch import LlamaForCausalLM as LRingLlamaForCausalLM
from .modeling_llama_lring import LlamaForCausalLM as LRingLlamaForCausalLM
from .modeling_qwen_lring import Qwen2ForCausalLM as LRingQwenForCausalLM
from .modeling_llama_lring_dist import LlamaForCausalLM as DistLRingLlamaForCausalLM
# from .modeling_llama_star_batch import LlamaForCausalLM as StarLlamaForCausalLM
from .modeling_llama_star import LlamaForCausalLM as StarLlamaForCausalLM
from .modeling_qwen_star import Qwen2ForCausalLM as StarQwenForCausalLM
from .modeling_llama_star_dist import LlamaForCausalLM as DistStarLlamaForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm


def plain_decode(model, input_ids):
    output = model(input_ids, use_cache=True)
    past_key_values = output.past_key_values
    generated_ids = []
    next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_ids.append(next_ids.item())
    for _ in range(10):
        output = model(next_id, past_key_values=past_key_values)
        past_key_values = output.past_key_values
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_ids.append(next_ids.item())
            

def load_ring_model_tokenizer(path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = Qwen2ForCausalLM.from_pretrained(path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map=torch.device(f"cuda:{rank}"))
    model.set_dist(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def load_ulysses_model_tokenizer(path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = UlyssesQwenForCausalLM.from_pretrained(path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map=torch.device(f"cuda:{rank}"))
    model.set_dist(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def load_lring_model_tokenizer(path: str, locret_path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = LRingQwenForCausalLM.from_pretrained(path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map=torch.device(f"cuda:{rank}"))
    ckpt = torch.load(locret_path)
    pruned_ckpt = {}
    for k, v in ckpt.items():
        if 'fc' in k:
            pruned_ckpt[k] = v
    model.load_state_dict(pruned_ckpt, strict=False)
    model.set_dist(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

def load_star_model_tokenizer(path: str):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = StarQwenForCausalLM.from_pretrained(path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map=torch.device(f"cuda:{rank}"))
    model.set_dist(rank, world_size)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

@torch.no_grad()
def ring_prefill_and_decode(model, input_ids, position_ids, max_new_tokens = 12, stop_ids = [], debug=False):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    # 1. prefill by ring attention
    output = model(input_ids, position_ids=position_ids, use_cache=True, output_hidden_states=debug)
    hiddens = []
    if debug and rank == world_size - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if rank == world_size - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and rank == world_size - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if rank == world_size - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids
    
@torch.no_grad()
def ulysses_prefill_and_decode(model, input_ids, position_ids, max_new_tokens = 12, stop_ids = [], debug=False):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    # 1. prefill
    output = model(input_ids, position_ids=position_ids, use_cache=True, output_hidden_states=debug)
    hiddens = []
    if debug and rank == world_size - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if rank == world_size - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and rank == world_size - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if rank == world_size - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids

# batch
@torch.no_grad()
def lring_prefill_and_decode_batch(model, input_ids, position_ids, context_len, max_new_tokens = 12, stop_ids = [], debug=False):
    bsz = 12
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    
    input_ids = input_ids.repeat(bsz, 1)
    position_ids = position_ids.repeat(bsz, 1)
    
    # 1. prefill by ring attention
    output = model(input_ids[:, :context_len], query_ids=input_ids[:, context_len:], position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug)
    dist.barrier()
    print(f"batch size: {bsz} success!")
    exit(0)
    # past_key_values = output.past_key_values
    # output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)
    # if rank == world_size - 1:
    #     past_key_values = output.past_key_values
    # hiddens = []
    # if debug and rank == world_size - 1:
    #     hiddens.append(output.hidden_states[-1][:, -1, :])
    # if rank == world_size - 1: # generate next_id on the last gpu
    #     # if output.logits.shape[1] == 0:
    #     #     breakpoint()
    #     next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    # else: # wait and receive next_id for other gpus
    #     next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # # dist.barrier()
    # dist.broadcast(next_id, src=world_size-1)
    # generated_ids.append(next_id.item())
    # # print(next_id)
    # position_ids = position_ids[:, -1:] + 1
    # past_key_values = output.past_key_values
    # for _ in range(max_new_tokens - 1):
    #     output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
    #     if debug and rank == world_size - 1:
    #         hiddens.append(output.hidden_states[-1][:, -1, :])
    #     position_ids = position_ids[:, -1:] + 1
    #     if rank == world_size - 1:
    #         next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #         past_key_values = output.past_key_values # only update the kv cache on the last gpu
    #     else:
    #         next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    #     dist.broadcast(next_id, src=world_size-1)
    #     # print(next_id)
    #     if next_id.item() in stop_ids:
    #         break
    #     generated_ids.append(next_id.item())
    # if debug:
    #     return generated_ids, hiddens
    # else:
    #     return generated_ids
    
@torch.no_grad()
def lring_prefill_and_decode(model, input_ids, position_ids, context_len, max_new_tokens = 12, stop_ids = [], debug=False, query_ids=None):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    # 1. prefill by ring attention
    if query_ids is None:
        query_ids = input_ids[:, context_len:]

    output = model(input_ids[:, :context_len], query_ids=query_ids, external_query_ids=True, position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug)
    past_key_values = output.past_key_values
    output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)
    if rank == world_size - 1:
        past_key_values = output.past_key_values
    hiddens = []
    if debug and rank == world_size - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if rank == world_size - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and rank == world_size - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if rank == world_size - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids
        
# batch  
@torch.no_grad()
def star_prefill_and_decode_batch(model, input_ids, position_ids, context_len, max_new_tokens = 12, stop_ids = [], debug=False):
    bsz = 7
    
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    
    
    
    input_ids = input_ids.repeat(bsz, 1)
    position_ids = position_ids.repeat(bsz, 1)
    
    
    # 1. prefill by ring attention
    output = model(input_ids[:, :context_len], position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug, num_logits_to_keep=1)
    # past_key_values = output.past_key_values
    
    print(f"batch size: {bsz} success!")
    exit(0)
    
    
    # output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)
    # if rank == world_size - 1:
    #     past_key_values = output.past_key_values
    # hiddens = []
    # if debug and rank == world_size - 1:
    #     hiddens.append(output.hidden_states[-1][:, -1, :])
    # if rank == world_size - 1: # generate next_id on the last gpu
    #     # if output.logits.shape[1] == 0:
    #     #     breakpoint()
    #     next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    # else: # wait and receive next_id for other gpus
    #     next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # # dist.barrier()
    # dist.broadcast(next_id, src=world_size-1)
    # generated_ids.append(next_id.item())
    # # print(next_id)
    # position_ids = position_ids[:, -1:] + 1
    # past_key_values = output.past_key_values
    # for _ in range(max_new_tokens - 1):
    #     output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
    #     if debug and rank == world_size - 1:
    #         hiddens.append(output.hidden_states[-1][:, -1, :])
    #     position_ids = position_ids[:, -1:] + 1
    #     if rank == world_size - 1:
    #         next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    #         past_key_values = output.past_key_values # only update the kv cache on the last gpu
    #     else:
    #         next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    #     dist.broadcast(next_id, src=world_size-1)
    #     # print(next_id)
    #     if next_id.item() in stop_ids:
    #         break
    #     generated_ids.append(next_id.item())
    # if debug:
    #     return generated_ids, hiddens
    # else:
    #     return generated_ids
        
@torch.no_grad()
def star_prefill_and_decode(model, input_ids, position_ids, context_len, max_new_tokens = 12, stop_ids = [], debug=False):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    # 1. prefill by ring attention
    output = model(input_ids[:, :context_len], position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug)
    past_key_values = output.past_key_values
    output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)
    if rank == world_size - 1:
        past_key_values = output.past_key_values
    hiddens = []
    if debug and rank == world_size - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if rank == world_size - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and rank == world_size - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if rank == world_size - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids
        
    
    

def _gather_hidden_states(output):
    hs = output.hidden_states
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered_hs = []

    for i, hsl in enumerate(hs):
        g_hs = [torch.empty_like(hsl) for _ in range(world_size)] if rank == 0 else None
        
        dist.barrier()
        dist.gather(hsl, gather_list=g_hs, dst=0)
        dist.barrier()

        if rank == 0:
            gathered_layer_states = torch.cat(g_hs, dim=1) 
            gathered_hs.append(gathered_layer_states)
        else:
            gathered_hs.append(None)

    return gathered_hs

def load_dist_model(model, path, nn, np):
    rank = dist.get_rank()
    print(f"load model on rank: {rank}")
    machine_idx = rank // np
    layer_num = len(model.model.layers)
    
    param_index_path = os.path.join(path, "model.safetensors.index.json")
    with open(param_index_path, 'r') as fp:
        param_index = json.load(fp)
    load_files = []
    for k, v in param_index['weight_map'].items():
        if k.startswith("model.layers"):
            k = k.split(".")
            layer = int(k[2])
            if layer >= machine_idx * layer_num and layer < (machine_idx + 1) * layer_num:
                load_files.append(v)
    if machine_idx == 0:
        load_files.append(param_index['weight_map']['model.embed_tokens.weight'])
    if machine_idx == nn - 1:
        load_files.append(param_index['weight_map']['lm_head.weight'])
        load_files.append(param_index['weight_map']['model.norm.weight'])
    load_files = set(load_files)
    
    state_dict = {}
    for file in tqdm(load_files):
        ckpt_path = os.path.join(path, file)
        sd = load_safetensors(ckpt_path)
        state_dict.update(sd)
    
    clean_state_dict = {}
    device = torch.device(f"cuda:{rank % np}")
    for k, v in tqdm(state_dict.items()):
        if k.startswith("model.layers"):
            k = k.split(".")
            layer = int(k[2])
            k[2] = str(layer - layer_num * machine_idx)
            k_name = ".".join(k)
            if layer >= machine_idx * layer_num and layer < (machine_idx + 1) * layer_num:
                clean_state_dict[k_name] = v.to(torch.bfloat16).to(device)
                if rank % np == 0:
                    k[2] = str(layer)
                    print(".".join(k))
                del v
        else:
            if rank % np == 0:
                print(k)
            clean_state_dict[k] = v.to(torch.bfloat16).to(device)
            del v
        torch.cuda.empty_cache()
    
    print(f"loading state dict on rank: {rank}")
    if rank % np == 0:
        print(clean_state_dict.keys())
    output = model.load_state_dict(clean_state_dict, strict=False)
    print(f"load model finished on rank: {rank}")
    return model

def load_lring_model_tokenizer_dist(path: str, locret_path: str, nn: int, np: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    config = AutoConfig.from_pretrained(path)
    assert config.num_hidden_layers % nn == 0
    config.num_hidden_layers //= nn
    config._attn_implementation = 'flash_attention_2'
    model = DistLRingLlamaForCausalLM(config) #.to(torch.bfloat16)
    
    for name, param in model.named_parameters():
        # print(f"Converting parameter: {name} (current dtype: {param.dtype})")
        param.data = param.data.to(torch.bfloat16)
    
    # if rank == 0:
    #     breakpoint()
    # dist.barrier()
    
    # load params
    
    model = load_dist_model(model, path, nn, np)
    
    device = torch.device(f"cuda:{rank % np}")
    
    # load locret
    ckpt = torch.load(locret_path, weights_only=True, map_location=device)
    pruned_ckpt = {}
    machine_idx = rank // np
    layer_num = config.num_hidden_layers
    for k, v in tqdm(ckpt.items()):
        if 'fc' in k:
            layer_idx = int(k.split('.')[2])
            if layer_idx >= machine_idx * layer_num and layer_idx < (machine_idx + 1) * layer_num:
                k_name = k.split('.')
                k_name[2] = str(layer_idx - layer_num * machine_idx)
                pruned_ckpt['.'.join(k_name)] = v
    print(f"load locret on rank {rank}")
    model.load_state_dict(pruned_ckpt, strict=False)
    
    model.to(f"cuda:{rank % np}")
    # model.model.rotary_emb.inv_freq = model.model.rotary_emb.inv_freq.to(torch.float32) # restore fp32
    # model.model.rotary_emb._init_rope(torch.device(f"cuda:{rank % np}"))
    model.set_dist(rank, world_size, nn, np)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

@torch.no_grad()
def lring_prefill_and_decode_dist(model, input_ids, position_ids, context_len, query_ids = None, max_new_tokens = 12, stop_ids = [], debug=False, nn=1, np=1):
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    local_rank = rank % np
    # 1. prefill by ring attention
    if query_ids is None:
        query_ids = input_ids[:, context_len:]
    output = model(input_ids[:, :context_len], query_ids=query_ids, position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug)
    
    past_key_values = output.past_key_values
    output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)

    if local_rank == np - 1:
        past_key_values = output.past_key_values
    hiddens = []
    if debug and local_rank == np - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if local_rank == np - 1: # generate next_id on the last gpu
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
    
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and local_rank == np - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if local_rank == np - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids


def load_ulysses_model_tokenizer_dist(path: str, nn: int, np: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    config = AutoConfig.from_pretrained(path)
    assert config.num_hidden_layers % nn == 0
    config.num_hidden_layers //= nn
    config._attn_implementation = 'flash_attention_2'
    model = DistUlyssesLlamaForCausalLM(config) #.to(torch.bfloat16)
    
    for name, param in model.named_parameters():
        # print(f"Converting parameter: {name} (current dtype: {param.dtype})")
        param.data = param.data.to(torch.bfloat16)
    
    # load params
    model = load_dist_model(model, path, nn, np)
    
    model.to(f"cuda:{rank % np}")
    model.set_dist(rank, world_size, nn, np)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


@torch.no_grad()
def ulysses_prefill_and_decode_dist(model, input_ids, position_ids, max_new_tokens = 12, stop_ids = [], debug=False, nn=1, np=1):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    local_rank = rank % np
    # 1. prefill
    output = model(input_ids, position_ids=position_ids, use_cache=True, output_hidden_states=debug)
    hiddens = []
    if debug and local_rank == np - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if local_rank == np - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and local_rank == np - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if local_rank == np - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids
    
def load_star_model_tokenizer_dist(path: str, nn: int, np: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    config = AutoConfig.from_pretrained(path)
    assert config.num_hidden_layers % nn == 0
    config.num_hidden_layers //= nn
    config._attn_implementation = 'flash_attention_2'
    model = DistStarLlamaForCausalLM(config) 
    
    for name, param in model.named_parameters():
        param.data = param.data.to(torch.bfloat16)
    
    # load params
    model = load_dist_model(model, path, nn, np)
    
    model.to(f"cuda:{rank % np}")
    model.set_dist(rank, world_size, nn, np)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

@torch.no_grad()
def star_prefill_and_decode_dist(model, input_ids, position_ids, context_len, max_new_tokens = 12, stop_ids = [], debug=False, nn=1, np=1):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    local_rank = rank % np
    # 1. prefill by ring attention
    output = model(input_ids[:, :context_len], position_ids=position_ids[:, :context_len], use_cache=True, output_hidden_states=debug)
    past_key_values = output.past_key_values
    output = model(input_ids[:, context_len:], position_ids=position_ids[:, context_len:], use_cache=True, output_hidden_states=debug, past_key_values=past_key_values, use_star=True)
    if local_rank == np - 1:
        past_key_values = output.past_key_values
    hiddens = []
    if debug and local_rank == np - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if local_rank == np - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and local_rank == np - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if local_rank == np - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids
    

def load_ring_model_tokenizer_dist(path: str, nn: int, np: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    config = AutoConfig.from_pretrained(path)
    assert config.num_hidden_layers % nn == 0
    config.num_hidden_layers //= nn
    config._attn_implementation = 'flash_attention_2'
    model = DistRingLlamaForCausalLM(config) #.to(torch.bfloat16)
    
    for name, param in model.named_parameters():
        # print(f"Converting parameter: {name} (current dtype: {param.dtype})")
        param.data = param.data.to(torch.bfloat16)
    
    # load params
    model = load_dist_model(model, path, nn, np)
    
    model.to(f"cuda:{rank % np}")
    model.set_dist(rank, world_size, nn, np)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


@torch.no_grad()
def ring_prefill_and_decode_dist(model, input_ids, position_ids, max_new_tokens = 12, stop_ids = [], debug=False, nn=1, np=1):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generated_ids = []
    local_rank = rank % np
    # 1. prefill
    output = model(input_ids, position_ids=position_ids, use_cache=True, output_hidden_states=debug)
    hiddens = []
    if debug and local_rank == np - 1:
        hiddens.append(output.hidden_states[-1][:, -1, :])
    if local_rank == np - 1: # generate next_id on the last gpu
        # if output.logits.shape[1] == 0:
        #     breakpoint()
        next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True) # （1, 1）
    else: # wait and receive next_id for other gpus
        next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
    # dist.barrier()
    dist.broadcast(next_id, src=world_size-1)
    generated_ids.append(next_id.item())
    # print(next_id)
    position_ids = position_ids[:, -1:] + 1
    past_key_values = output.past_key_values
    for _ in range(max_new_tokens - 1):
        output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, use_cache=True, output_hidden_states=debug)
        if debug and local_rank == np - 1:
            hiddens.append(output.hidden_states[-1][:, -1, :])
        position_ids = position_ids[:, -1:] + 1
        if local_rank == np - 1:
            next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values # only update the kv cache on the last gpu
        else:
            next_id = torch.zeros((1, 1), dtype=torch.long, device=torch.device(f"cuda:{local_rank}"))
        dist.broadcast(next_id, src=world_size-1)
        # print(next_id)
        if next_id.item() in stop_ids:
            break
        generated_ids.append(next_id.item())
    if debug:
        return generated_ids, hiddens
    else:
        return generated_ids