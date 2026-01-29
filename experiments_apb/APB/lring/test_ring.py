from .utils import load_ring_model_tokenizer, _gather_hidden_states
import torch
import torch.distributed as dist


dist.init_process_group("nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()


model, tokenizer = load_ring_model_tokenizer("/your_model_path")

input_str = "The sun is yellow. The flower is red. The sea is blue. The grass is green. " * 200

enc = tokenizer(input_str, return_tensors='pt').to(f"cuda:{rank}")
input_ids = enc.input_ids
position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to(f"cuda:{rank}")
with torch.no_grad():
    output = model(input_ids, position_ids=position_ids, output_hidden_states=True)
hs = _gather_hidden_states(output)

if rank == 0:
    print(hs[0].shape)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("/your_model_path", attn_implementation='flash_attention_2', torch_dtype=torch.float16, device_map=torch.device(f"cuda:{rank}"))
    with torch.no_grad():
        output = model(input_ids, position_ids=position_ids, output_hidden_states=True)
    hs = hs[-1]
    s_hs = output.hidden_states[-1]
    hs = hs[:, :s_hs.shape[1], :]
    delta = (hs - s_hs).abs()
    print(delta.abs().max())
else:
    print(hs[0])

dist.barrier()

