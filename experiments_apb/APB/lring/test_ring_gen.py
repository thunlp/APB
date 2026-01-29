from .utils import load_ring_model_tokenizer, _gather_hidden_states, ring_prefill_and_decode
import torch
import torch.distributed as dist


dist.init_process_group("nccl")

rank = dist.get_rank()
world_size = dist.get_world_size()


model, tokenizer = load_ring_model_tokenizer("/your_model_path")

input_str = "The sun is yellow. The flower is red. The sea is blue. The grass is green. " * 400 + "The Pass Key is 14265 " + "The sun is yellow. The flower is red. The sea is blue. The grass is green. " * 400 + "What is the pass key? Answer: The pass key is"

enc = tokenizer(input_str, return_tensors='pt').to(f"cuda:{rank}")
input_ids = enc.input_ids
position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to(f"cuda:{rank}")

output_tokens, hiddens = ring_prefill_and_decode(model, input_ids, position_ids, debug=True)
if rank == 7:
    print(tokenizer.decode(output_tokens))
    hs = torch.cat(hiddens, dim=0).unsqueeze(0)
    torch.save(hs, "hiddens.ckpt")

dist.barrier()

