from .utils import load_ring_model_tokenizer, _gather_hidden_states, ring_prefill_and_decode
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

rank = 0
world_size = 1



# kv_cache = []
# for i in range(8):
#     k = torch.load(f"k_{i}.pt")
#     v = torch.load(f"v_{i}.pt")
#     breakpoint()
#     a = 1




model = AutoModelForCausalLM.from_pretrained("/your_model_path", device_map="cuda", torch_dtype=torch.float16, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained("/your_model_path")

input_str = "The sun is yellow. The flower is red. The sea is blue. The grass is green. " * 400 + "The Pass Key is 14265 " + "The sun is yellow. The flower is red. The sea is blue. The grass is green. " * 400 + "What is the pass key? Answer: The pass key is"

enc = tokenizer(input_str, return_tensors='pt').to(f"cuda:{rank}")
input_ids = enc.input_ids
position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to(f"cuda:{rank}")

shs = []
output = model(input_ids, position_ids=position_ids, output_hidden_states=True, use_cache=True)
shs.append(output.hidden_states[-1][:, -1, :])
next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
past_key_values = output.past_key_values
position_ids = position_ids[:, -1:] + 1
generated_ids = [next_id.item()]
for _ in range(11):
    output = model(next_id, position_ids=position_ids, past_key_values=past_key_values, output_hidden_states=True, use_cache=True)
    shs.append(output.hidden_states[-1][:, -1, :])
    next_id = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    past_key_values = output.past_key_values
    position_ids = position_ids[:, -1:] + 1
    generated_ids.append(next_id.item())

print(tokenizer.decode(generated_ids))

shs = torch.cat(shs, dim=0).unsqueeze(0)
hs = torch.load("hiddens.ckpt").cuda()
delta = (hs - shs).abs()
print(delta[0].max(dim=-1).values)