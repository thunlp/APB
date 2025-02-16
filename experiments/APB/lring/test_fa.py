from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

rank = 0
model = AutoModelForCausalLM.from_pretrained("/home/test/test01/anonymous/models/Meta-Llama-3-8B", attn_implementation='eager', torch_dtype=torch.float32, device_map=torch.device(f"cuda:{rank}"))
tokenizer = AutoTokenizer.from_pretrained("/home/test/test01/anonymous/models/Meta-Llama-3-8B")

input_str = "The sun is yellow. The flower is red. The sea is blue. The grass is green. " * 100

enc = tokenizer(input_str, return_tensors='pt').to(f"cuda:{rank}")
input_ids = enc.input_ids
position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0).to(f"cuda:{rank}")

# output = model(input_ids, position_ids=position_ids, output_hidden_states=True)


with torch.no_grad():
    output = model(input_ids, position_ids=position_ids, output_hidden_states=True)
    hs = output.hidden_states
    
    soutput = model(input_ids[:, :256], position_ids=position_ids[:, :256], output_hidden_states=True)
    shs = soutput.hidden_states
    
    for i in range(33):
        _hs = hs[i][:, :256, :]
        _shs = shs[i]
        delta = _shs - _hs
        delta = delta.abs()
        print(delta.max())
    breakpoint()
    a = 1
    
    