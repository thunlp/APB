import torch
from safetensors.torch import load_file, save_file


def safetensor_to_ckpt(safetensor_path, ckpt_path):
    # Load the safetensor file
    safetensor_dict = load_file(safetensor_path)

    filtered_dict = {key: value for key, value in safetensor_dict.items() if 'fc' in key}
    
    # Save the filtered dictionary as a PyTorch checkpoint
    torch.save(filtered_dict, ckpt_path)
    print(f"Successfully converted {safetensor_path} to {ckpt_path}, keeping only 'fc' parameters")


safetensor_path = 'somepath/model.safetensors'
ckpt_path = 'somepath/model.bin'
safetensor_to_ckpt(safetensor_path, ckpt_path)