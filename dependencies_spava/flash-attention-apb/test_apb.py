import torch
from torch.nn.attention.flex_attention import flex_attention
from flash_attn_apb.flash_attn_interface import _flash_attn_forward


kv_len = 319 + 128 * 4
anchor_len = 128
prv_len = 128 * 4


query = torch.randn((1, kv_len - prv_len, 16, 128)).cuda().half()
key = torch.randn((1, kv_len, 2, 128)).cuda().half()
value = torch.randn((1, kv_len, 2, 128)).cuda().half()

anchor_out, _, _, _, _, _, _, _ = _flash_attn_forward(
    q=query[:, :anchor_len],
    k=key[:, :anchor_len],
    v=value[:, :anchor_len],
    dropout_p=0,
    softmax_scale=(query.shape[-1]**(-1/2)),
    causal=True,
    return_softmax=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap=False, 
    anchor_len=anchor_len,
    prv_len=0,
)

block_out, _, _, _, _, _, _, _ = _flash_attn_forward(
    q=query[:, anchor_len:],
    k=key,
    v=value,
    dropout_p=0,
    softmax_scale=(query.shape[-1]**(-1/2)),
    causal=True,
    return_softmax=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap=False, 
    anchor_len=0,
    prv_len=0,
)





key = torch.cat((key[:, anchor_len:anchor_len+prv_len, ...], key[:, :anchor_len, ...], key[:, anchor_len+prv_len:, ...]), dim=1)
value = torch.cat((value[:, anchor_len:anchor_len+prv_len, ...], value[:, :anchor_len, ...], value[:, anchor_len+prv_len:, ...]), dim=1)

out, _, _, _, _, _, _, _ = _flash_attn_forward(
    q=query,
    k=key,
    v=value,
    dropout_p=0,
    softmax_scale=(query.shape[-1]**(-1/2)),
    causal=True,
    return_softmax=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap=False, 
    anchor_len=anchor_len,
    prv_len=prv_len,
)



delta = anchor_out - out[:, :anchor_len]
print(delta)

delta = block_out - out[:, anchor_len:]
print(delta)
