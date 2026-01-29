import torch
from flash_attn_apb.flash_attn_interface import _flash_attn_forward, flash_attn_with_kvcache, _flash_attn_varlen_forward


kv_len1 = 4096
kv_len2 = 4096
anchor_len = 512
prv_len1 = 512
prv_len2 = 1024



query = torch.randn((1, 8, anchor_len + kv_len1 + kv_len2, 128)).cuda().half() # [A B1 B2]
key = torch.randn((1, 8, prv_len1 + prv_len2 + anchor_len + kv_len1 + kv_len2, 128)).cuda().half() # [P1 P2 A B1 B2] 
value = torch.randn((1, 8, prv_len1 + prv_len2 + anchor_len + kv_len1 + kv_len2, 128)).cuda().half() # [P1 P2 A B1 B2] 


query = query.transpose(1, 2)
key = key.transpose(1, 2)
value = value.transpose(1, 2)

def apb_zigzag_attn(query, key, value, anchor_len, passing_len_1, passing_len_2, block_len_1, block_len_2, block_size=256):
    _, _, h, d = key.shape
    k_cache = key.reshape(-1, block_size, h, d)
    v_cache = value.reshape(-1, block_size, h, d)
    # first forward for (A B1 * P1 A B1)
    query_first = query[:, :-block_len_2] # (A B1)
    block_table = torch.cat((torch.arange(passing_len_1 // block_size), torch.arange((anchor_len + block_len_1) // block_size )+ (passing_len_1 + passing_len_2) // block_size), dim=-1).unsqueeze(0).cuda().to(torch.int32)

    cu_seqlens_q = torch.tensor([0, query_first.shape[1]], dtype=torch.int, device=query_first.device)
    max_seqlen_q = query_first.shape[1]
    cu_seqlens_k = torch.tensor([0, anchor_len + passing_len_1 + block_len_1], dtype=torch.int, device=query_first.device)
    max_seqlen_k = anchor_len + passing_len_1 + block_len_1

    out = flash_attn_with_kvcache(
        q=query_first,
        k_cache=k_cache,
        v_cache=v_cache,
        causal=True,
        block_table=block_table,
        anchor_len=anchor_len,
        prv_len=passing_len_1,
    )
    block_output_1 = out

    # second forward for (B2 * P2 A B2)
    query_second = query[:, -block_len_2:]
    block_table = torch.cat((torch.arange(passing_len_2 // block_size) + passing_len_1 // block_size, torch.arange(anchor_len // block_size) + (passing_len_1 + passing_len_2) // block_size, torch.arange(block_len_2 // block_size) + (passing_len_1 + passing_len_2 + anchor_len + block_len_1) // block_size), dim=-1).unsqueeze(0).cuda().to(torch.int32)

    out = flash_attn_with_kvcache(
        q=query_second,
        k_cache=k_cache,
        v_cache=v_cache,
        causal=True,
        block_table=block_table,
    )
    block_output_2 = out
    return torch.cat((block_output_1, block_output_2), dim=1)

output = apb_zigzag_attn(query, key, value, anchor_len, prv_len1, prv_len2, kv_len1, kv_len2)



q = query[:, :anchor_len + kv_len1]
k = torch.cat(
    (
        key[:, :prv_len1],
        key[:, prv_len1 + prv_len2 : prv_len1 + prv_len2 + anchor_len + kv_len1],
    ), dim=1
)
v = torch.cat(
    (
        value[:, :prv_len1],
        value[:, prv_len1 + prv_len2 : prv_len1 + prv_len2 + anchor_len + kv_len1],
    ), dim=1
)


out, _, _, _, _, _, _, _ = _flash_attn_forward(
    q=q,
    k=k,
    v=v,
    dropout_p=0,
    softmax_scale=(query.shape[-1]**(-1/2)),
    causal=True,
    return_softmax=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap=False,
    # anchor_len=0,
    # prv_len=0,
    anchor_len=anchor_len,
    prv_len=prv_len1,
)

std_ab1 = out






q = torch.cat((
    query[:, :anchor_len],
    query[:, -kv_len2:],
), dim=1)
k = torch.cat(
    (
        key[:, prv_len1 : prv_len1 + prv_len2],
        key[:, prv_len1 + prv_len2 : prv_len1 + prv_len2 + anchor_len],
        key[:, -kv_len2:],
    ), dim=1
)
v = torch.cat(
    (
        value[:, prv_len1 : prv_len1 + prv_len2],
        value[:, prv_len1 + prv_len2 : prv_len1 + prv_len2 + anchor_len],
        value[:, -kv_len2:],
    ), dim=1
)


out, _, _, _, _, _, _, _ = _flash_attn_forward(
    q=q,
    k=k,
    v=v,
    dropout_p=0,
    softmax_scale=(query.shape[-1]**(-1/2)),
    causal=True,
    return_softmax=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    softcap=False,
    anchor_len=anchor_len,
    prv_len=prv_len2,
)

std_b2 = out[:, anchor_len:]

std_output = torch.cat((std_ab1, std_b2), dim=1)

delta = std_output - output
print(delta.abs().max())
print(delta)