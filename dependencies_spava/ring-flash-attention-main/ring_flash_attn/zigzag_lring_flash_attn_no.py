import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from flash_attn_apb.flash_attn_interface import _flash_attn_forward as zigzag_lring_fa_func
from .utils import RingComm, update_out_and_lse, get_default_args, All2AllComm
from torch.nn.attention.flex_attention import flex_attention
import time
import math
from torch import autocast
import torch.cuda.nvtx as nvtx 

sparsity = 0.1
block_size = 1024 * 128 / 8
blk_sz = 1
# blk_sz = 1024

attn_pool = 128
stabilizers = 256
flex_attention = torch.compile(flex_attention)

def zigzag_lring_flash_attn_forward_no(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    k_nope: torch.Tensor,
    v: torch.Tensor,
    cis: torch.Tensor,
    query_len: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    breakdown=False,
    rank=None,
    fake_anchor_len=0,
    passing_len=0,
    block_len_1=0,
    block_len_2=0,
):

    # comm = RingComm(process_group)
    _comm1 = All2AllComm(process_group)
    _comm2 = All2AllComm(process_group)
    rank = _comm1.rank
    world_size = _comm1.world_size    

    out = None
    lse = None
    
    bsz, seq_len, head_num, head_dim = k.shape
    
    anchor_len = fake_anchor_len

    # test
    q_ids = q[:, -query_len:]
    q_ids = q_ids.reshape(bsz, query_len, head_num, -1, head_dim).mean(dim=-2)

    prob_k = k[:, :-query_len, :, :]
    with autocast(device_type='cuda', dtype=torch.float32):
        prob_score = torch.matmul(prob_k.transpose(1, 2), q_ids.transpose(1, 2).transpose(-1, -2)).mean(dim=-1)
    prob_score = torch.nn.functional.max_pool1d(prob_score, kernel_size=5, padding=5 // 2, stride=1)
    score = prob_score.transpose(-1, -2)


    select_num = passing_len 
    
    # do the first block first
    indices_1 = torch.topk(score[:, anchor_len : anchor_len + block_len_1], k=min(select_num, 128 * (block_len_1 // 128)), dim=1).indices + anchor_len
    indices_1 = indices_1.unsqueeze(-1).repeat(1, 1, 1, head_dim)
    sel_k_1 = torch.gather(k, 1, indices_1)
    sel_v_1 = torch.gather(v, 1, indices_1) 
    sel_1 = torch.cat([sel_k_1, sel_v_1], dim=-1) 


    _comm1.all2all(sel_1) # first block: go!
    _comm1.wait() # first block: received!


    # overlab as much as possible
    indices_2 = torch.topk(score[:, anchor_len + block_len_1 : anchor_len + block_len_1 + block_len_2], k=min(select_num, 128 * (block_len_2 // 128)), dim=1).indices + anchor_len + block_len_1
    indices_2 = indices_2.unsqueeze(-1).repeat(1, 1, 1, head_dim)
    sel_k_2 = torch.gather(k, 1, indices_2)
    sel_v_2 = torch.gather(v, 1, indices_2) 
    sel_2 = torch.cat([sel_k_2, sel_v_2], dim=-1) 
    nvtx.range_pop()



    _comm2.all2all(sel_2) # second block: go!
    _comm2.wait() # second block: received!


    anchor_block_len = math.ceil(anchor_len / world_size)
    anchor_left = rank * anchor_block_len
    anchor_right = min((rank + 1) * anchor_block_len, anchor_len)
    query_k = torch.cat((k[:, anchor_left:anchor_right], k[:, anchor_len:-query_len]), dim=1) if rank > 0 else torch.cat((k[:, anchor_left:anchor_right], k[:, anchor_len:]), dim=1)
    query_v = torch.cat((v[:, anchor_left:anchor_right], v[:, anchor_len:-query_len]), dim=1) if rank > 0 else torch.cat((v[:, anchor_left:anchor_right], v[:, anchor_len:]), dim=1)


    block_out_q, _, _, _, _, block_lse_q, _, _ = _flash_attn_forward(  
        q=q[:, -query_len:],
        k=query_k,
        v=query_v,
        dropout_p=0,
        softmax_scale=(q.shape[-1]**(-1/2)),
        causal=True,
        return_softmax=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        softcap=False
    )

    out_gather_big = torch.empty((world_size,) + block_out_q.shape, dtype=block_out_q.dtype, device=block_out_q.device)
    lse_gather_big = torch.empty((world_size,) + block_lse_q.shape, dtype=block_lse_q.dtype, device=block_lse_q.device)
    out_gather = list(out_gather_big)
    lse_gather = list(lse_gather_big)

    req_q_out = dist.all_gather(out_gather, block_out_q, async_op=True)
    req_q_lse = dist.all_gather(lse_gather, block_lse_q, async_op=True)
    req_q_lse.wait()
    req_q_out.wait()



    prv1 = _comm1.data()
    passing_cat_1 = prv1[0:rank]

    if rank > 0:
        passing_cat_1 = torch.cat(passing_cat_1, dim=1)
        pk_1, pv_1 = passing_cat_1[..., :head_dim], passing_cat_1[..., head_dim:]
        prv_len_1 = pk_1.shape[1]
        block_out, _, _, _, _, _, _, _ = zigzag_lring_fa_func(  
            q=q[:, :anchor_len + block_len_1],
            k=torch.cat((pk_1, k[:, :anchor_len + block_len_1]), dim=1),
            v=torch.cat((pv_1, v[:, :anchor_len + block_len_1]), dim=1),
            anchor_len=anchor_len,
            prv_len=prv_len_1,
            dropout_p=0,
            softmax_scale=(q.shape[-1]**(-1/2)),
            causal=True,
            return_softmax=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            softcap=False
        )    
        out_1 = block_out.to(q.dtype)
    else:
        block_out, _, _, _, _, _, _, _ = _flash_attn_forward(  
            q=q[:, :anchor_len + block_len_1],
            k=k[:, :anchor_len + block_len_1],
            v=v[:, :anchor_len + block_len_1],
            dropout_p=0,
            softmax_scale=(q.shape[-1]**(-1/2)),
            causal=True,
            return_softmax=False,
            window_size=(-1, -1),
            alibi_slopes=None,
            softcap=False
        )
        out_1 = block_out.to(q.dtype)


    # do the second block
    prv2 = _comm2.data()
    prv2 = prv2[::-1]
    prv2 = prv1 + prv2
    host_2_rank = 2 * world_size - 1 - rank
    prv2 = prv2[:host_2_rank]
    prv2 = torch.cat(prv2, dim=1)
    pk_2, pv_2 = prv2[..., :head_dim], prv2[..., head_dim:]

    block_out, _, _, _, _, _, _, _ = _flash_attn_forward(  
        q=q[:, anchor_len + block_len_1 :  anchor_len + block_len_1 + block_len_2],
        k=torch.cat((pk_2, k[:, :anchor_len], k[:, anchor_len + block_len_1 :  anchor_len + block_len_1 + block_len_2]), dim=1),
        v=torch.cat((pv_2, v[:, :anchor_len], v[:, anchor_len + block_len_1 :  anchor_len + block_len_1 + block_len_2]), dim=1),
        dropout_p=0,
        softmax_scale=(q.shape[-1]**(-1/2)),
        causal=True,
        return_softmax=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        softcap=False
    )
    out_2 = block_out.to(q.dtype)



    # we merge the query at last

    out_q = None
    lse_q = None
    for i in range(len(out_gather)):
        out_q, lse_q = update_out_and_lse(out_q, lse_q, out_gather[i], lse_gather[i])
    out_q = out_q.to(q.dtype)


    out = torch.cat((out_1, out_2, out_q), dim=1)

    return out



class ZigZagLRingNOFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        k_nope,
        v,
        cis,
        query_len,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        breakdown,
        rank,
        anchor_len,
        passing_len,
        block_len_1,
        block_len_2,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out = zigzag_lring_flash_attn_forward_no(
            group,
            q,
            k,
            k_nope,
            v,
            cis,
            query_len,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            breakdown=breakdown,
            rank=rank,
            fake_anchor_len=anchor_len,
            passing_len=passing_len,
            block_len_1=block_len_1,
            block_len_2=block_len_2,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, None, None)

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError


def zigzag_lring_no_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    anchor_len=0,
    passing_len=0,
    block_len_1=0,
    block_len_2=0,
    query_len=0,
):
    return ZigZagLRingNOFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        anchor_len,
        passing_len,
        block_len_1,
        block_len_2,
    )


def zigzag_lring_no_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    anchor_len=0,
    passing_len=0,
    block_len_1=0,
    block_len_2=0,
    query_len=0,
):
    return ZigZagLRingNOFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        anchor_len,
        passing_len,
        block_len_1,
        block_len_2,
    )


def zigzag_lring_no_flash_attn_func(
    q,
    k,
    k_nope,
    v,
    cis=None,
    query_len=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    breakdown=False,
    rank=None,
    anchor_len=0,
    passing_len=0,
    block_len_1=0,
    block_len_2=0,
):
    return ZigZagLRingNOFlashAttnFunc.apply(
        q,
        k,
        k_nope,
        v,
        cis,
        query_len,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        breakdown,
        rank,
        anchor_len,
        passing_len,
        block_len_1,
        block_len_2,
    )
