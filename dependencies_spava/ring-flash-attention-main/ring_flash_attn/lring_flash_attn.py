import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from flash_attn_apb.flash_attn_interface import _flash_attn_forward as lring_fa_func
from .utils import RingComm, update_out_and_lse, get_default_args, All2AllComm
from torch.nn.attention.flex_attention import flex_attention
import time

sparsity = 0.1
block_size = 1024 * 128 / 8
blk_sz = 1
# blk_sz = 1024

attn_pool = 128
stabilizers = 256
flex_attention = torch.compile(flex_attention)

def lring_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
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
):
    # comm = RingComm(process_group)
    _comm = All2AllComm(process_group)


    out = None
    lse = None
    
    bsz, seq_len, head_num, head_dim = k.shape
    

    anchor_len = fake_anchor_len if _comm.rank > 0 else 0
    
    prv_len = 0
    score = cis
    if score.shape[-1] < head_num:
        score = score.unsqueeze(-1).repeat(1, 1, 1, head_num // score.shape[-1]).reshape(score.shape[0], score.shape[1], -1)

    # make block

    #dist.barrier()
    #if rank == 1:
    #    breakpoint()
    #dist.barrier()

    score_tail_len = seq_len - (blk_sz * (seq_len // blk_sz))
    s_len = (blk_sz * (seq_len // blk_sz))
    if score_tail_len > 0:
        score_tail = score[:, -score_tail_len:, :]  
        score = score[:, :-score_tail_len, :]  
    # if _comm.rank == 0:
    #     breakpoint()
    # dist.barrier()
    score = score.reshape(bsz, s_len // blk_sz, blk_sz, head_num)
    score = score.max(dim=2, keepdim=True).values
    score = score.repeat(1, 1, blk_sz, 1)
    score = score.reshape(bsz, s_len, head_num)
    if score_tail_len > 0:
        score = torch.cat((score, score_tail), dim=1)
    
    # abl
    # score = torch.randn_like(score)
    
    
    score[:, :fake_anchor_len, :] = torch.finfo(score.dtype).min
    
    select_num = passing_len 
    
    
    indices = torch.topk(score, k=select_num, dim=1).indices
    indices = indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
    sel_k = torch.gather(k, 1, indices)
    sel_v = torch.gather(v, 1, indices) 
    sel = torch.cat([sel_k, sel_v], dim=-1)   

    if breakdown:
        dist.barrier()
        comm_beg = time.time()
    
    _comm.all2all(sel)
    
    _comm.wait()
    
    if breakdown:
        dist.barrier()
        comm_end = time.time()
    
    if _comm.rank > 0:
        prv = _comm.data()
        prv = prv[0:_comm.rank]
        prv = torch.cat(prv, dim=1)
        prv_k = prv[..., :head_dim]
        prv_v = prv[..., -head_dim:]
        prv_len = prv_k.shape[1]
        # k = torch.cat([k[:, :anchor_len, :, :], prv_k, k[:, anchor_len:, :, :]], dim=1)
        # v = torch.cat([v[:, :anchor_len, :, :], prv_v, v[:, anchor_len:, :, :]], dim=1)
        k = torch.cat((prv_k, k), dim=1)
        v = torch.cat((prv_v, v), dim=1)
    
    
    if breakdown:
        dist.barrier()
        attn_beg = time.time()
    block_out, _, _, _, _, _, _, _ = lring_fa_func(  
        q=q,
        k=k,
        v=v,
        anchor_len=anchor_len,
        prv_len=prv_len,
        dropout_p=0,
        softmax_scale=(q.shape[-1]**(-1/2)),
        causal=True,
        return_softmax=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        softcap=False
        # block_pos=block_pos 
    )
    
    out = block_out.to(q.dtype)
    if breakdown:
        dist.barrier()
        attn_end = time.time()
    
    if breakdown:
        return out, comm_end - comm_beg, attn_end - attn_beg
    
    return out

class LRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
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
        passing_len
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out = lring_flash_attn_forward(
            group,
            q,
            k,
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


def lring_flash_attn_qkvpacked_func(
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
):
    return LRingFlashAttnFunc.apply(
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
    )


def lring_flash_attn_kvpacked_func(
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
):
    return LRingFlashAttnFunc.apply(
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
    )


def lring_flash_attn_func(
    q,
    k,
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
    passing_len=0
):
    return LRingFlashAttnFunc.apply(
        q,
        k,
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
    )
