import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse, get_default_args, All2AllComm
import time

def ulysses_prefill_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    breakdown=False,
):
    
    _comm = All2AllComm(process_group)
    world_size = _comm.world_size
    num_heads = q.shape[-2]
    num_kv_heads = k.shape[-2]
        
    if breakdown:
        dist.barrier()
        comm_beg = time.time()
        
    # split q
    q_split_size = num_heads // world_size
    q_split = list(torch.split(q, q_split_size, dim=2))
    q_split = [t.contiguous() for t in q_split]
    # split kv
    kv_split_size = num_kv_heads // world_size
    k_split = list(torch.split(k, kv_split_size, dim=2))
    k_split = [t.contiguous() for t in k_split]
    v_split = list(torch.split(v, kv_split_size, dim=2))
    v_split = [t.contiguous() for t in v_split]
    
    # comm qkv
    _comm.all2all_list(q_split)
    _comm.wait()
    q_list = _comm.data()
    _comm.all2all_list(k_split)
    _comm.wait()
    k_list = _comm.data()
    _comm.all2all_list(v_split)
    _comm.wait()
    v_list = _comm.data()
    
    q = torch.cat(q_list, dim=1)
    k = torch.cat(k_list, dim=1)
    v = torch.cat(v_list, dim=1)
    
    if breakdown:
        dist.barrier()
        comm_end = time.time()
    if breakdown:
        dist.barrier()
        attn_beg = time.time()

    out = None
    lse = None

    params = get_default_args(_flash_attn_forward).copy()
    params.update(
        {
            "q": q,
            "k": k,
            "v": v,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
    )
    block_out, _, _, _, _, _, _, _ = _flash_attn_forward(**params)

    out = block_out.to(q.dtype)
    
    if breakdown:
        dist.barrier()
        attn_end = time.time()
    
    if breakdown:
        dist.barrier()
        comm2_beg = time.time()
    
    # split o
    full_seq_len = out.shape[1]
    o_split_size = full_seq_len // world_size
    o_split = list(torch.split(out, o_split_size, dim=1))
    o_split = [t.contiguous() for t in o_split]
    # comm o
    _comm.all2all_list(o_split)
    _comm.wait()
    o_list = _comm.data()
    out = torch.cat(o_list, dim=-2)
    
    if breakdown:
        dist.barrier()
        comm2_end = time.time()
        
    if breakdown:
        return out, comm_end - comm_beg + comm2_end - comm2_beg, attn_end - attn_beg
    
    return out

class UlyssesPrefillFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        breakdown,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out = ulysses_prefill_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            breakdown=breakdown,
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


def ulysses_prefill_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return UlyssesPrefillFlashAttnFunc.apply(
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
    )


def ulysses_prefill_flash_attn_kvpacked_func(
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
):
    return UlyssesPrefillFlashAttnFunc.apply(
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
    )


def ulysses_prefill_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    breakdown=False,
):
    return UlyssesPrefillFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        breakdown,
    )
