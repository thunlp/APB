import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
# from flash_attn_lring.flash_attn_interface import _flash_attn_varlen_forward as lring_fa_func
from flash_attn_apb.flash_attn_interface import _flash_attn_varlen_forward as lring_fa_func
from .utils import RingComm, update_out_and_lse, get_default_args, All2AllComm
from torch.nn.attention.flex_attention import flex_attention

sparsity = 0.1
block_size = 1024 * 128 / 8
blk_sz = 1
# blk_sz = 1024

attn_pool = 128
stabilizers = 256
flex_attention = torch.compile(flex_attention)

def lring_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: torch.Tensor,
    cis: torch.Tensor,
    query_len: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # comm = RingComm(process_group)
    _comm = All2AllComm(process_group)

    out = None
    lse = None
    
    bsz, seq_len, head_num, head_dim = k.shape
    
    # fake_anchor_len = seq_len // 2
    fake_anchor_len = 4096
    anchor_len = fake_anchor_len if _comm.rank > 0 else 0
    
    prv_len = 0
    score = cis.reshape(-1, max_seqlen, cis.shape[-1])
    bsz, seq_len = score.shape[:2]
    # score = torch.randn((bsz, seq_len, head_num), device=k.device)

    
    score[:, :fake_anchor_len, :] = torch.finfo(score.dtype).min
    
    select_num = 2048

    k = k.reshape(-1, max_seqlen, k.shape[-2], k.shape[-1])
    v = v.reshape(-1, max_seqlen, v.shape[-2], v.shape[-1])
    
    indices = torch.topk(score, k=select_num, dim=1).indices
    indices = indices.unsqueeze(-1).repeat(1, 1, 1, head_dim)
    
    sel_k = torch.gather(k, 1, indices)
    sel_v = torch.gather(v, 1, indices)
    
    
    
    sel = torch.cat([sel_k, sel_v], dim=-1)
    _comm.all2all(sel)
    
    _comm.wait()
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
    
    

    
    cu_seqlens_k = torch.tensor([k.shape[1] * i for i in range(k.shape[0] + 1)], dtype=torch.int, device=k.device)
    max_seqlen_k = k.shape[1]
    k = k.reshape(1, -1, head_num, head_dim)
    v = v.reshape(1, -1, head_num, head_dim)
    
    # block_pos = torch.tensor(
    #     [anchor_len, prv_len] + [0] * (k.shape[1] - 2), device=q.device
    # ).int()
    
    block_out, _, _, _, _, _, _, _ = lring_fa_func(  
        q=q[0],
        k=k[0],
        v=v[0],
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0,
        softmax_scale=(q.shape[-1]**(-1/2)),
        causal=True,
        return_softmax=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        # block_pos=block_pos 
        anchor_len=anchor_len,
        prv_len=prv_len,
    )
    
    

    
    out = block_out.to(q.dtype)
    return out

class LRingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
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
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out = lring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            cis,
            query_len,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
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


def lring_flash_attn_qkvpacked_varlen_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return LRingFlashAttnVarlenFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def lring_flash_attn_kvpacked_varlen_func(
    q,
    kv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return LRingFlashAttnVarlenFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )


def lring_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
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
):
    return LRingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
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
    )
