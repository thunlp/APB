import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_varlen_forward
from .utils import RingComm, update_out_and_lse, get_default_args, All2AllComm


def star_prefill_flash_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):

    out = None
    lse = None

    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update(
        {
            "q": q[0],
            "k": k[0],
            "v": v[0],
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": max_seqlen,
            "max_seqlen_k": max_seqlen,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size": window_size,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
    )
    # if dist.get_rank() == 2:
    #     breakpoint()
    # dist.barrier()
    block_out, _, _, _, _, _, _, _ = _flash_attn_varlen_forward(**params)

    out = block_out.to(q.dtype)
    return out

class StarPrefillFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
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
        out = star_prefill_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
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


def star_prefill_flash_attn_qkvpacked_varlen_func(
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
    return StarPrefillFlashAttnVarlenFunc.apply(
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


def star_prefill_flash_attn_kvpacked_varlen_func(
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
    return StarPrefillFlashAttnVarlenFunc.apply(
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


def star_prefill_flash_attn_varlen_func(
    q,
    k,
    v,
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
    return StarPrefillFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
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
