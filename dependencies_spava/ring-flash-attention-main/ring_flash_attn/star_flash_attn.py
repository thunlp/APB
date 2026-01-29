import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .utils import RingComm, update_out_and_lse, get_default_args


def star_flash_attn_forward(
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
):
    block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=False,
        alibi_slopes=alibi_slopes,
        return_softmax= (dropout_p > 0),
    )

    out_gather = [torch.zeros_like(block_out) for _ in range(dist.get_world_size(group=process_group))]
    lse_gather = [torch.zeros_like(block_lse) for _ in range(dist.get_world_size(group=process_group))]
    dist.all_gather(out_gather, block_out, async_op=False, group=process_group)
    dist.all_gather(lse_gather, block_lse, async_op=False, group=process_group)
    # dist.barrier()
    
    # # torch.save(k, f"k_{dist.get_rank()}.pt")
    # # torch.save(v, f"v_{dist.get_rank()}.pt")
    
    
    # if dist.get_rank() == 0:
    #     breakpoint()
    # dist.barrier()
    # # exit(0)
    
    # stacked_lse = torch.stack(lse_gather, dim=0).transpose(-1, -2).unsqueeze(-1).to(torch.float32)
    # se = stacked_lse.exp().sum(dim=0)
    # new_lse = se.log()
    
    # stacked_out = torch.stack(out_gather, dim=0).to(torch.float32)
    # new_out = (torch.exp(stacked_lse - new_lse) * stacked_out).sum(0)
    
   
    # # if dist.get_rank() == 0:
    # #     breakpoint()
    # # dist.barrier() 
    # return new_out.to(q.dtype), new_lse

    # log sum exp

    out = None
    lse = None
    for i in range(len(out_gather)):
        out, lse = update_out_and_lse(out, lse, out_gather[i], lse_gather[i])

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse

class StarFlashAttnFunc(torch.autograd.Function):
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
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = star_flash_attn_forward(
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
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        raise NotImplementedError


def star_flash_attn_qkvpacked_func(
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
    return StarFlashAttnFunc.apply(
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


def star_flash_attn_kvpacked_func(
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
    return StarFlashAttnFunc.apply(
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


def star_flash_attn_func(
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
):
    return StarFlashAttnFunc.apply(
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
    )
