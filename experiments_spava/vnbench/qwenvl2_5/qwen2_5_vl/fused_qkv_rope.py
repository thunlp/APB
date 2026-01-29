import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# fused_qkv_rope_kernel：针对每一个 (b, s, idx_o) 计算 Q/K/V
#
# 1) idx_o ∈ [0, 3*E) 分为三段：
#    - idx_o < E      → 计算 Q，若 dim_idx < R 且 dim_idx 是偶数，就算偶数+奇数对来做 RoPE；
#                      若 dim_idx≥R 就直接存线性结果；若 dim_idx<R 且是奇数维，就不做任何事（留给它的“偶数维”线程来写）。
#    - E ≤ idx_o < 2E  → 计算 K，逻辑同上（也是 RoPE 或原样存）。
#    - 2E ≤ idx_o < 3E → 计算 V，只做线性 + bias 后原样写。
#
# 2) 一定要用 tl.arange(0, E) + tl.dot(...) 来做向量化的 dot-product；
#    同时把 w/b/… 的偏移都写成 “i * stride_w_e + idx_o * stride_w_o” 的通用形式。
# 3) cos/sin 索引统一改成 “dim_idx” 而不是 “dim_pair”。
# ------------------------------------------------------------------------------

@triton.jit
def fused_qkv_rope_kernel(
    x_ptr,      # (float16*) 输入 x，形状 (B, S, E)
    w_ptr,      # (float16*) 权重 w，形状 (E, 3*E)
    b_ptr,      # (float16*) bias b，形状 (3*E,)
    cos_ptr,    # (float16*) cos 值，形状 (S, R)
    sin_ptr,    # (float16*) sin 值，形状 (S, R)
    q_ptr,      # (float16*) q 输出，形状 (B, S, H, D)
    k_ptr,      # (float16*) k 输出，形状 (B, S, H, D)
    v_ptr,      # (float16*) v 输出，形状 (B, S, H, D)
    B, S, E, H, D, R,  # 各种维度参数
    stride_x_b, stride_x_s, stride_x_e,  # x.stride()
    stride_w_e, stride_w_o,              # w.stride()
    stride_b_o,                          # b.stride()[0]
    stride_cos_s, stride_cos_r,          # cos.stride()
    stride_sin_s, stride_sin_r,          # sin.stride()
    stride_q_b, stride_q_s, stride_q_h, stride_q_d,  # q.stride()
    stride_k_b, stride_k_s, stride_k_h, stride_k_d,  # k.stride()
    stride_v_b, stride_v_s, stride_v_h, stride_v_d   # v.stride()
):
    # -----------------------------------
    # 1) 先算出 global program id
    #    我们把 “一共 B * S * (3*E) 个线程” 分配到一个一维 grid 上
    # -----------------------------------
    pid = tl.program_id(0)                                     # pid ∈ [0, B * S * (3*E))
    idx_o = pid % (3 * E)                                      # idx_o ∈ [0, 3*E) → 哪个 Q/K/V 的哪一维
    tmp = pid // (3 * E)
    pos_s = tmp % S                                            # pos_s ∈ [0, S)
    tmp2 = tmp // S
    b_idx = tmp2 % B                                           # b_idx ∈ [0, B)

    # 判断当前 idx_o 属于 Q/K/V
    is_q = idx_o < E
    is_k = (idx_o >= E) & (idx_o < 2 * E)
    is_v = idx_o >= 2 * E

    # 下面把 idx_e、head_idx、dim_idx 都算出来
    # idx_e = idx_o % E   → Q/K 的“行”索引，对应同一个 batch+pos 下，第几维（0..E-1）
    idx_e = idx_o % E
    head_idx = idx_e // D         # 属于哪个 head（0..H-1）
    dim_idx = idx_e % D          # 属于当前 head 的第几维（0..D-1）

    # 先算出 x、cos/sin 的 base pointer
    x_base = x_ptr + b_idx * stride_x_b + pos_s * stride_x_s    # 指向当前 (b_idx, pos_s) 的 x 向量开头
    # 注意：后面要做 tl.arange(0,E) * stride_x_e 来 load x

    # 用于做 dot-product 的 “索引向量”
    i = tl.arange(0, E)  # shape=(E,) —— Triton 会把它当作一个长度为 E 的向量

    # --------------------------
    # v 的分支：只做线性 + bias → 直接存到 v_ptr
    # idx_o ∈ [2*E, 3*E)
    # --------------------------
    if is_v:
        # 先算出偏移：v_off = b_idx * stride_v_b + pos_s * stride_v_s + head_idx * stride_v_h + dim_idx * stride_v_d
        # 但注意：idx_o 对应的是“3E” 里面排到最末尾的那一段 → 它的 idx_e = idx_o % E = head_idx*D + dim_idx，正好再次映射到 (head, dim)
        off_v = b_idx * stride_v_b + pos_s * stride_v_s + head_idx * stride_v_h + dim_idx * stride_v_d

        # 取 w 中对应 (i, idx_o) 的列
        # 通用写法：w[i, idx_o] -> w_ptr + i*stride_w_e + idx_o*stride_w_o
        w_col_ptr = w_ptr + i * stride_w_e + idx_o * stride_w_o   # 这一行构造了一个“长度为 E”的指针向量，后面 tl.load 时会自动扩展
        x_vec     = tl.load(x_base + i * stride_x_e, mask=i < E, other=0.0)  # (E,)
        w_vec     = tl.load(w_col_ptr,               mask=i < E, other=0.0)  # (E,)
        dot_v     = tl.dot(x_vec, w_vec)                           # float16 dot, 结果是 float16

        # bias 也要拿出来
        bias_v = tl.load(b_ptr + idx_o * stride_b_o, mask=idx_o < 3 * E, other=0.0)  # float16

        out_v = dot_v + bias_v
        tl.store(v_ptr + off_v, out_v)

        # v 部分处理完毕，后面的 Q/K 逻辑都不用进来了
        return

    # --------------------------
    # Q 或 K 的分支：需要看 dim_idx 是否 < R 来决定要不要 RoPE
    # --------------------------
    # pre-compute一下：当前 dim_idx 对应的（b,s,head,dim）在 Q 或 K 中的存储偏移
    off_base = (b_idx * stride_q_b +
                pos_s * stride_q_s +
                head_idx * stride_q_h +
                dim_idx * stride_q_d)  # 这是 Q 或 K 在偏移时的 base，后续如果是 K，就把 q_ptr 换成 k_ptr

    # 把 idx_o 映射到“Q 区域” or “K 区域”的相对列索引
    # 例如：若 idx_o < E → 真正的 “Q 列”索引是 idx_o；若 E<=idx_o<2E → “K 列”索引是 idx_o - E
    rel_o = idx_o if is_q else (idx_o - E)  # rel_o ∈ [0, E)
    # 注意：rel_o 的 “行号”=rel_e = rel_o % E，然后 rel_head、rel_dim = 一样的计算

    # rel_e、rel_head、rel_dim 在 Q/K 中会跟上面 is_q/is_k 的 idx_e head_idx dim_idx 一致，
    # 复用第一步的计算，rel_dim==dim_idx，rel_head==head_idx，没有问题。

    # 如果在 RoPE 范围内，且是“偶数层”，就一次性算偶+奇两个 dot+bias，再做旋转
    if dim_idx < R:
        # 针对 RoPE 需要“成对”处理（偶数维 + 奇数维），
        # 只有在 dim_idx 为偶数时，让它负责把“自己(dim_idx)和 dim_idx+1”成对算一次、一起写。
        if (dim_idx % 2) == 0:
            # 先算出“偶数维”的完整偏移和“奇数维”的偏移
            # off_base 对应的是 dim_idx 维；dim_idx+1 维的偏移正好 off_base + stride_q_d
            off_even = off_base
            off_odd  = off_base + stride_q_d   # 对应 dim_idx+1

            # 真正的 idx_o_even = 如果 is_q，那么就是 idx_o；如果 is_k，就要 + E
            idx_o_even = idx_o                              # 即当前线程看到的 idx_o，一定是 Q 或 K 对应的“偶数维”那一维
            idx_o_odd  = idx_o + 1                          # 对应奇数维（列数就 +1）

            # ---- 1) 取 x_vec —— 跟 V 一样，(b,s) 下的长度为 E 的 x
            x_vec = tl.load(x_base + i * stride_x_e, mask=i < E, other=0.0)  # (E,)

            # ---- 2) 取“偶数维”的权重列 w[:, idx_o_even] 以及“奇数维”的 w[:, idx_o_odd]
            w_col_even = tl.load(w_ptr + i * stride_w_e + idx_o_even * stride_w_o,
                                 mask=i < E, other=0.0)  # (E,)
            w_col_odd  = tl.load(w_ptr + i * stride_w_e + idx_o_odd  * stride_w_o,
                                 mask=i < E, other=0.0)  # (E,)

            # ---- 3) 计算 dot_e，dot_o
            dot_even = tl.dot(x_vec, w_col_even)  # float16
            dot_odd  = tl.dot(x_vec, w_col_odd)   # float16

            # ---- 4) 取 bias_e, bias_o（也是 float16）
            bias_even = tl.load(b_ptr + idx_o_even * stride_b_o, mask=idx_o_even < 3 * E, other=0.0)
            bias_odd  = tl.load(b_ptr + idx_o_odd  * stride_b_o, mask=idx_o_odd  < 3 * E, other=0.0)

            # ---- 5) 转到 float32 做 RoPE
            val_e_f32 = tl.cast(dot_even + bias_even, tl.float32)
            val_o_f32 = tl.cast(dot_odd  + bias_odd,  tl.float32)

            # 下标 dim_idx 就是 2*k；cos/sin 整个张量形状是 (S, R)，我们这里直接用 dim_idx
            cos_f32 = tl.cast(
                tl.load(cos_ptr + pos_s * stride_cos_s + dim_idx * stride_cos_r,
                        mask=dim_idx < R, other=0.0), tl.float32)
            sin_f32 = tl.cast(
                tl.load(sin_ptr + pos_s * stride_sin_s + dim_idx * stride_sin_r,
                        mask=dim_idx < R, other=0.0), tl.float32)

            # ---- 6) 真正做旋转
            #
            #   out_e = val_e * cos - val_o * sin
            #   out_o = val_e * sin + val_o * cos
            #
            out_e_f32 = val_e_f32 * cos_f32 - val_o_f32 * sin_f32
            out_o_f32 = val_e_f32 * sin_f32 + val_o_f32 * cos_f32

            # 都 cast 回 float16 存
            out_e_fp16 = tl.cast(out_e_f32, tl.float16)
            out_o_fp16 = tl.cast(out_o_f32, tl.float16)

            # ---- 7) 根据 is_q/is_k 决定存到 q_ptr 还是 k_ptr
            if is_q:
                tl.store(q_ptr + off_even, out_e_fp16)
                tl.store(q_ptr + off_odd,  out_o_fp16)
            else:  # is_k
                tl.store(k_ptr + off_even, out_e_fp16)
                tl.store(k_ptr + off_odd,  out_o_fp16)
        # 若 dim_idx < R 但是是奇数维，就直接 “不做任何事”，留给奇数维对应的偶数维线程去写
        return

    # 走到这里，说明“dim_idx >= R”，不需要做 RoPE
    # 只要把线性 + bias 原样存，就行
    # 下面的代码对 Q/K 都通用，把当前 idx_o 对应的那一维 dot+bias 算出来，再存回
    w_col = tl.load(w_ptr + i * stride_w_e + idx_o * stride_w_o,
                    mask=i < E, other=0.0)  # (E,)
    x_vec = tl.load(x_base + i * stride_x_e, mask=i < E, other=0.0)
    dot_val = tl.dot(x_vec, w_col)                            # float16

    bias_val = tl.load(b_ptr + idx_o * stride_b_o,
                       mask=idx_o < 3 * E, other=0.0)        # float16

    out_qk = dot_val + bias_val
    if is_q:
        tl.store(q_ptr + off_base, out_qk)
    else:  # is_k
        tl.store(k_ptr + off_base, out_qk)


# ------------------------------------------------------------------------------
# Host 端的 Python 接口：把参数打平，调用 kernel
# ------------------------------------------------------------------------------
def fused_qkv_rope(x, w, b, cos, sin, H, D, R):
    """
    x:   (B, S, E)  float16，E=H*D
    w:   (E, 3*E)  float16
    b:   (3*E,)    float16
    cos: (S, R)    float16
    sin: (S, R)    float16

    返回：
      q, k, v 三个张量，形状都是 (B, S, H, D)，dtype=float16
    """
    assert x.is_cuda and w.is_cuda and b.is_cuda and cos.is_cuda and sin.is_cuda
    B, S, E = x.shape
    assert E == H * D, f"E=H*D 必须成立，得到 E={E}, H*D={H*D}"
    assert w.shape == (E, 3 * E)
    assert b.shape == (3 * E,)
    assert cos.shape == (S, R)
    assert sin.shape == (S, R)

    # 准备输出
    q = x.new_empty((B, S, H, D))
    k = x.new_empty((B, S, H, D))
    v = x.new_empty((B, S, H, D))

    # 拿各个张量的 stride
    stride_x_b, stride_x_s, stride_x_e = x.stride()
    stride_w_e, stride_w_o = w.stride()
    stride_b_o = b.stride()[0]
    stride_cos_s, stride_cos_r = cos.stride()
    stride_sin_s, stride_sin_r = sin.stride()
    stride_q_b, stride_q_s, stride_q_h, stride_q_d = q.stride()
    stride_k_b, stride_k_s, stride_k_h, stride_k_d = k.stride()
    stride_v_b, stride_v_s, stride_v_h, stride_v_d = v.stride()

    # 计算需要启动的总线程数
    num_programs = B * S * (3 * E)

    # Triton 要求：grid 必须是一个 tuple of ints
    grid = (num_programs,)

    # 调用 Triton kernel 时，不要再传 num_programs=…
    fused_qkv_rope_kernel[grid](
        x, w, b, cos, sin, q, k, v,
        B, S, E, H, D, R,
        stride_x_b, stride_x_s, stride_x_e,
        stride_w_e, stride_w_o,
        stride_b_o,
        stride_cos_s, stride_cos_r,
        stride_sin_s, stride_sin_r,
        stride_q_b, stride_q_s, stride_q_h, stride_q_d,
        stride_k_b, stride_k_s, stride_k_h, stride_k_d,
        stride_v_b, stride_v_s, stride_v_h, stride_v_d,
    )
    return q, k, v



# ------------------------------------------------------------------------------
# 简单的 main() 做测试
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    B = 2
    S = 16
    H = 4
    D = 32
    E = H * D
    R = 32  # RoPE 只作用在头部的前 R=32 维

    torch.manual_seed(0)
    x = torch.randn((B, S, E), device='cuda', dtype=torch.float16)
    w = torch.randn((E, 3 * E), device='cuda', dtype=torch.float16)
    b = torch.randn((3 * E,), device='cuda', dtype=torch.float16)

    # 生成 RoPE 所需的 cos/sin 张量 (S, R)
    position = torch.arange(S, dtype=torch.float32, device='cuda')[:, None]          # (S, 1)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, R, 2, dtype=torch.float32, device='cuda') / R))  # (R/2,)
    sinusoid_in = position * inv_freq[None, :]  # (S, R/2)
    cos = torch.zeros((S, R), device='cuda', dtype=torch.float16)
    sin = torch.zeros((S, R), device='cuda', dtype=torch.float16)
    cos[:, 0::2] = torch.cos(sinusoid_in).half()
    cos[:, 1::2] = torch.cos(sinusoid_in).half()
    sin[:, 0::2] = torch.sin(sinusoid_in).half()
    sin[:, 1::2] = torch.sin(sinusoid_in).half()

    # 调用我们修复后的 fused qkv+RoPE
    q, k, v = fused_qkv_rope(x, w, b, cos, sin, H, D, R)

    print("q shape:", q.shape)   # 应该是 (2,16,4,32)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    print("q[0,0,0,0]:", q[0, 0, 0, 0].item())
    print("k[0,0,0,0]:", k[0, 0, 0, 0].item())
    print("v[0,0,0,0]:", v[0, 0, 0, 0].item())
