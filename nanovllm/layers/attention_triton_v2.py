import triton
import triton.language as tl
import torch
from typing import Optional, Tuple, Union

@triton.jit
def _attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    seqlen_q, seqlen_k,
    scale,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):

    start_m = tl.program_id(0)
    head_id = tl.program_id(1)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + head_id * stride_qh + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)

    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, seqlen_k, BLOCK_N):

        k_ptrs = K_ptr + (start_n + offs_n)[None, :] * stride_km + head_id * stride_kh + offs_d[:, None] * stride_kd
        v_ptrs = V_ptr + (start_n + offs_n)[:, None] * stride_vm + head_id * stride_vh + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=(start_n + offs_n)[None, :] < seqlen_k, other=0.0)
        v = tl.load(v_ptrs, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)

        scores = tl.dot(q, k) * scale

        if causal:
            q_pos = offs_m
            k_pos = start_n + offs_n
            mask = k_pos[None, :] > q_pos[:, None]
            scores = tl.where(mask, -float("inf"), scores)

        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        alpha = tl.exp(m_i - m_ij)
        l_i = alpha * l_i + l_ij

        acc = acc * alpha[:, None]
        p = p.to(tl.bfloat16)
        acc += tl.dot(p, v)

        m_i = m_ij

    acc = acc / l_i[:, None]

    o_ptrs = O_ptr + offs_m[:, None] * stride_om + head_id * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=offs_m[:, None] < seqlen_q)


def triton_attention(q, k, v, causal, scale):
    seqlen_q, nheads, dim = q.shape
    seqlen_k = k.shape[0]

    out = torch.empty_like(q)

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (
        triton.cdiv(seqlen_q, BLOCK_M),
        nheads,
    )

    _attn_fwd_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        seqlen_q,
        seqlen_k,
        scale,
        causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HEAD_DIM=dim,
    )

    return out


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
    causal: bool = True,
    block_table: torch.Tensor = None,
) -> torch.Tensor:
    """
    Args:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        softmax_scale: scaling factor (default: 1/sqrt(head_dim))
        causal: whether to apply causal masking
        block_table: prefix cache block table (not fully supported yet)
    
    Return:
        out: (total, nheads, headdim).
    """
    total_q, nheads_q, head_dim = q.shape
    total_k, nheads_k, _ = k.shape

    batch_size = cu_seqlens_q.numel() - 1

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    out = torch.empty_like(q)

    for i in range(batch_size):

        start_q = int(cu_seqlens_q[i])
        end_q = int(cu_seqlens_q[i + 1])

        start_k = int(cu_seqlens_k[i])
        end_k = int(cu_seqlens_k[i + 1])

        if end_q == start_q:
            continue

        q_seq = q[start_q:end_q]
        k_seq = k[start_k:end_k]
        v_seq = v[start_k:end_k]

        # GQA support
        if nheads_q != nheads_k:
            rep = nheads_q // nheads_k
            k_seq = k_seq.repeat_interleave(rep, dim=1)
            v_seq = v_seq.repeat_interleave(rep, dim=1)

        out_seq = triton_attention(
            q_seq.contiguous(),
            k_seq.contiguous(),
            v_seq.contiguous(),
            causal,
            softmax_scale,
        )

        out[start_q:end_q] = out_seq

    return out


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor = None,
    block_table: torch.Tensor = None,
    softmax_scale: float = None,
    causal: bool = True,
) -> torch.Tensor:

    """
    FlashAttention-style KV cache attention using Triton kernel.

    Args:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        cache_seqlens: context lengths for each query
        block_table: block table for paging (not fully supported)
        softmax_scale: scaling factor
        causal: whether to apply causal masking
    
    Returns:
        out: (batch_size, seqlen, nheads, headdim).
    """

    B, Sq, Hq, D = q.shape
    _, Sk_max, Hk, _ = k_cache.shape

    device = q.device

    # -------------------------------------------------------
    # Default scale
    # -------------------------------------------------------

    if softmax_scale is None:
        softmax_scale = D ** -0.5

    # -------------------------------------------------------
    # Normalize cache_seqlens
    # -------------------------------------------------------

    if isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (B,), cache_seqlens, dtype=torch.int32, device=device
        )

    if cache_seqlens is None:
        cache_seqlens = torch.zeros(B, dtype=torch.int32, device=device)

    # -------------------------------------------------------
    # Update KV cache with new tokens
    # -------------------------------------------------------

    if k is not None and v is not None:

        new_tokens = k.shape[1]

        for b in range(B):

            start = cache_seqlens[b].item()
            end = start + new_tokens

            k_cache[b, start:end] = k[b]
            v_cache[b, start:end] = v[b]

            cache_seqlens[b] = end

    # -------------------------------------------------------
    # Run Triton attention per batch
    # -------------------------------------------------------

    outputs = []

    for b in range(B):

        seqlen_k = cache_seqlens[b].item()

        if seqlen_k == 0:
            outputs.append(torch.zeros_like(q[b]))
            continue

        q_b = q[b]                           # [Sq, Hq, D]
        k_b = k_cache[b, :seqlen_k]          # [Sk, Hk, D]
        v_b = v_cache[b, :seqlen_k]

        out_b = triton_attention(
            q_b,
            k_b,
            v_b,
            causal=causal,
            scale=softmax_scale,
        )

        outputs.append(out_b)

    out = torch.stack(outputs, dim=0)

    return out

