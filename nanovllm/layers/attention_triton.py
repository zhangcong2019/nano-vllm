"""
Triton-based Flash Attention implementation for Intel GPUs.
This replaces the flash_attn package for Intel GPU compatibility.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i, q, K_block_ptr, K_block_slice, head_dim, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Inner attention loop."""
    # load k - shape: [BLOCK_N, head_dim]
    k = tl.load(K_block_ptr)
    # compute qk - shape: [BLOCK_M, BLOCK_N]
    qk = tl.dot(q, k.T)
    
    # apply causal mask
    if BLOCK_N < BLOCK_M:
        qk = tl.where(qk < 0, qk, float("-inf"))
    
    # softmax
    m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
    qk = qk - m_ij[:, None]
    p = tl.exp(qk)
    l_ij = tl.sum(p, axis=1)
    
    # scale and accumulate
    m_i = m_ij
    p = p * (1.0 / tl.exp(m_i - m_ij[:, None]))
    l_i = l_i * tl.exp(m_i - m_ij[:, None]) + l_ij
    
    # update accumulator
    # p: [BLOCK_M, BLOCK_N], v: [BLOCK_N, head_dim] -> [BLOCK_M, head_dim]
    v = tl.load(K_block_ptr + BLOCK_N * head_dim)
    acc += tl.dot(p, v)
    
    return acc, l_i, m_i


@triton.jit
def _attn_fwd_kernel(
    Q, K, V, Out,
    Lse,  # logsumexp for backward
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Forward attention kernel."""
    # batch and head index
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # block row index (over sequence dimension M)
    row_idx = tl.program_id(2)
    row_offset = row_idx * BLOCK_M
    
    # pointers
    Q += batch_idx * stride_qb + head_idx * stride_qh + row_offset * stride_qm
    K += batch_idx * stride_kb + head_idx * stride_kh
    V += batch_idx * stride_vb + head_idx * stride_vh
    Out += batch_idx * stride_ob + head_idx * stride_oh + row_offset * stride_om
    Lse += batch_idx * H * M + head_idx * M + row_offset
    
    # initialize accumulator
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    
    # load q - shape: [BLOCK_M, HEAD_DIM]
    q_ptrs = Q + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    q_mask = (row_offset + tl.arange(0, BLOCK_M)) < M
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    # loop over k/v blocks
    num_blocks = tl.cdiv(N, BLOCK_N)
    for block_idx in range(num_blocks):
        K_block_ptr = K + block_idx * BLOCK_N * stride_kn
        V_block_ptr = V + block_idx * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, BLOCK_N, HEAD_DIM, BLOCK_M, BLOCK_N
        )
    
    # softmax scale
    softmax_scale = 1.0  # TODO: make this configurable
    acc = acc * softmax_scale
    
    # normalize
    l_i = l_i + 1e-8  # numerical stability
    acc = acc / l_i[:, None]
    
    # store output
    out_ptrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=q_mask[:, None])
    
    # store logsumexp for backward
    m_i = m_i + tl.log(l_i)
    tl.store(Lse + tl.arange(0, BLOCK_M), m_i, mask=q_mask)


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
    Triton implementation of flash_attn_varlen_func for Intel GPUs.
    
    Args:
        q: [total_q, num_heads, head_dim] - query tensor
        k: [total_k, num_kv_heads, head_dim] - key tensor
        v: [total_k, num_kv_heads, head_dim] - value tensor
        max_seqlen_q: maximum sequence length for queries
        cu_seqlens_q: cumulative sequence lengths for queries
        max_seqlen_k: maximum sequence length for keys
        cu_seqlens_k: cumulative sequence lengths for keys
        softmax_scale: scaling factor (default: 1/sqrt(head_dim))
        causal: whether to apply causal masking
        block_table: prefix cache block table (not fully supported yet)
    
    Returns:
        out: [total_q, num_heads, head_dim] - attention output
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    # Get dimensions
    total_q, num_heads, head_dim = q.shape
    total_k = k.shape[0]
    
    # For simplicity, handle single sequence case
    # Variable length support would require splitting by cu_seqlens
    B, H, M, N = 1, num_heads, total_q, total_k
    HEAD_DIM = head_dim
    
    # Allocate output
    out = torch.empty_like(q)
    lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=q.device)
    
    # Triton config
    BLOCK_M = 64
    BLOCK_N = 64
    
    grid = (B, H, triton.cdiv(M, BLOCK_M))
    
    _attn_fwd_kernel[grid](
        q, k, v, out, lse,
        q.stride(0), q.stride(1), q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(0), k.stride(2),
        v.stride(0), v.stride(1), v.stride(0), v.stride(2),
        out.stride(0), out.stride(1), out.stride(0), out.stride(2),
        B, H, M, N, HEAD_DIM, BLOCK_M, BLOCK_N,
    )
    
    return out


@triton.jit
def _attn_kvcache_kernel(
    Q, K_cache, V_cache, Out,
    cache_seqlens, block_table,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_oh, stride_om, stride_ok,
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Forward attention kernel with KV cache."""
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    row_offset = row_idx * BLOCK_M
    
    # pointers
    Q += batch_idx * stride_qb + head_idx * stride_qh + row_offset * stride_qm
    Out += batch_idx * stride_ob + head_idx * stride_oh + row_offset * stride_om
    
    # get context length for this query
    context_len = tl.load(cache_seqlens + batch_idx * M + row_offset)
    
    # initialize accumulator
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    
    # load q
    q_ptrs = Q + tl.arange(0, BLOCK_M)[:, None] * stride_qm + tl.arange(0, HEAD_DIM)[None, :] * stride_qk
    q_mask = row_offset + tl.arange(0, BLOCK_M) < M
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    
    # loop over available KV cache blocks
    num_blocks = tl.cdiv(context_len, BLOCK_N)
    for block_idx in range(num_blocks):
        K_block_ptr = K_cache + batch_idx * stride_kb + head_idx * stride_kh + block_idx * BLOCK_N * stride_kn
        V_block_ptr = V_cache + batch_idx * stride_vb + head_idx * stride_vh + block_idx * BLOCK_N * stride_vn
        acc, l_i, m_i = _attn_fwd_inner(
            acc, l_i, m_i, q, K_block_ptr, BLOCK_N, HEAD_DIM, BLOCK_M, BLOCK_N
        )
    
    # apply causal mask if needed
    if context_len < N:
        pass  # causal is handled in inner loop
    
    # softmax scale
    softmax_scale = 1.0
    acc = acc * softmax_scale
    
    # normalize
    l_i = l_i + 1e-8
    acc = acc / l_i[:, None]
    
    # store output
    out_ptrs = Out + tl.arange(0, BLOCK_M)[:, None] * stride_om + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=q_mask[:, None])


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
    Triton implementation of flash_attn_with_kvcache for Intel GPUs.
    
    Args:
        q: [batch_size, num_heads, head_dim] - query tensor (decode step)
        k_cache: [batch_size, num_kv_heads, cache_max_len, head_dim] - cached keys
        v_cache: [batch_size, num_kv_heads, cache_max_len, head_dim] - cached values
        cache_seqlens: context lengths for each query
        block_table: block table for paging (not fully supported)
        softmax_scale: scaling factor
        causal: whether to apply causal masking
    
    Returns:
        out: [batch_size, num_heads, head_dim] - attention output
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    # Get dimensions - q is [batch, num_heads, 1, head_dim] or [batch, num_heads, head_dim]
    if q.dim() == 4:
        q = q.squeeze(2)  # [batch, num_heads, head_dim]
    
    batch_size, num_heads, head_dim = q.shape
    cache_max_len = k_cache.shape[2]
    
    # Allocate output
    out = torch.empty_like(q)
    
    # Triton config
    BLOCK_M = 64
    BLOCK_N = 64
    
    # For decode, M=1 typically
    M = q.shape[1]  # num_heads actually for this case
    N = cache_max_len
    H = 1  # treat as single "head" in grid since we iterate over actual heads
    B = batch_size
    
    grid = (B, 1, 1)  # Simplified grid
    
    _attn_kvcache_kernel[grid](
        q, k_cache, v_cache, out, cache_seqlens, block_table,
        q.stride(0), q.stride(1), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(1), out.stride(2),
        B, H, M, N, HEAD_DIM, BLOCK_M, BLOCK_N,
    )
    
    return out
