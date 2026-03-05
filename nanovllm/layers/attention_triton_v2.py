"""
Triton-based Flash Attention implementation v2.
Simplified version with float32 precision.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    batch_size, num_heads, num_kv_heads, qlen, kvlen, head_dim,
    q_stride_batch, q_stride_head, q_stride_seq, q_stride_dim,
    k_stride_batch, k_stride_head, k_stride_seq, k_stride_dim,
    v_stride_batch, v_stride_head, v_stride_seq, v_stride_dim,
    o_stride_batch, o_stride_head, o_stride_seq, o_stride_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, HEADS_PER_GROUP: tl.constexpr, 
    sm_scale,
):
    """Triton Flash Attention kernel - float32 version."""
    sm_scale = tl.cast(sm_scale, tl.float32)
    
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_m_id = tl.program_id(2)
    
    start_m = block_m_id * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    qlen_mask = offs_m < qlen
    
    head_start = group_id * HEADS_PER_GROUP
    n_rep = num_heads // num_kv_heads
    
    for head_offset in range(HEADS_PER_GROUP):
        head_id = head_start + head_offset
        head_valid = head_id < num_heads
        
        if head_valid:
            # Load Q
            q_ptrs = (
                Q_ptr + batch_id * q_stride_batch + head_id * q_stride_head
                + offs_m[:, None] * q_stride_seq + offs_k[None, :] * q_stride_dim
            )
            q = tl.load(q_ptrs, mask=qlen_mask[:, None] & (offs_k[None, :] < head_dim)).to(tl.float32)
            q = q * sm_scale
            
            # Initialize accumulator
            acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            
            max_valid_kv = kvlen
            
            # KV loop
            for start_n in tl.range(0, max_valid_kv, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                kvlen_mask = offs_n < kvlen
                
                kv_head_id = head_id // n_rep
                
                # Load K
                k_ptrs = (
                    K_ptr + batch_id * k_stride_batch + kv_head_id * k_stride_head
                    + offs_n[:, None] * k_stride_seq + offs_k[None, :] * k_stride_dim
                )
                k = tl.load(k_ptrs, mask=kvlen_mask[:, None] & (offs_k[None, :] < head_dim)).to(tl.float32)
                
                # Load V
                v_ptrs = (
                    V_ptr + batch_id * v_stride_batch + kv_head_id * v_stride_head
                    + offs_n[:, None] * v_stride_seq + offs_k[None, :] * v_stride_dim
                )
                v = tl.load(v_ptrs, mask=kvlen_mask[:, None] & (offs_k[None, :] < head_dim)).to(tl.float32)
                
                # Compute attention scores
                scores = tl.dot(q, tl.trans(k))
                
                # Causal mask
                causal_mask = (start_m + offs_n)[None, :] >= (start_m + offs_m)[:, None]
                scores = tl.where(causal_mask, float('-inf'), scores)
                
                # Online softmax
                m_ij = tl.max(scores, axis=1)
                m_ij_new = tl.maximum(m_i, m_ij)
                
                p = tl.math.exp(scores - m_ij_new[:, None]).to(tl.float32)
                l_ij = tl.sum(p, axis=1)
                
                alpha = tl.math.exp(m_i - m_ij_new).to(tl.float32)
                acc_out = acc_out * alpha[:, None]
                acc_out = acc_out + tl.dot(p, v).to(tl.float32)
                
                m_i = m_ij_new.to(tl.float32)
                l_i = l_i * alpha + l_ij
            
            # Normalize
            acc_out = acc_out / l_i[:, None]
            
            # Store output
            o_ptrs = (
                O_ptr + batch_id * o_stride_batch + head_id * o_stride_head
                + offs_m[:, None] * o_stride_seq + offs_k[None, :] * o_stride_dim
            )
            
            tl.store(o_ptrs, acc_out, mask=qlen_mask[:, None] & (offs_k[None, :] < head_dim))


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
    Triton implementation of flash_attn_varlen_func.
    
    Args:
        q: [total_q, num_heads, head_dim]
        k: [total_k, num_kv_heads, head_dim]
        v: [total_k, num_kv_heads, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    total_q, num_heads, head_dim = q.shape
    total_k = k.shape[0]
    num_kv_heads = k.shape[1]
    batch = 1
    
    # Handle GQA
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    
    output = torch.empty_like(q)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = head_dim
    NUM_STAGES = 1
    HEADS_PER_GROUP = 1
    
    num_groups = triton.cdiv(num_heads, HEADS_PER_GROUP)
    grid = (batch, num_groups, triton.cdiv(total_q, BLOCK_M))
    
    _attn_kernel[grid](
        q, k, v, output,
        batch, num_heads, num_kv_heads, total_q, total_k, head_dim,
        q.stride(0), q.stride(1), q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(0), k.stride(2),
        v.stride(0), v.stride(1), v.stride(0), v.stride(2),
        output.stride(0), output.stride(1), output.stride(0), output.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES, HEADS_PER_GROUP=HEADS_PER_GROUP,
        sm_scale=softmax_scale,
    )
    
    return output


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
    Triton implementation of flash_attn_with_kvcache.
    
    Args:
        q: [batch, num_heads, head_dim]
        k_cache: [batch, num_kv_heads, cache_len, head_dim]
        v_cache: [batch, num_kv_heads, cache_len, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    batch, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    cache_len = k_cache.shape[2]
    
    # Handle GQA - expand KV heads
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        k_cache = k_cache.repeat_interleave(rep, dim=1)
        v_cache = v_cache.repeat_interleave(rep, dim=1)
    
    output = torch.empty_like(q)
    
    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_K = head_dim
    NUM_STAGES = 1
    HEADS_PER_GROUP = 1
    
    num_groups = triton.cdiv(num_heads, HEADS_PER_GROUP)
    grid = (batch, num_groups, triton.cdiv(num_heads, BLOCK_M))  # Treat num_heads as seq_len
    
    # Adjust strides for kvcache layout
    _attn_kernel[grid](
        q, k_cache, v_cache, output,
        batch, num_heads, num_kv_heads, num_heads, cache_len, head_dim,
        q.stride(0), q.stride(1), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        output.stride(0), output.stride(1), output.stride(1), output.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES, HEADS_PER_GROUP=HEADS_PER_GROUP,
        sm_scale=softmax_scale,
    )
    
    return output
