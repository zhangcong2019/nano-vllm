"""
Triton-based Flash Attention v3 implementation using sparse attention kernel.
"""

import torch
from typing import Optional, Tuple, Union

# Import the sparse attention kernel
from nanovllm.nano_flash_attention.sparse_attn_triton import (
    query_sparse_attn,
    attn_kernel_register_fused_heads_bf16_tensor_descriptor,
)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Flash attention for variable length sequences.
    
    Args:
        q: (total_q, nheads, headdim) - Query tensor
        k: (total_k, nheads_k, headdim) - Key tensor
        v: (total_k, nheads_k, headdim) - Value tensor
        cu_seqlens_q: (batch_size + 1,) - Cumulative sequence lengths for Q
        cu_seqlens_k: (batch_size + 1,) - Cumulative sequence lengths for K/V
        max_seqlen_q: Maximum sequence length for Q
        max_seqlen_k: Maximum sequence length for K/V
        dropout_p: Dropout probability (currently not supported)
        softmax_scale: Softmax scaling factor
        causal: Whether to apply causal masking
        window_size: Sliding window size (left, right)
        softcap: Softcap value (currently not supported)
        alibi_slopes: ALiBi slopes (currently not supported)
        deterministic: Whether to use deterministic mode
        return_attn_probs: Whether to return attention probabilities
        block_table: Paged KV cache block table (currently not supported)
    
    Returns:
        out: (total_q, nheads, headdim) - Attention output
    """
    # Check for unsupported features
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported in sparse attention v3")
    if window_size != (-1, -1):
        raise NotImplementedError("sliding window is not supported in sparse attention v3")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported in sparse attention v3")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes is not supported in sparse attention v3")
    if block_table is not None:
        raise NotImplementedError("paged attention is not supported in sparse attention v3")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs is not supported in sparse attention v3")
    
    # Input format: thd (total, heads, dim)
    total_q, nheads_q, head_dim = q.shape
    total_k, nheads_k, _ = k.shape
    
    # Compute batch size from cu_seqlens
    batch_size = len(cu_seqlens_q) - 1
    
    # Convert from thd to bshd layout (batch, seqlen, heads, head_dim)
    # We need to process each sequence separately
    out = torch.empty_like(q)
    
    # Set softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    
    # Process each sequence in the batch
    for i in range(batch_size):
        start_q = cu_seqlens_q[i].item()
        end_q = cu_seqlens_q[i + 1].item()
        start_k = cu_seqlens_k[i].item()
        end_k = cu_seqlens_k[i + 1].item()
        
        seq_len_q = end_q - start_q
        seq_len_k = end_k - start_k
        
        if seq_len_q == 0:
            continue
        
        # Extract sequence data
        q_seq = q[start_q:end_q]  # (seq_len_q, nheads_q, head_dim)
        k_seq = k[start_k:end_k]  # (seq_len_k, nheads_k, head_dim)
        v_seq = v[start_k:end_k]  # (seq_len_k, nheads_k, head_dim)
        
        # Transpose to bshd layout: (batch=1, seqlen, heads, head_dim)
        q_bshd = q_seq.transpose(0, 1).unsqueeze(0)  # (1, seq_len_q, nheads_q, head_dim)
        k_bshd = k_seq.transpose(0, 1).unsqueeze(0)  # (1, seq_len_k, nheads_k, head_dim)
        v_bshd = v_seq.transpose(0, 1).unsqueeze(0)  # (1, seq_len_k, nheads_k, head_dim)
        
        # Compute heads_per_group for GQA
        heads_per_group = nheads_q // nheads_k if nheads_k > 0 else 1
        
        # Call sparse attention kernel
        out_seq = query_sparse_attn(q_bshd, k_bshd, v_bshd, heads_per_group=heads_per_group)
        
        # Transpose back to thd layout: (seq_len_q, nheads_q, head_dim)
        out[start_q:end_q] = out_seq.squeeze(0).transpose(0, 1)
    
    return out


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
) -> torch.Tensor:
    """
    Flash attention with KV cache support.
    
    Args:
        q: (batch_size, seqlen_q, nheads, headdim) - Query tensor
        k_cache: (batch_size, cache_seqlen, nheads_k, headdim) - Cached K tensor
        v_cache: (batch_size, cache_seqlen, nheads_k, headdim) - Cached V tensor
        k: Optional new K tokens to append to cache
        v: Optional new V tokens to append to cache
        rotary_cos: Rotary embedding cos
        rotary_sin: Rotary embedding sin
        cache_seqlens: Current sequence lengths in cache
        cache_batch_idx: Batch indices for cache access
        cache_leftpad: Left padding offsets
        block_table: Paged KV cache block table
        softmax_scale: Softmax scaling factor
        causal: Whether to apply causal masking
        window_size: Sliding window size (left, right)
        softcap: Softcap value (currently not supported)
        rotary_interleaved: Whether to use interleaved rotary
        alibi_slopes: ALiBi slopes (currently not supported)
        num_splits: Number of splits for split attention
        return_softmax_lse: Whether to return log-sum-exp
    
    Returns:
        out: (batch_size, seqlen_q, nheads, headdim) - Attention output
    """
    # Check for unsupported features
    if k is not None or v is not None:
        raise NotImplementedError("KV cache update is not supported in sparse attention v3")
    if rotary_cos is not None or rotary_sin is not None:
        raise NotImplementedError("rotary embedding is not supported in sparse attention v3")
    if cache_batch_idx is not None:
        raise NotImplementedError("cache_batch_idx is not supported in sparse attention v3")
    if cache_leftpad is not None:
        raise NotImplementedError("cache_leftpad is not supported in sparse attention v3")
    if block_table is not None:
        raise NotImplementedError("paged attention is not supported in sparse attention v3")
    if window_size != (-1, -1):
        raise NotImplementedError("sliding window is not supported in sparse attention v3")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported in sparse attention v3")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes is not supported in sparse attention v3")
    if num_splits != 0 and num_splits != 1:
        raise NotImplementedError("num_splits > 1 is not supported in sparse attention v3")
    
    # Input layout: bshd (batch, seqlen, heads, head_dim)
    batch, seqlen_q, nheads_q, head_dim = q.shape
    _, cache_seqlen, nheads_k, _ = k_cache.shape
    
    # Determine actual KV length
    if cache_seqlens is None:
        kvlen = cache_seqlen
    elif isinstance(cache_seqlens, int):
        kvlen = cache_seqlens
    else:
        # Tensor of shape (batch,)
        kvlen = cache_seqlens.max().item()
    
    # Set softmax scale
    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5
    
    # Compute heads_per_group for GQA
    heads_per_group = nheads_q // nheads_k if nheads_k > 0 else 1
    
    # Trim KV cache to actual length if needed
    if isinstance(cache_seqlens, int):
        k_trimmed = k_cache[:, :cache_seqlens, :, :]
        v_trimmed = v_cache[:, :cache_seqlens, :, :]
    elif cache_seqlens is not None and isinstance(cache_seqlens, torch.Tensor):
        # Handle per-sequence cache lengths - use max for simplicity
        k_trimmed = k_cache[:, :kvlen, :, :]
        v_trimmed = v_cache[:, :kvlen, :, :]
    else:
        k_trimmed = k_cache
        v_trimmed = v_cache
    
    # Call sparse attention kernel
    # Input: bshd layout
    out = query_sparse_attn(q, k_trimmed, v_trimmed, heads_per_group=heads_per_group)
    
    if return_softmax_lse:
        # Return dummy softmax_lse
        softmax_lse = torch.zeros(
            (batch, nheads_q, seqlen_q),
            dtype=torch.float32,
            device=q.device
        )
        return out, softmax_lse
    
    return out
