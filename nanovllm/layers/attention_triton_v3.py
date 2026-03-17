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
    max_seqlen_q: int,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_k: int,
    cu_seqlens_k: torch.Tensor,
    softmax_scale: float = None,
    causal: bool = True,
    block_table: torch.Tensor = None,
) -> torch.Tensor:
    """
    Flash attention for variable length sequences.
    
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
    # Check for unsupported features
    if block_table is not None:
        raise NotImplementedError("paged attention is not supported in sparse attention v3")
    
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
        out[:,start_q:end_q,:] = out_seq.squeeze(0).transpose(0, 1)
    
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
    Flash attention with KV cache support.
    
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
    # Check for unsupported features
    if block_table is not None:
        raise NotImplementedError("paged attention is not supported in sparse attention v3")
    
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
