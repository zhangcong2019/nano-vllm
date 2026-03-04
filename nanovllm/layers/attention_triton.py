"""
PyTorch SDPA-based Flash Attention implementation for Intel GPUs.
This replaces the flash_attn package for Intel GPU compatibility.
"""

import torch
import torch.nn.functional as F


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
    PyTorch SDPA-based implementation of flash_attn_varlen_func.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    
    # Handle GQA - repeat k/v if needed to match num_heads
    if num_heads != num_kv_heads:
        rep = num_heads // num_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    
    # Reshape for SDPA: [total_q, num_heads, head_dim] -> [batch=1, num_heads, total_q, head_dim]
    q = q.transpose(0, 1).unsqueeze(0)  # [1, num_heads, total_q, head_dim]
    k = k.transpose(0, 1).unsqueeze(0)  # [1, num_heads, total_k, head_dim]
    v = v.transpose(0, 1).unsqueeze(0)  # [1, num_heads, total_k, head_dim]
    
    # Apply attention using PyTorch SDPA
    out = F.scaled_dot_product_attention(
        q, k, v,
        scale=softmax_scale,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=causal,
    )
    
    # Reshape back: [batch, num_heads, total_q, head_dim] -> [total_q, num_heads, head_dim]
    out = out.squeeze(0).transpose(0, 1).contiguous()
    
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
    PyTorch SDPA-based implementation of flash_attn_with_kvcache.
    
    k_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)
    
    # q is [batch, 1, num_heads, head_dim]
    batch_size = q.shape[0]
    num_heads = q.shape[2]
    head_dim = q.shape[3]
    
    # k_cache shape: [num_blocks, block_size, num_kv_heads, head_dim]
    num_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    
    # Get context lengths
    if cache_seqlens is None:
        context_lens = [0] * batch_size
    else:
        context_lens = cache_seqlens.tolist() if isinstance(cache_seqlens, torch.Tensor) else cache_seqlens
    
    outputs = []
    for b in range(batch_size):
        q_b = q[b]  # [1, num_heads, head_dim]
        context_len = context_lens[b] if isinstance(context_lens, list) else int(context_lens)
        
        if context_len > 0:
            # Calculate how many blocks are filled
            num_filled_blocks = (context_len + block_size - 1) // block_size
            filled_len = context_len
            
            # Flatten the filled blocks: [num_filled_blocks, block_size, num_kv_heads, head_dim]
            k_b = k_cache[:num_filled_blocks, :filled_len, :, :].reshape(-1, num_kv_heads, head_dim)
            v_b = v_cache[:num_filled_blocks, :filled_len, :, :].reshape(-1, num_kv_heads, head_dim)
            
            # Handle GQA: repeat k/v to match num_heads
            if num_heads != num_kv_heads:
                rep = num_heads // num_kv_heads
                k_b = k_b.repeat(1, rep, 1)  # [seq_len, num_heads, head_dim]
                v_b = v_b.repeat(1, rep, 1)
            
            # Reshape for SDPA
            seq_len = k_b.shape[0]
            k_b = k_b.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq_len, head_dim]
            v_b = v_b.transpose(0, 1).unsqueeze(0)
            q_b_for_sdpa = q_b.unsqueeze(2)  # [1, num_heads, 1, head_dim]
            
            # Apply attention
            out_b = F.scaled_dot_product_attention(
                q_b_for_sdpa, k_b, v_b,
                scale=softmax_scale,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal,
            )
            
            # Reshape back: [1, num_heads, 1, head_dim] -> [num_heads, head_dim]
            out_b = out_b.squeeze(0).transpose(0, 1).squeeze(1)
        else:
            out_b = torch.zeros(num_heads, head_dim, dtype=torch.float32)
        
        outputs.append(out_b)
    
    # Stack: [batch, num_heads, head_dim]
    result = torch.stack(outputs, dim=0)
    
    # Add seq_len=1 dimension: [batch, 1, num_heads, head_dim]
    result = result.unsqueeze(1)
    
    return result
