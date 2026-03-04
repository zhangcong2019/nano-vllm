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
    
    # Handle variable length sequences using cu_seqlens
    outputs = []
    
    # Convert to list if tensor
    if isinstance(cu_seqlens_q, torch.Tensor):
        cu_seqlens_q = cu_seqlens_q.tolist()
    if isinstance(cu_seqlens_k, torch.Tensor):
        cu_seqlens_k = cu_seqlens_k.tolist()
    
    for i in range(len(cu_seqlens_q) - 1):
        start_q = cu_seqlens_q[i]
        end_q = cu_seqlens_q[i + 1]
        start_k = cu_seqlens_k[i]
        end_k = cu_seqlens_k[i + 1]
        
        q_i = q[start_q:end_q]
        k_i = k[start_k:end_k]
        v_i = v[start_k:end_k]
        
        # Reshape to SDPA format
        q_i = q_i.transpose(0, 1).unsqueeze(0)
        k_i = k_i.transpose(0, 1).unsqueeze(0)
        v_i = v_i.transpose(0, 1).unsqueeze(0)
        
        # Apply attention
        out_i = F.scaled_dot_product_attention(
            q_i, k_i, v_i,
            scale=softmax_scale,
            is_causal=causal,
        )
        
        # Reshape back
        out_i = out_i.squeeze(0).transpose(0, 1)
        outputs.append(out_i)
    
    # Concatenate outputs
    out = torch.cat(outputs, dim=0)
    
    return out


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    softmax_scale: float = None,
) -> torch.Tensor:
    """
    Correct SDPA-based implementation of flash_attn_with_kvcache (decode stage).

    Args:
        q:              [batch, 1, num_heads, head_dim]
        k_cache:        [num_blocks, block_size, num_kv_heads, head_dim]
        v_cache:        [num_blocks, block_size, num_kv_heads, head_dim]
        cache_seqlens:  [batch]  (context length per sample)
        block_table:    [batch, max_num_blocks_per_seq] (logical → physical block mapping)

    Returns:
        out: [batch, num_heads, head_dim]
    """

    device = q.device
    dtype = q.dtype

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    batch_size, _, num_heads, head_dim = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]

    outputs = []

    for b in range(batch_size):
        context_len = int(cache_seqlens[b])

        if context_len == 0:
            outputs.append(
                torch.zeros(num_heads, head_dim, device=device, dtype=dtype)
            )
            continue

        # 1. Determine how many blocks are needed
        num_blocks_needed = (context_len + block_size - 1) // block_size

        # 2. Gather physical blocks via block_table
        physical_block_ids = block_table[b, :num_blocks_needed]

        k_blocks = k_cache[physical_block_ids]  # [num_blocks, block_size, kv_heads, dim]
        v_blocks = v_cache[physical_block_ids]

        # 3. Flatten blocks
        k_flat = k_blocks.reshape(-1, num_kv_heads, head_dim)
        v_flat = v_blocks.reshape(-1, num_kv_heads, head_dim)

        # 4. Truncate to true context length
        k_flat = k_flat[:context_len]
        v_flat = v_flat[:context_len]

        # 5. Handle GQA (expand kv heads if needed)
        if num_heads != num_kv_heads:
            rep = num_heads // num_kv_heads
            k_flat = k_flat.repeat_interleave(rep, dim=1)
            v_flat = v_flat.repeat_interleave(rep, dim=1)

        # 6. Convert to SDPA layout
        # q: [1, num_heads, 1, head_dim]
        q_b = q[b].permute(1, 0, 2).unsqueeze(0)

        # k/v: [1, num_heads, context_len, head_dim]
        k_b = k_flat.permute(1, 0, 2).unsqueeze(0)
        v_b = v_flat.permute(1, 0, 2).unsqueeze(0)

        # 7. Decode attention
        # IMPORTANT: is_causal=False
        out = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )

        # [1, num_heads, 1, dim] → [num_heads, dim]
        out = out.squeeze(0).squeeze(1)

        outputs.append(out)

    # 8. Stack results
    result = torch.stack(outputs, dim=0)  # [batch, num_heads, head_dim]

    return result
