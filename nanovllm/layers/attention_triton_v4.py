"""
PyTorch SDPA-based Flash Attention implementation for Intel GPUs.
This replaces the flash_attn package for Intel GPU compatibility.
"""
import torch
from nanovllm.nano_flash_attention.sparse_attn_triton import (
    query_sparse_attn,
    attn_kernel_register_fused_heads_bf16_tensor_descriptor,
)

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

        out_i = query_sparse_attn(
            q_i, k_i, v_i, 2
        )
        print(f"out_i shape sparse: {out_i.shape}")

        # out_i = F.scaled_dot_product_attention(
        #     q_i, k_i, v_i,
        #     scale=softmax_scale,
        #     is_causal=causal,
        # )
        # print(f"out_i shape: {out_i.shape}")
        
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

        out = query_sparse_attn(
            q_b,
            k_b,
            v_b,
            2,
            is_causal=False,
        )
        # print(f"out shape sparse: {out.shape}")
        
        # out = F.scaled_dot_product_attention(
        #     q_b,
        #     k_b,
        #     v_b,
        #     attn_mask=None,
        #     dropout_p=0.0,
        #     is_causal=False,
        #     scale=softmax_scale,
        # )
        # print(f"out shape: {out.shape}")

        # [1, num_heads, 1, dim] → [num_heads, dim]
        out = out.squeeze(0).squeeze(1)

        outputs.append(out)

    # 8. Stack results
    result = torch.stack(outputs, dim=0)  # [batch, num_heads, head_dim]

    return result
