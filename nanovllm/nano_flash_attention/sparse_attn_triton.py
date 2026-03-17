import torch
import triton
import triton.language as tl
import time
import numpy as np

@triton.jit
def attn_kernel_register_fused_heads_bf16(
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
    sm_scale = tl.cast(sm_scale, tl.bfloat16)
    
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_m_id = tl.program_id(2)
    
    start_m = block_m_id * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    qlen_mask = offs_m < qlen
    offs_k = tl.arange(0, BLOCK_K)
    
    head_start = group_id * HEADS_PER_GROUP
    n_rep = num_heads // num_kv_heads
    
    for head_offset in range(HEADS_PER_GROUP):
        head_id = head_start + head_offset
        head_valid = head_id < num_heads
        
        if head_valid:
            q_ptrs = (
                Q_ptr + batch_id * q_stride_batch + head_id * q_stride_head
                + offs_m[:, None] * q_stride_seq + offs_k[None, :] * q_stride_dim
            )
            q = tl.load(q_ptrs, mask=qlen_mask[:, None] & (offs_k[None, :] < head_dim)).to(tl.bfloat16)
            q = q * sm_scale
            
            acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.bfloat16)
            m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.bfloat16)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.bfloat16)
            
            # 简化的 causal mask: 直接使用 kvlen
            max_valid_kv = kvlen
            
            for start_n in tl.range(0, max_valid_kv, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                kvlen_mask = offs_n < kvlen
                
                kv_head_id = head_id // n_rep
                
                k_ptrs = (
                    K_ptr + batch_id * k_stride_batch + kv_head_id * k_stride_head
                    + offs_n[:, None] * k_stride_seq + offs_k[None, :] * k_stride_dim
                )
                k = tl.load(k_ptrs, mask=kvlen_mask[:, None] & (offs_k[None, :] < head_dim)).to(tl.bfloat16)
                
                v_ptrs = (
                    V_ptr + batch_id * v_stride_batch + kv_head_id * v_stride_head
                    + offs_n[:, None] * v_stride_seq + offs_k[None, :] * v_stride_dim
                )
                v = tl.load(v_ptrs, mask=kvlen_mask[:, None] & (offs_k[None, :] < head_dim)).to(tl.bfloat16)
                
                scores = tl.dot(q, tl.trans(k))
                
                # Causal mask: K 位置 >= Q 位置时置为 -inf
                causal_mask = (start_m + offs_n)[None, :] >= (start_m + offs_m)[:, None]
                scores = tl.where(causal_mask, float('-inf'), scores)
                
                m_ij = tl.max(scores, axis=1)
                m_ij_new = tl.maximum(m_i, m_ij)
                
                p = tl.math.exp(scores - m_ij_new[:, None]).to(tl.bfloat16)
                l_ij = tl.sum(p, axis=1)
                
                alpha = tl.math.exp(m_i - m_ij_new).to(tl.bfloat16)
                acc_out = acc_out * alpha[:, None]
                acc_out = acc_out + tl.dot(p, v).to(tl.bfloat16)
                
                m_i = m_ij_new.to(tl.bfloat16)
                l_i = l_i * alpha + l_ij
            
            acc_out = acc_out / l_i[:, None]
            
            o_ptrs = (
                O_ptr + batch_id * o_stride_batch + head_id * o_stride_head
                + offs_m[:, None] * o_stride_seq + offs_k[None, :] * o_stride_dim
            )
            
            tl.store(o_ptrs, acc_out, mask=qlen_mask[:, None] & (offs_k[None, :] < head_dim))

@triton.jit
def attn_kernel_register_fused_heads_bf16_tensor_descriptor(
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
    """bf16 version with tensor descriptor, optimized for xpu"""

    sm_scale = tl.cast(sm_scale, tl.bfloat16)
    
    batch_id = tl.program_id(0)
    group_id = tl.program_id(1)
    block_m_id = tl.program_id(2)
    
    start_m = block_m_id * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    qlen_mask = offs_m < qlen
    
    head_start = group_id * HEADS_PER_GROUP
    n_rep = num_heads // num_kv_heads
    
    for head_offset in range(HEADS_PER_GROUP):
        head_id = head_start + head_offset
        head_valid = head_id < num_heads
        
        if head_valid:
            # Create 2D tensor descriptors for Q (per head)
            q_base_ptr = Q_ptr + batch_id * q_stride_batch + head_id * q_stride_head
            Q_desc = tl.make_tensor_descriptor(
                q_base_ptr,
                shape=[qlen, head_dim],
                strides=[q_stride_seq, q_stride_dim],
                block_shape=[BLOCK_M, BLOCK_K]
            )
            
            # Load Q using tensor descriptor
            q = tl.load_tensor_descriptor(
                Q_desc,
                offsets=[start_m.to(tl.int32), 0],
            ).to(tl.bfloat16)
            q = q * sm_scale
            
            acc_out = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.bfloat16)
            m_i = tl.full((BLOCK_M,), value=float('-inf'), dtype=tl.bfloat16)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.bfloat16)
            
            # 简化的 causal mask: 直接使用 kvlen
            max_valid_kv = kvlen
            
            kv_head_id = head_id // n_rep
            
            # Create 2D tensor descriptors for K, V (per kv head)
            k_base_ptr = K_ptr + batch_id * k_stride_batch + kv_head_id * k_stride_head
            K_desc = tl.make_tensor_descriptor(
                k_base_ptr,
                shape=[kvlen, head_dim],
                strides=[k_stride_seq, k_stride_dim],
                block_shape=[BLOCK_N, BLOCK_K]
            )
            
            v_base_ptr = V_ptr + batch_id * v_stride_batch + kv_head_id * v_stride_head
            V_desc = tl.make_tensor_descriptor(
                v_base_ptr,
                shape=[kvlen, head_dim],
                strides=[v_stride_seq, v_stride_dim],
                block_shape=[BLOCK_N, BLOCK_K]
            )
            
            o_base_ptr = O_ptr + batch_id * o_stride_batch + head_id * o_stride_head
            O_desc = tl.make_tensor_descriptor(
                o_base_ptr,
                shape=[qlen, head_dim],
                strides=[o_stride_seq, o_stride_dim],
                block_shape=[BLOCK_M, BLOCK_K]
            )
            
            for start_n in tl.range(0, max_valid_kv, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                
                # Load K using tensor descriptor
                k = tl.load_tensor_descriptor(
                    K_desc,
                    offsets=[start_n.to(tl.int32), 0],
                ).to(tl.bfloat16)
                
                # Load V using tensor descriptor
                v = tl.load_tensor_descriptor(
                    V_desc,
                    offsets=[start_n.to(tl.int32), 0],
                ).to(tl.bfloat16)
                
                scores = tl.dot(q, tl.trans(k))                
                # Causal mask: K 位置 >= Q 位置时置为 -inf
                q_pos = offs_m
                k_pos = offs_n

                causal_mask = k_pos[None, :] > q_pos[:, None]
                scores = tl.where(causal_mask, float('-inf'), scores)
                
                m_ij = tl.max(scores, axis=1)
                m_ij_new = tl.maximum(m_i, m_ij)
                
                p = tl.math.exp(scores - m_ij_new[:, None]).to(tl.bfloat16)
                l_ij = tl.sum(p, axis=1)
                
                alpha = tl.math.exp(m_i - m_ij_new).to(tl.bfloat16)
                acc_out = acc_out * alpha[:, None]
                acc_out = acc_out + tl.dot(p, v).to(tl.bfloat16)
                
                m_i = m_ij_new.to(tl.bfloat16)
                l_i = l_i * alpha + l_ij
            
            acc_out = acc_out / l_i[:, None]

            # Store O using tensor descriptor
            tl.store_tensor_descriptor(
                O_desc,
                offsets=[start_m.to(tl.int32), 0],
                value=acc_out.to(tl.bfloat16),
            )


def query_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_per_group: int = 1,
) -> torch.Tensor:
    """
    batch, num_attention_heads, qlen, head_dim = q.shape
    batch_k, num_key_value_heads, kvlen, _ = k.shape
    """
    batch, num_attention_heads, qlen, head_dim = q.shape
    batch_k, num_key_value_heads, kvlen, _ = k.shape
    
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_heads ({num_attention_heads}) must be divisible by num_kv_heads ({num_key_value_heads})"
        )
    
    sm_scale = head_dim ** -0.5
    BLOCK_M = 32
    BLOCK_N = 32
    NUM_STAGES = 1
    BLOCK_K = head_dim
    
    output = torch.empty_like(q)
    num_groups = triton.cdiv(num_attention_heads, heads_per_group)
    grid = (1, num_groups, triton.cdiv(qlen, BLOCK_M))
    
    device = q.device

    kernel_func = attn_kernel_register_fused_heads_bf16_tensor_descriptor if device.type == 'xpu' \
                  else attn_kernel_register_fused_heads_bf16
    
    kernel_func[grid](
        q, k, v, output,
        batch, num_attention_heads, num_key_value_heads, qlen, kvlen, head_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_STAGES=NUM_STAGES, HEADS_PER_GROUP=heads_per_group,
        sm_scale=sm_scale,
        num_stages=NUM_STAGES,
    )
    
    # output = output.transpose(1, 2).contiguous()
    return output

_KERNEL_WARMED_UP = False
def warmup_kernels_device(device: str):
    global _KERNEL_WARMED_UP
    if _KERNEL_WARMED_UP:
        return
    print("[Triton Attention] Warming up all kernel variants...")
    start = time.time()

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()

    q = torch.randn(1, 4, 1, 128, device=device, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 1, 128, device=device, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 1, 128, device=device, dtype=torch.bfloat16)
    _ = query_sparse_attn(q, k, v)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()

    warmup_time = time.time() - start
    _KERNEL_WARMED_UP = True
    print(f"done in {warmup_time:.3f}s")

def warmup_kernels():
    if torch.cuda.is_available():
        warmup_kernels_device("cuda")

    if torch.xpu.is_available():
        warmup_kernels_device("xpu")

warmup_kernels()