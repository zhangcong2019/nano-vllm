"""
Triton implementation of SiluAndMul activation.

The kernel fuses three steps into a single GPU pass:
  1. Splitting the last dimension in half: x, gate = input.chunk(2, dim=-1)
  2. SiLU:  silu(x) = x * sigmoid(x)
  3. Gating: output = silu(x) * gate

Input  shape: [..., 2*d]
Output shape: [..., d]

Optimisation notes
------------------
* IS_EVEN constexpr: when d is already a power-of-two (all common hidden
  sizes), the per-element mask in tl.load / tl.store is eliminated and
  the compiler emits 128-bit vectorised loads.
* ELEM_BYTES in the autotune key: allows separate configs to be cached
  for fp16 vs fp32 inputs (different memory-bandwidth pressure).

Compatible with NVIDIA and Intel GPUs via OpenAI Triton.
"""

import torch
from torch import nn
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Autotune over num_warps and num_stages only.
# BLOCK_N is chosen per call from next_power_of_2(d) so it is not swept here.
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=nw, num_stages=ns)
    for nw in [4, 8, 16, 32]
    for ns in [1, 2]
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "ELEM_BYTES"])
@triton.jit
def _silu_and_mul_kernel(
    x_ptr,              # input  [M, 2*N]  — contiguous in the last dim
    out_ptr,            # output [M, N]
    M,                  # number of rows   (batch * seq …)
    N,                  # half-width == output columns
    stride_xm,          # row stride of x   (in elements; equals 2*N for contig)
    stride_om,          # row stride of out (in elements; equals   N for contig)
    ELEM_BYTES: tl.constexpr,  # x.element_size() — drives dtype-aware autotuning
    BLOCK_N:    tl.constexpr,  # next_power_of_2(N)
    IS_EVEN:    tl.constexpr,  # N == BLOCK_N  → skip per-element masking
):
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)   # [0, 1, …, BLOCK_N-1]

    # Base pointers for this row
    x_row = row * stride_xm

    # Load first half (x) and second half (gate)
    # When IS_EVEN the mask is statically false → compiler drops it entirely.
    if IS_EVEN:
        xv   = tl.load(x_ptr + x_row + cols)
        gate = tl.load(x_ptr + x_row + N + cols)
    else:
        mask = cols < N
        xv   = tl.load(x_ptr + x_row + cols,          mask=mask, other=0.0)
        gate = tl.load(x_ptr + x_row + N + cols,       mask=mask, other=0.0)

    # Fused SiLU-and-mul: silu(x) * gate
    # tl.sigmoid only accepts fp32/fp64.  Upcast to fp32, compute, then
    # cast the result back to the original element type.  For fp32 inputs
    # the casts are no-ops and the compiler removes them.
    xv_f32   = xv.to(tl.float32)
    gate_f32 = gate.to(tl.float32)
    out_f32  = xv_f32 * tl.sigmoid(xv_f32) * gate_f32
    out      = out_f32.to(xv.dtype)

    # Store result
    o_row = row * stride_om
    if IS_EVEN:
        tl.store(out_ptr + o_row + cols, out)
    else:
        tl.store(out_ptr + o_row + cols, out, mask=mask)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Fused SiLU-and-mul (Triton).

    Parameters
    ----------
    x : Tensor, shape [..., 2*d]
        The last dimension is split in half; the first half is passed through
        SiLU and the result is multiplied by the second half (gate).

    Returns
    -------
    Tensor, shape [..., d]
    """
    orig_shape = x.shape
    d   = orig_shape[-1] // 2            # output half-width
    M   = x.numel() // orig_shape[-1]    # rows = product of all dims except last

    # Flatten to 2-D for the kernel, then reshape back afterwards.
    x_2d = x.reshape(M, orig_shape[-1])
    out  = torch.empty(M, d, dtype=x.dtype, device=x.device)

    # BLOCK_N must cover all d columns; IS_EVEN lets the compiler drop masks.
    BLOCK_N = triton.next_power_of_2(d)
    IS_EVEN = int(d == BLOCK_N)  # pass as int so Triton treats it as constexpr bool

    _silu_and_mul_kernel[(M,)](
        x_2d, out,
        M, d,
        x_2d.stride(0), out.stride(0),
        ELEM_BYTES=x.element_size(),
        BLOCK_N=BLOCK_N,
        IS_EVEN=IS_EVEN,
    )

    return out.reshape(*orig_shape[:-1], d)


class SiluAndMul(nn.Module):
    """Drop-in Triton replacement for orig_kernel.activation.SiluAndMul."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)
