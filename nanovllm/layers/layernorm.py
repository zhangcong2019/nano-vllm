"""
Triton implementation of RMSNorm (as used in vLLM) — optimised.

Optimisations applied:
  1. @triton.autotune with ROWS_PER_PROG, num_warps, num_stages grid-search:
       - ROWS_PER_PROG: each CTA processes multiple rows, loading the weight
         tensor W once and reusing it across rows.  This cuts repeated L2
         weight reads — for M=16 384 and ROWS_PER_PROG=8 the W traffic
         drops from 16 384×16 KB to 2 048×16 KB.
       - num_stages > 1: Triton SW pipeline prefetches row i+1 data
         while row i is being computed, hiding DRAM latency.
       - num_warps: controls occupancy per SM for the optimal compute/
         latency-hide trade-off.
       - Autotune key is ["N", "ELEM_BYTES"] only — NOT M.  In LLM
         inference M changes every call (decode steps, prefill chunks),
         so including it in the key would trigger a full 32-config
         benchmark on every unique batch shape, destroying throughput.
         The winning config depends only on hidden size and dtype, both
         of which are fixed for a given model.
  2. IS_EVEN constexpr: eliminates per-element masking when N is a power
     of two (all standard LLM hidden sizes), enabling 128-bit vectorised
     loads/stores throughout.
  3. inv_N float arg: float multiply (x * inv_N) replaces integer divide
     (x / N) — cheaper and FMA-friendly on GPU.
  4. Grid lambda: reads the autotuned ROWS_PER_PROG to compute the correct
     launch grid — no wasted/excess CTAs.

Compatible with NVIDIA and Intel GPUs via OpenAI Triton.
"""

import torch
from torch import nn
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Autotune configs:
#   ROWS_PER_PROG in {1,2,4,8} × num_warps in {4,8,16,32} × num_stages in
#   {1,2} = 32 configs per kernel; measured once per unique N and cached.
# ---------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({"ROWS_PER_PROG": rpp}, num_warps=nw, num_stages=ns)
    for rpp in [1, 2, 4, 8]
    for nw  in [4, 8, 16, 32]
    for ns  in [1, 2]
]


# ---------------------------------------------------------------------------
# standard RMSNorm forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "ELEM_BYTES"])
@triton.jit
def _rms_norm_fwd_kernel(
    X,                          # [M, N]  input
    W,                          # [N]     fp32 scale weights
    Y,                          # [M, N]  output
    stride,                     # row stride of X / Y (in elements)
    M,                          # number of rows
    N,                          # hidden size
    eps,
    inv_N,                      # 1.0 / N  (precomputed on CPU)
    ELEM_BYTES,                 # x.element_size() — drives dtype-aware autotune
    BLOCK_SIZE:    tl.constexpr,  # next_power_of_2(N)
    IS_EVEN:       tl.constexpr,  # N == BLOCK_SIZE → no mask needed
    ROWS_PER_PROG: tl.constexpr,  # rows per CTA (from autotune)
):
    prog_id = tl.program_id(0)
    cols    = tl.arange(0, BLOCK_SIZE)

    # -----------------------------------------------------------------------
    # Load W once per CTA — shared across ROWS_PER_PROG rows.
    # SW pipeline (num_stages > 1) can prefetch W for the next CTA wave
    # while current rows are being computed.
    # -----------------------------------------------------------------------
    if IS_EVEN:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=(cols < N), other=1.0).to(tl.float32)

    # -----------------------------------------------------------------------
    # Static-unrolled row loop: Triton unrolls at compile time so the
    # compiler can SW-pipeline independent loads across iterations when
    # num_stages > 1.
    # -----------------------------------------------------------------------
    for i in tl.static_range(ROWS_PER_PROG):
        row = prog_id * ROWS_PER_PROG + i
        if row < M:
            base = row * stride + cols
            if IS_EVEN:
                x = tl.load(X + base).to(tl.float32)
            else:
                x = tl.load(X + base, mask=(cols < N), other=0.0).to(tl.float32)

            var  = tl.sum(x * x, axis=0) * inv_N
            rstd = tl.rsqrt(var + eps)
            y    = x * rstd * w

            if IS_EVEN:
                tl.store(Y + base, y)
            else:
                tl.store(Y + base, y, mask=(cols < N))


# ---------------------------------------------------------------------------
# fused residual-add + RMSNorm forward kernel
# ---------------------------------------------------------------------------

@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["N", "ELEM_BYTES"])
@triton.jit
def _rms_norm_add_fwd_kernel(
    X,                          # [M, N]  input           (read-only)
    Residual,                   # [M, N]  residual input  (read-only)
    ResidualOut,                # [M, N]  residual output = x + r
    W,                          # [N]     fp32 scale weights
    Y,                          # [M, N]  normalised output
    stride,
    M,
    N,
    eps,
    inv_N,
    ELEM_BYTES,                 # x.element_size() — drives dtype-aware autotune
    BLOCK_SIZE:    tl.constexpr,
    IS_EVEN:       tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    prog_id = tl.program_id(0)
    cols    = tl.arange(0, BLOCK_SIZE)

    # Load W once per CTA
    if IS_EVEN:
        w = tl.load(W + cols).to(tl.float32)
    else:
        w = tl.load(W + cols, mask=(cols < N), other=1.0).to(tl.float32)

    for i in tl.static_range(ROWS_PER_PROG):
        row = prog_id * ROWS_PER_PROG + i
        if row < M:
            base = row * stride + cols
            if IS_EVEN:
                x = tl.load(X        + base).to(tl.float32)
                r = tl.load(Residual + base).to(tl.float32)
            else:
                mask = cols < N
                x = tl.load(X        + base, mask=mask, other=0.0).to(tl.float32)
                r = tl.load(Residual + base, mask=mask, other=0.0).to(tl.float32)

            x = x + r   # fused residual add (fp32 precision)

            if IS_EVEN:
                tl.store(ResidualOut + base, x)
            else:
                tl.store(ResidualOut + base, x, mask=(cols < N))

            var  = tl.sum(x * x, axis=0) * inv_N
            rstd = tl.rsqrt(var + eps)
            y    = x * rstd * w

            if IS_EVEN:
                tl.store(Y + base, y)
            else:
                tl.store(Y + base, y, mask=(cols < N))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _block_size(N: int) -> int:
    """Smallest power-of-2 >= N, capped at 65536."""
    return min(triton.next_power_of_2(N), 65536)


# ---------------------------------------------------------------------------
# Module — drop-in replacement for orig_kernel.layernorm.RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x    = x.contiguous().view(-1, x.shape[-1])
        M, N = x.shape
        BS   = _block_size(N)
        y    = torch.empty_like(x)
        # Grid lambda reads ROWS_PER_PROG from the winning autotune config,
        # ensuring exactly ceil(M / ROWS_PER_PROG) CTAs are launched.
        grid = lambda meta: (triton.cdiv(M, meta["ROWS_PER_PROG"]),)
        _rms_norm_fwd_kernel[grid](
            x, self.weight, y,
            x.stride(0), M, N, self.eps, 1.0 / N,
            x.element_size(),
            BLOCK_SIZE=BS, IS_EVEN=(N == BS),
        )
        return y.view(orig_shape)

    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x        = x.contiguous().view(-1, x.shape[-1])
        residual = residual.contiguous().view(-1, residual.shape[-1])
        M, N     = x.shape
        BS       = _block_size(N)
        y            = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(M, meta["ROWS_PER_PROG"]),)
        _rms_norm_add_fwd_kernel[grid](
            x, residual, residual_out, self.weight, y,
            x.stride(0), M, N, self.eps, 1.0 / N,
            x.element_size(),
            BLOCK_SIZE=BS, IS_EVEN=(N == BS),
        )
        return y.view(orig_shape), residual_out.view(orig_shape)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
