import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Put the tensor to xpu or cuda
    device = 'cuda' if torch.cuda.is_available() else 'xpu'
    output = torch.empty_like(x, device=device)
    assert x.device.type in ['cuda', 'xpu'] and y.device.type in ['cuda', 'xpu'] and output.device.type in ['cuda', 'xpu']
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

# Set random seed - use CUDA if available, otherwise XPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)
    device = 'cuda'
else:
    torch.xpu.manual_seed(0)
    device = 'xpu'

size = 512
x = torch.rand(size, device=device)
y = torch.rand(size, device=device)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)
