from beartype.typing import Tuple
import torch
from torch.utils.cpp_extension import load
from pathlib import Path

sources = [str(Path(__file__).parent  / filename)
            for filename in 
          ["full_cumsum.cu", "radix_sort_pairs.cu", "segmented_sort_pairs.cu", "module.cpp"]]

cuda_lib = load("cuda_lib", sources=sources, verbose=True)

def check_cuda(name, arg):
  assert arg.is_cuda, f"{name}: device must be a cuda device, got {arg.device}"


def full_cumsum(x:torch.Tensor) -> Tuple[torch.Tensor, int]:
  check_cuda("full_cumsum", x)

  if x.shape[0] == 0:
    return x.new_zeros((1, )), 0
  
  out = x.new_empty((x.shape[0] + 1, *x.shape[1:]))

  total = cuda_lib.full_cumsum(x, out)
  return out, total

segmented_sort_pairs = cuda_lib.segmented_sort_pairs
def radix_sort_pairs(keys:torch.Tensor, values:torch.Tensor, start_bit=0, end_bit=None):
  check_cuda("keys", keys)
  check_cuda("values", values)

  if end_bit is None:
    end_bit = -1

  return cuda_lib.radix_sort_pairs(keys, values, start_bit, end_bit)


def radix_argsort(keys:torch.Tensor):
  idx = torch.arange(keys.shape[0], dtype=torch.int32, device=keys.device)
  _, idx = radix_sort_pairs(keys, idx)
  return idx


__all__ = ["full_cumsum", "radix_sort_pairs", "segmented_sort_pairs", "radix_argsort"]


if __name__ == "__main__":
  k = torch.randint(100, (20,), dtype=torch.int32, device="cuda")
  v = torch.arange(20, dtype=torch.int32, device="cuda")

  start_offsets = torch.tensor([0, 8, 16], dtype=torch.long, device="cuda")
  end_offsets = torch.tensor([8, 16, 20], dtype=torch.long, device="cuda")

  k1, v1 = segmented_sort_pairs(k, v, start_offsets, end_offsets)
  print(k1)
  print(v1)

  k2, v2 = radix_sort_pairs(k.long(), v)
  print(k2, k2.dtype)
  print(v2)

  # y = full_cumsum(x)
  # print(y)

  # print(torch.cumsum(x, dim=0))
