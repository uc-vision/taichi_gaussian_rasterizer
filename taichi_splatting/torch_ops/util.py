from typing import Mapping
import torch

def check_finite(t, name, warn=False):
  

  if isinstance(t, torch.Tensor):
    n = (~torch.isfinite(t)).sum()
    if n > 0:
      if warn:
        print(f'Found {n} non-finite values in {name}')
        t[~torch.isfinite(t)] = 0
      else:
        raise ValueError(f'Found {n} non-finite values in {name}')
    
    if t.grad is not None:
      check_finite(t.grad, f'{name}.grad', warn)

  if isinstance(t, Mapping):
    for k, v in t.items():
      check_finite(v, f'{name}.{k}', warn)
    