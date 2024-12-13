import torch
from typing import Mapping, Sequence
from tensordict import TensorDict
from tensordict.tensorclass import is_tensorclass

def check_finite(t, name, warn=False):

  d = count_nonfinite(t, name, warn)
  if len(d) > 0:
    if warn:
      print(f'Non-finite entries: {d}')
    else:
      raise ValueError(f'Non-finite entries: {d}')
  
  

def count_nonfinite(t, name, warn=False) -> dict:
  d = {}

  if isinstance(t, torch.Tensor):
    d[name] = (~torch.isfinite(t)).sum()
    if t.grad is not None:
      d[f'{name}.grad'] = count_nonfinite(t.grad, f'{name}.grad', warn)
    return d
  
  if isinstance(t, Sequence) and not isinstance(t, str):
    d = {}
    for i, v in enumerate(t):
      d.update(count_nonfinite(v, f'{name}[{i}]', warn))
    return d
  
  if isinstance(t, Mapping):
    d = {}
    for k, v in t.items():
      d.update(count_nonfinite(v, f'{name}.{k}', warn))
    return d

  if is_tensorclass(t):
    for k, v in t.items():
      d.update(count_nonfinite(v, f'{name}.{k}', warn))
    return d

  return {}
