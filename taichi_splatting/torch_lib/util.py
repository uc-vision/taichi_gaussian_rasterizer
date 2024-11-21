from beartype import beartype
import numpy as np
import torch
from typing import Mapping
from tensordict.tensorclass import is_tensorclass
from tensordict import TensorDict

@beartype
def check_finite(t, name:str, warn:bool=False):
  
  if isinstance(t, torch.Tensor):
    n = (~torch.isfinite(t)).sum()
    if n > 0:
      if warn:
        print(f'Found {n} non-finite values in {name}')
        t[~torch.isfinite(t)] = 0
      else:
        raise ValueError(f'Found {n} non-finite values in {name}')
      
    if isinstance(t, np.ndarray):
      check_finite(torch.from_numpy(t), name, warn)
    
    if t.grad is not None:
      check_finite(t.grad, f'{name}.grad', warn)

  if isinstance(t, Mapping):
    for k, v in t.items():
      check_finite(v, f'{name}.{k}', warn)

  if is_tensorclass(t):
    check_finite(t.to_dict(), name, warn)

  if isinstance(t, TensorDict):
    check_finite(t.to_dict(), name, warn)


