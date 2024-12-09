from dataclasses import dataclass
from functools import cache, partial
import types
from beartype import beartype
import torch
from typing import Optional, Tuple

from taichi_splatting.optim import fractional_adam 
from taichi_splatting.optim import fractional_laprop 

from .util import  get_vector_state, get_scalar_state, get_total_weight, flatten_param

@dataclass 
class Group:
  name: str

  param: torch.Tensor
  grad: Optional[torch.Tensor]
  state: dict

  lr: float
  betas: Tuple[float, float]
  eps: float
  mask_lr: Optional[torch.Tensor]
  bias_correction: bool

  type: str

  @property
  def num_points(self):
    return self.param.shape[0]

def make_group(group, state) -> Group:
  n = len(group["params"])  
  assert n == 1, f"expected 1 tensor in group {group['name']}, got {n}"
  params = group["params"][0]

  state = state[params]
  return Group(group["name"], params.view(params.shape[0], -1), params.grad.view(params.shape[0], -1), 
               state, lr=group["lr"], 
               betas=group["betas"], eps=group["eps"], 
               mask_lr=group["mask_lr"],   
               bias_correction=group["bias_correction"],
               type=group["type"])



def weighted_step(group:Group, 
        visible_weight: torch.Tensor,
        visible_indexes: torch.Tensor, 
        total_weight: torch.Tensor,
        module: types.ModuleType,
        basis: Optional[torch.Tensor]=None):
    

  if group.type in ["vector", "local_vector"]:
    avg_exp, avg_exp_sq = get_vector_state(group.state, group.param)
    kernel = module.vector_kernel(betas=group.betas, eps=group.eps, dims=group.param.shape[1], bias_correction=group.bias_correction)
  elif group.type == "scalar":
    avg_exp, avg_exp_sq = get_scalar_state(group.state, group.param)
    kernel = module.scalar_kernel(betas=group.betas, eps=group.eps, bias_correction=group.bias_correction)
  else:
    raise ValueError(f"unknown group type {group.type}")
    
  if group.type == "local_vector":
    assert basis is not None, "basis is required for local_vector optimizer"

    inv_basis = torch.linalg.inv(basis)
    group.grad[visible_indexes] = torch.einsum('bij,bj->bi', inv_basis, group.grad[visible_indexes])

  lr_step = group.param.new_zeros(visible_indexes.shape[0], group.param.shape[1])
  kernel(lr_step, 
        visible_indexes, visible_weight,
        avg_exp, avg_exp_sq, total_weight,
        group.grad, group.lr)

  if group.type == "local_vector":
    lr_step = torch.einsum('bij,bj->bi', basis, lr_step)

  if group.mask_lr is not None:
    lr_step *= group.mask_lr.view(-1).unsqueeze(0)

  return lr_step

def saturate(x:torch.Tensor):
  return 1 - 1/torch.exp(2 *x)


class FractionalOpt(torch.optim.Optimizer):
  
  def __init__(self, kernels:types.ModuleType, param_groups:list[dict], lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    
    assert lr > 0, f"Invalid learning rate: {lr}"
    assert eps > 0, f"Invalid epsilon: {eps}"
    assert 0.0 <= betas[0] < 1.0, f"Invalid beta1: {betas[0]}"
    assert 0.0 <= betas[1] < 1.0, f"Invalid beta2: {betas[1]}"

    defaults = dict(lr=lr, betas=betas, eps=eps, mask_lr=None, type="scalar", bias_correction=bias_correction)  


    self.kernels = kernels
    super().__init__(param_groups, defaults)


  @torch.no_grad()
  def step(self, 
          indexes: torch.Tensor, 
          weight: torch.Tensor, 
          basis: Optional[torch.Tensor]=None):
    
    assert weight.shape == indexes.shape, f"shape mismatch {weight.shape} != {indexes.shape}"

    groups = [make_group(group, self.state) for group in self.param_groups]
    n = groups[0].param.shape[0]

    total_weight = get_total_weight(groups[0].state, n, device=weight.device)
    total_weight[indexes] += weight

    for group in groups:
      if group.grad is None:
        continue
      
      assert group.num_points == n, f"param shape {group.num_points} != {n}"
      lr_step = weighted_step(group, weight, indexes, total_weight, self.kernels, basis)    

      group.param[indexes] -= lr_step * saturate(weight).unsqueeze(1)

class FractionalAdam(FractionalOpt):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    super().__init__(fractional_adam, params, lr, betas, eps, bias_correction)


class FractionalLaProp(FractionalOpt):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    super().__init__(fractional_laprop, params, lr, betas, eps, bias_correction)


class SparseAdam(FractionalOpt):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    super().__init__(fractional_adam, params, lr, betas, eps, bias_correction)

  def step(self, indexes: torch.Tensor, basis: Optional[torch.Tensor]=None):
    weight = torch.ones(indexes.shape[0], device=indexes.device, dtype=torch.float32)
    super().step(indexes, weight, basis)

class SparseLaProp(FractionalOpt):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    super().__init__(fractional_laprop, params, lr, betas, eps, bias_correction)

  def step(self, indexes: torch.Tensor, basis: Optional[torch.Tensor]=None):
    weight = torch.ones(indexes.shape[0], device=indexes.device, dtype=torch.float32)
    super().step(indexes, weight, basis)

