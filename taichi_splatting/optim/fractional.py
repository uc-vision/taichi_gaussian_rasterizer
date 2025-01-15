from dataclasses import dataclass
import types
import torch
from typing import Optional, Tuple

from taichi_splatting.optim import fractional_adam 
from taichi_splatting.optim import fractional_laprop 

from .util import  get_vector_state, get_scalar_state, get_total_weight

@dataclass 
class Group:
  name: str
  type: str

  # Parameter and gradient
  param: torch.Tensor
  grad: Optional[torch.Tensor]
  state: dict

  # Optimizer hyperparameters
  lr: float
  betas: Tuple[float, float]
  eps: float
  bias_correction: bool
  
  # Learning rate masking
  mask_lr: Optional[torch.Tensor]
  point_lr: Optional[torch.Tensor]



  def __repr__(self):
    indent = "    "
    state_str = f",\n{indent}".join(_format_state_item(k, v) for k, v in self.state.items())
    param_stats = _format_tensor_stats("param", self.param)
    grad_stats = _format_tensor_stats("grad", self.grad)
    mask_lr_stats = _format_tensor_stats("mask_lr", self.mask_lr)
    point_lr_stats = _format_tensor_stats("point_lr", self.point_lr)

    content = (
        f"{param_stats},\n"
        f"{grad_stats},\n"
        f"state=[\n{indent}{state_str}\n],\n"
        f"{mask_lr_stats}"
        f"{point_lr_stats}"
    )

    return (f"Group({self.name}, type={self.type}, "
            f"bias_correction={self.bias_correction}, "
            f"lr={self.lr}, betas={self.betas}, eps={self.eps},\n"
            f"{_indent_block(content, indent)})")

  @property
  def num_points(self):
    return self.param.shape[0]

def _format_stats_value(value: float, use_scientific: bool) -> str:
    return f"{value:.2e}" if use_scientific else f"{value:.3f}"

def _format_tensor_stats(name: str, tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return f"{name}=None"
    use_scientific = tensor.abs().mean() < 1e-3  # Use scientific notation for small values
    mean_str = _format_stats_value(tensor.mean(), use_scientific)
    std_str = _format_stats_value(tensor.std(), use_scientific)
    return f"{name}(shape={tuple(tensor.shape)}, mean={mean_str}, std={std_str})"

def _format_state_item(k: str, v) -> str:
  if isinstance(v, torch.Tensor):
    return _format_tensor_stats(k, v)
  return f"{k}={v}"

def _indent_block(text: str, indent: str) -> str:
    """Helper function to indent multiline text blocks."""
    return '\n'.join(indent + line if line else line for line in text.split('\n'))


def make_group(group, state) -> Group:
  n = len(group["params"])  
  assert n == 1, f"expected 1 tensor in group {group['name']}, got {n}"
  params = group["params"][0]

  state = state[params]
  return Group(
      name=group["name"],
      type=group["type"],

      # Parameter and gradient
      param=params.view(params.shape[0], -1),
      grad=params.grad.view(params.shape[0], -1) if params.grad is not None else None,
      state=state,

      # Optimizer hyperparameters
      lr=group["lr"],
      betas=group["betas"],
      eps=group["eps"],
      bias_correction=group["bias_correction"],
      
      # Learning rate masking
      mask_lr=group["mask_lr"],
      point_lr=group["point_lr"]

  )



def weighted_step(group:Group, 
        visible_weight: torch.Tensor,
        visible_indexes: torch.Tensor, 
        total_weight: torch.Tensor,
        module: types.ModuleType,
        basis: Optional[torch.Tensor]=None):
    

  if group.type in ["vector", "local_vector"]:
    m, v = get_vector_state(group.state, group.param)
    kernel = module.vector_kernel(betas=group.betas, eps=group.eps, dims=group.param.shape[1], bias_correction=group.bias_correction)
  elif group.type == "scalar":
    m, v = get_scalar_state(group.state, group.param)
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
        m, v, total_weight,
        group.grad, group.lr)

  if group.type == "local_vector":
    lr_step = torch.einsum('bij,bj->bi', basis, lr_step)

  if group.mask_lr is not None:
    lr_step *= group.mask_lr.view(-1).unsqueeze(0)

  # per row learning rate
  if group.point_lr is not None:
    lr_step *= group.point_lr[visible_indexes].unsqueeze(1)

  return lr_step

def saturate(x:torch.Tensor):
  return 1 - 1/torch.exp(2 * x)


class FractionalOpt(torch.optim.Optimizer):
  
  def __init__(self, kernels:types.ModuleType, param_groups:list[dict], lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
    
    assert lr > 0, f"Invalid learning rate: {lr}"
    assert eps > 0, f"Invalid epsilon: {eps}"
    assert 0.0 <= betas[0] < 1.0, f"Invalid beta1: {betas[0]}"
    assert 0.0 <= betas[1] < 1.0, f"Invalid beta2: {betas[1]}"

    defaults = dict(lr=lr, betas=betas, eps=eps, mask_lr=None, point_lr=None, type="scalar", bias_correction=bias_correction)  


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

