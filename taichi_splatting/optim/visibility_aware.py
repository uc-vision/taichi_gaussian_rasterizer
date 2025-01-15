from dataclasses import replace
import types
from typing import Optional
import torch

from taichi_splatting.optim import fractional_adam, fractional_laprop
from taichi_splatting.optim.fractional import make_group, saturate, weighted_step
from taichi_splatting.optim.util import get_total_weight

def get_running_vis(state:dict,  n:int, device:torch.device):
  if 'running_vis' not in state:
    state['running_vis'] = torch.zeros( (n,), device=device, dtype=torch.float32)

  return state['running_vis']

@torch.compile
def exp_lerp(t, a, b):
    max_ab = torch.maximum(a, b)
    return max_ab + torch.log(torch.lerp(torch.exp(a - max_ab), torch.exp(b - max_ab), t))


def lerp(t, a, b):
  return a + (b - a) * t


def max_decaying(t, a, b):
  return torch.maximum(a, lerp(t, a, b))


def power_lerp(t, a, b, k=2):
  return lerp(t, a ** k, b ** k) ** (1/k)


def update_visibility(running_vis: torch.Tensor, 
                      visibility: torch.Tensor, indexes: torch.Tensor, 
                      total_weight: torch.Tensor,
                      beta: float = 0.9,
                      eps:float=1e-12):

  updated_vis = power_lerp(beta, visibility, running_vis[indexes], k=4)
  running_vis[indexes] = updated_vis

  weight = visibility / torch.clamp_min(updated_vis, eps)
  return weight


def set_indexes(target:torch.Tensor, values:torch.Tensor, indexes:torch.Tensor):
  result = torch.zeros_like(target)
  result[indexes] = values
  return result


class VisibilityOptimizer(torch.optim.Optimizer):
  
  def __init__(self, kernels:types.ModuleType, param_groups:list[dict], lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, vis_beta=0.9, vis_smooth:float=0.01, bias_correction=True, grad_scale:float=1.0):
    
    assert lr > 0, f"Invalid learning rate: {lr}"
    assert eps > 0, f"Invalid epsilon: {eps}"
    assert 0.0 <= betas[0] < 1.0, f"Invalid beta1: {betas[0]}"
    assert 0.0 <= betas[1] < 1.0, f"Invalid beta2: {betas[1]}"
    assert 0.0 <= vis_beta < 1.0, f"Invalid visibility beta: {vis_beta}"
    defaults = dict(lr=lr, betas=betas, eps=eps, mask_lr=None, point_lr=None, type="scalar", bias_correction=bias_correction)  

    self.vis_beta = vis_beta
    self.vis_smooth = vis_smooth
    self.grad_scale = grad_scale
    self.kernels = kernels
    super().__init__(param_groups, defaults)


  @torch.no_grad()
  def step(self, 
          indexes: torch.Tensor, 
          visibility: torch.Tensor, 
          basis: Optional[torch.Tensor]=None):
    
    assert visibility.shape == indexes.shape, f"shape mismatch {visibility.shape} != {indexes.shape}"

    groups = [make_group(group, self.state) for group in self.param_groups]
    n = groups[0].num_points


    total_weight = get_total_weight(groups[0].state, n, device=visibility.device)
    running_vis = get_running_vis(groups[0].state, n, device=visibility.device)


    weight = update_visibility(running_vis, visibility, indexes, total_weight, self.vis_beta)
    total_weight[indexes] += weight

    for group in groups:
      if group.grad is None: 
        continue

      assert group.num_points == n, f"param shape {group.num_points} != {n}"
      group = replace(group, grad=set_indexes(group.grad, 
                        group.grad[indexes] * self.grad_scale  / (visibility.unsqueeze(1) + self.vis_smooth), 
                          indexes))

      lr_step = weighted_step(group, weight, indexes, total_weight, self.kernels, basis)

      # group.param[indexes] -= lr_step * weight.unsqueeze(1)
      group.param[indexes] -= lr_step * saturate(weight).unsqueeze(1)



class VisibilityAwareAdam(VisibilityOptimizer):
  def __init__(self, param_groups, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, vis_beta=0.5, 
               vis_smooth:float = 0.01, bias_correction=True):
    super().__init__(fractional_adam, param_groups=param_groups,
                     lr=lr, betas=betas, eps=eps, vis_beta=vis_beta, 
                     vis_smooth=vis_smooth, bias_correction=bias_correction)


class VisibilityAwareLaProp(VisibilityOptimizer):
  def __init__(self, param_groups, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-16, vis_beta=0.5, 
               vis_smooth:float=0.01, bias_correction=True):
    
    
    super().__init__(fractional_laprop, param_groups=param_groups,
                     lr=lr, betas=betas, eps=eps, vis_beta=vis_beta, 
                     vis_smooth=vis_smooth, bias_correction=bias_correction)
