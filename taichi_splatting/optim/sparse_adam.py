from functools import cache
import torch
import taichi as ti


@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)


@cache
def adam_kernel(betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, use_point_lr=False, use_mask_lr=False):
  b1, b2 = betas

  @ti.kernel
  def kernel(param: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),  # N x D

             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             point_lr: ti.types.ndarray(dtype=ti.f32, ndim=1), # N learning rate multipliers across points
             mask_lr: ti.types.ndarray(dtype=ti.f32, ndim=1),  # D learning rate multipliers for each member of a param vector

             global_lr: ti.f32):

    for i in indexes:
      idx = indexes[i]
      lr_idx = point_lr[idx] if ti.static(use_point_lr) else 1.0

      for j in range(param.shape[1]):
        g = grad[idx, j]

        if ti.static(weight_decay != 0.0):
          g += weight_decay * param[idx, j]

        avg = lerp(b1, exp_avg[idx, j], g)
        avg_sq = lerp(b2, exp_avg_sq[idx, j], g * g)

        lr = global_lr * lr_idx * (mask_lr[j] if ti.static(use_mask_lr) else 1.0)
          
        param[idx, j] -= lr * avg / (ti.sqrt(avg_sq) + eps)

        exp_avg[idx, j] = avg
        exp_avg_sq[idx, j] = avg_sq

  return kernel





class SparseAdam(torch.optim.Optimizer):
  def __init__(self, params, lr=0.001, 
               betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, momentum=0.):
    
    if not 0.0 <= lr:
      raise ValueError(f"Invalid learning rate: {lr}")
    if not 0.0 <= eps:
      raise ValueError(f"Invalid epsilon value: {eps}")
    if not 0.0 <= betas[0] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
    if not 0.0 <= betas[1] < 1.0:
      raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, point_lr=None, mask_lr=None)  
    super().__init__(params, defaults)

  def update_group(self, param, **kwargs):
    for group in self.param_groups:
      if param in group["params"]:
        group.update(kwargs)

  @torch.no_grad()
  def step(self, visible_indexes: torch.Tensor):
    for group in self.param_groups:

      assert len(group["params"]) == 1, "more than one tensor in group"
      param = group["params"][0]
      if param.grad is None:
        continue
      
      # Lazy state initialization
      state = self.state[param]

      grad = param.grad.view(param.shape[0], -1)
      param = param.view(param.shape[0], -1) 

      if len(state) == 0:
        state['step'] = torch.tensor(0.0, dtype=torch.float32)
        state['exp_avg'] = torch.zeros_like(param)
        state['exp_avg_sq'] = torch.zeros_like(param)

      exp_avg = state["exp_avg"]
      exp_avg_sq = state["exp_avg_sq"]

      kernel = adam_kernel(betas=group["betas"], eps=group["eps"], 
                          weight_decay=group["weight_decay"], 
                          use_point_lr=group["point_lr"] is not None, 
                          use_mask_lr=group["mask_lr"] is not None)
      
      if group["point_lr"] is not None:
        point_lr=group["point_lr"]
        assert point_lr.shape == (param.shape[0],), f"point_lr shape {point_lr.shape} != {param.shape[0]}"
      else:
        point_lr = torch.empty((0,), device=param.device, dtype=torch.float32)

      if group["mask_lr"] is not None:

        mask_lr = group["mask_lr"].view(-1)
        assert mask_lr.shape == param.shape[1:], f"mask_lr shape {mask_lr.shape} != {param.shape[1:]}"

      else:
        mask_lr = torch.empty((0,), device=param.device, dtype=torch.float32)
          
      kernel(param, grad, exp_avg, exp_avg_sq, visible_indexes, point_lr=point_lr, mask_lr=mask_lr, global_lr=group["lr"])

