from functools import cache
import taichi as ti

from taichi_splatting.taichi_queue import queued
from taichi_splatting.taichi_lib.f32 import lerp

@cache
def scalar_kernel(betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
  beta1, beta2 = betas

  @queued
  @ti.kernel
  def kernel(lr_step: ti.types.ndarray(dtype=ti.f32, ndim=2), # M x D - Output step (to be scaled by lr)
             
             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1),     # M visible indexes  
             weight: ti.types.ndarray(dtype=ti.f32, ndim=1),        # M weight of each visible index
             
             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),       # N x D - Running average of gradient
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D - Running square of gradient
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1),  # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),          # N x D - Gradient input
             
             lr: ti.f32, # Learning rate
            ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]
      
      exp_avg_lr_1 = 1.0 - beta1 ** total_weight[idx] if ti.static(bias_correction) else 1.0
      exp_avg_lr_2 = 1.0 - beta2 ** total_weight[idx] if ti.static(bias_correction) else 1.0

      for j in range(grad.shape[1]):
        g = grad[idx, j]
        
        avg_sq = lerp(beta2 ** w, exp_avg_sq[idx, j], g * g)
        avg = lerp(beta1 ** w, exp_avg[idx, j], 
                   g / ti.max(ti.sqrt(avg_sq / exp_avg_lr_2), eps))
        
        lr_step[i, j] = avg * lr / exp_avg_lr_1
        
        exp_avg_sq[idx, j] = avg_sq
        exp_avg[idx, j] = avg

  return kernel

@cache
def vector_kernel(betas=(0.9, 0.999), eps=1e-16, dims=3, bias_correction=True):
  beta1, beta2 = betas
  vec = ti.types.vector(n=dims, dtype=ti.f32)

  @queued
  @ti.kernel
  def kernel(lr_step: ti.types.ndarray(dtype=vec, ndim=1), # M x D - Output step (to be scaled by lr)
             
             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1),     # M visible indexes
             weight: ti.types.ndarray(dtype=ti.f32, ndim=1),        # M weight of each visible index
             
             exp_avg: ti.types.ndarray(dtype=vec, ndim=1),          # N x D - Running average of gradient
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1),    # N - Running norm of gradient
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1),  # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=vec, ndim=1),             # N x D - Gradient input
             
             lr: ti.f32, # Learning rate
            ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]
      
      exp_avg_lr_1 = 1.0 - beta1 ** total_weight[idx] if ti.static(bias_correction) else 1.0
      exp_avg_lr_2 = 1.0 - beta2 ** total_weight[idx] if ti.static(bias_correction) else 1.0

      g = grad[idx]
      
      avg_sq = lerp(beta2 ** w, exp_avg_sq[idx], g.dot(g))
      avg = lerp(beta1 ** w, exp_avg[idx], 
                 g / ti.max(ti.sqrt(avg_sq / exp_avg_lr_2), eps))
      
      lr_step[i] = avg * lr / exp_avg_lr_1
      
      exp_avg_sq[idx] = avg_sq
      exp_avg[idx] = avg

  return kernel

