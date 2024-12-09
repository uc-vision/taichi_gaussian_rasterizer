from functools import cache
import taichi as ti

from taichi_splatting.taichi_queue import queued
from taichi_splatting.taichi_lib.f32 import lerp

@cache
def scalar_kernel(betas=(0.9, 0.999), eps=1e-16, bias_correction=True):
  beta1, beta2 = betas

  @queued
  @ti.kernel
  def kernel(lr_step: ti.types.ndarray(dtype=ti.f32, ndim=2),   # M x D - Output step (to be scaled by lr)
             
             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1),    # M Visible indexes
             weight: ti.types.ndarray(dtype=ti.f32, ndim=1),       # M weight of each visible index

             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),      # N x D - Running average of gradient
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2),   # N x D - Running average of gradient squared
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1), # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),          # N x D Gradient input

             lr: ti.f32, # Learning rate
          ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]

      bias_factor = (ti.sqrt(1 - beta2 ** total_weight[idx])  / (1 - beta1 ** total_weight[idx])
                      if ti.static(bias_correction) else 1.0)

      for j in range(lr_step.shape[1]):
        g = grad[idx, j]

        avg = lerp(beta1 ** w, exp_avg[idx, j], g)
        avg_sq = lerp(beta2 ** w, exp_avg_sq[idx, j], g * g)

        lr_step[i, j] = (avg / ti.max(ti.sqrt(avg_sq),  eps)) * bias_factor * lr

        exp_avg[idx, j] = avg
        exp_avg_sq[idx, j] = avg_sq


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
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=1),    # N  - Running norm of gradient 
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1),  # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=vec, ndim=1),             # N x D - Gradient input

             lr: ti.f32, # Learning rate
        ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]

      bias_factor = (ti.sqrt(1 - beta2 ** total_weight[idx])  / (1 - beta1 ** total_weight[idx])
                      if ti.static(bias_correction) else 1.0)

      g = grad[idx]
      avg = lerp(beta1 ** w, exp_avg[idx], g)

      norm = ti.math.dot(g, g)
      avg_sq = lerp(beta2 ** w, exp_avg_sq[idx], norm)

      lr_step[i] = (avg / ti.max(ti.sqrt(avg_sq),  eps)) * bias_factor * lr

      exp_avg[idx] = avg
      exp_avg_sq[idx] = avg_sq

  return kernel


