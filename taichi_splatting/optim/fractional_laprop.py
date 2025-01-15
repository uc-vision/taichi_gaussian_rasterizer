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
             
             m_arr: ti.types.ndarray(dtype=ti.f32, ndim=2),       # N x D - Running average of gradient
             v_arr: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D - Running square of gradient
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1),  # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),          # N x D - Gradient input
             
             lr: ti.f32, # Learning rate
            ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]
      
      bias1 = 1.0 - beta1 ** total_weight[idx] if ti.static(bias_correction) else 1.0
      bias2 = 1.0 - beta2 ** total_weight[idx] if ti.static(bias_correction) else 1.0

      for j in range(grad.shape[1]):
        g = grad[idx, j]
        
        v = lerp(beta2 ** w, v_arr[idx, j], g * g)
        m = lerp(beta1 ** w, m_arr[idx, j], 
                  g / ti.max(ti.sqrt(v / bias2), eps))
        
        lr_step[i, j] = m * lr / bias1
        
        m_arr[idx, j] = m
        v_arr[idx, j] = v

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
             
             m_arr: ti.types.ndarray(dtype=vec, ndim=1),          # N x D - Running average of gradient
             v_arr: ti.types.ndarray(dtype=ti.f32, ndim=1),    # N - Running norm of gradient
             total_weight: ti.types.ndarray(dtype=ti.f32, ndim=1),  # N step for each point (total weight)
             grad: ti.types.ndarray(dtype=vec, ndim=1),             # N x D - Gradient input
             
             lr: ti.f32, # Learning rate
            ):

    for i in indexes:
      idx = indexes[i]
      w = weight[i]
      
      bias1 = 1.0 - beta1 ** total_weight[idx] if ti.static(bias_correction) else 1.0
      bias2 = 1.0 - beta2 ** total_weight[idx] if ti.static(bias_correction) else 1.0

      g = grad[idx]
      
      v = lerp(beta2 ** w, v_arr[idx], g.dot(g))
      m = lerp(beta1 ** w, m_arr[idx], 
                 g / ti.max(ti.sqrt(v / bias2), eps))
      
      lr_step[i] = m * lr / bias1
      
      m_arr[idx] = m
      v_arr[idx] = v

  return kernel

