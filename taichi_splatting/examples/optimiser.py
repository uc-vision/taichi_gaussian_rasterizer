import torch
import torchvision.ops as ops
import taichi as ti
from functools import cache
import torch
import taichi as ti
from typing import Optional

from taichi_splatting.taichi_queue import queued
# Initialize Taichi (GPU or CPU)
ti.init(arch=ti.gpu)
@ti.func
def lerp(t: ti.f32, a: ti.template(), b: ti.template()):
  return a * t + b * (1.0 - t)


@cache 
def optimiser_kernel(betas=(0.9, 0.999), eps=1e-08,  use_point_lr=False, use_mask_lr=False):
    b1, b2 = betas
    # Define vector type for Taichi
    dims = 3  # Dimension of each parameter (e.g., 3D vector)
    vec = ti.types.vector(n=dims, dtype=ti.f32)

    # Initialize MLP from torchvision.ops
    in_channels = dims   # Input: parameter + gradient
    hidden_channels = [64, 32]  # Example hidden layers
    mlp_grad_points = ops.MLP(in_channels=in_channels, hidden_channels=hidden_channels, activation_layer=torch.nn.ReLU)
    mlp_grad_adam = ops.MLP(in_channels=in_channels, hidden_channels=hidden_channels, activation_layer=torch.nn.ReLU)
    # Define Taichi fields (used to store parameters, gradients, etc.)
    # MLP-based optimizer kernel
    @ti.kernel
    def vector_mlp_optimizer(param: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D
             grad: ti.types.ndarray(dtype=ti.f32, ndim=2),  # N x D

             step: ti.types.ndarray(dtype=ti.f32, ndim=1), # N

             exp_avg: ti.types.ndarray(dtype=ti.f32, ndim=2),    # N x D
             exp_avg_sq: ti.types.ndarray(dtype=ti.f32, ndim=2), # N x D

             indexes: ti.types.ndarray(dtype=ti.int64, ndim=1), # M visible indexes

             point_lr: ti.types.ndarray(dtype=ti.f32, ndim=1), # N learning rate multipliers across points
             mask_lr: ti.types.ndarray(dtype=ti.f32, ndim=1),  # D learning rate multipliers for each member of a param vector

             lr: ti.f32):
        for i in indexes:
            idx = indexes[i]
            
            # Use MLP to compute the update
            with torch.no_grad():  # Disable gradient computation for the forward pass
                update_param = mlp_grad_points(param).cpu().numpy()
                update_grad = mlp_grad_adam(grad).cpu().numpy()

            # Apply the update to the parameter
            param[idx] -= update_param * lr  # Apply MLP update, scaled by learning rate

            # Update momentum-like values (optional, for future use)
            exp_avg[idx] = lerp(b1, exp_avg[idx], grad[idx])
            exp_avg_sq[idx] = lerp(b2, exp_avg_sq[idx], ti.math.dot(grad[idx], grad[idx]))


