from functools import cache
import math
from beartype import beartype
import taichi as ti
import torch

from taichi_splatting.misc.autograd import restore_grad
from taichi_splatting.taichi_lib.conversions import torch_taichi

# Derived from torch-spherical-harmonics
# https://github.com/cheind/torch-spherical-harmonics

def check_sh_degree(sh_features):
  assert len(sh_features.shape) == 3, f"SH features must have 3 dimensions, got {sh_features.shape}"

  n_sh = sh_features.shape[2]
  n = int(math.sqrt(n_sh))

  assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
  return (n - 1)

@cache
def sh_function(degree:int=3, dimension:int=3, 
                dtype=torch.float32):
  
  ti_dtype = torch_taichi[dtype]
  point_vec = ti.types.vector(3, ti_dtype)



  vec16 = ti.types.vector(16, ti_dtype)
  vec9 = ti.types.vector(9,   ti_dtype)
  vec4 = ti.types.vector(4,   ti_dtype)
  vec3 = ti.types.vector(3,   ti_dtype)
  vec1 = ti.types.vector(1,   ti_dtype)

  @ti.func
  def rsh_cart_0(_: vec3) -> vec1:
      return vec1(
          0.282094791773878
      )


  @ti.func
  def rsh_cart_1(xyz: vec3) -> vec4:
      x, y, z = xyz
      return vec4(
          0.282094791773878,
          -0.48860251190292 * y,
          0.48860251190292 * z,
          -0.48860251190292 * x,
      )

  @ti.func
  def rsh_cart_2(xyz: vec3) -> vec9:
      x, y, z = xyz

      x2 = x**2
      y2 = y**2
      z2 = z**2
      xy = x * y
      xz = x * z
      yz = y * z

      return vec9(
          0.282094791773878,
          -0.48860251190292 * y,
          0.48860251190292 * z,
          -0.48860251190292 * x,
          1.09254843059208 * xy,
          -1.09254843059208 * yz,
          0.94617469575756 * z2 - 0.31539156525252,
          -1.09254843059208 * xz,
          0.54627421529604 * x2 - 0.54627421529604 * y2,
      )

  @ti.func
  def rsh_cart_3(xyz: vec3) -> vec16:
      x, y, z = xyz.x, xyz.y, xyz.z

      x2 = x**2
      y2 = y**2
      z2 = z**2
      xy = x * y
      xz = x * z
      yz = y * z

      return vec16(
          0.282094791773878,
          -0.48860251190292 * y,
          0.48860251190292 * z,
          -0.48860251190292 * x,
          1.09254843059208 * xy,
          -1.09254843059208 * yz,
          0.94617469575756 * z2 - 0.31539156525252,
          -1.09254843059208 * xz,
          0.54627421529604 * x2 - 0.54627421529604 * y2,
          -0.590043589926644 * y * (3.0 * x2 - y2),
          2.89061144264055 * xy * z,
          0.304697199642977 * y * (1.5 - 7.5 * z2),
          1.24392110863372 * z * (1.5 * z2 - 0.5) - 0.497568443453487 * z,
          0.304697199642977 * x * (1.5 - 7.5 * z2),
          1.44530572132028 * z * (x2 - y2),
          -0.590043589926644 * x * (x2 - 3.0 * y2),
      )        



  assert degree >= 0 and degree <= 3

  rsh_cart_n = [rsh_cart_0, rsh_cart_1, rsh_cart_2, rsh_cart_3]
  rsh_cart = rsh_cart_n[degree]

  param_mat = ti.types.matrix(n=dimension, m=(degree + 1)**2, dtype=ti_dtype)
  vec = ti.types.vector(n=dimension, dtype=ti_dtype)

  @ti.kernel    
  def evaluate_sh_at_kernel(params:ti.types.ndarray(param_mat, ndim=1), 
                          points:ti.types.ndarray(point_vec, ndim=1), 
                          indexes :ti.types.ndarray(ti.i64, ndim=1),
                          camera_pos:ti.types.ndarray(ti_dtype, ndim=1),
                          out:ti.types.ndarray(vec, ndim=1)):
      
      for idx in indexes:
          cam_pos = vec3(camera_pos[0], camera_pos[1], camera_pos[2])

          coeffs = rsh_cart(ti.math.normalize(points[idx] - cam_pos))
          params_i = params[idx]

          for d in ti.static(range(dimension)):
              out[idx][d] = ti.math.clamp(
                coeffs.dot(params_i[d, :]) + 0.5, 0, 1)



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params:torch.Tensor, points:torch.Tensor, indexes:torch.Tensor, camera_pos:torch.Tensor) -> torch.Tensor:
        
        out = torch.empty(params.shape[0], params.shape[1], dtype=dtype, device=params.device)
        evaluate_sh_at_kernel(params, points, indexes, camera_pos, out)

        ctx.save_for_backward(params, points, camera_pos, out)
        
        ctx.indexes = indexes
        ctx.mark_non_differentiable(indexes)

        return out

    @staticmethod
    def backward(ctx, doutput):
        params, points, camera_pos, out = ctx.saved_tensors

        with restore_grad(params, points, camera_pos, out):
          out.grad = doutput.contiguous()
          evaluate_sh_at_kernel.grad(params, points, ctx.indexes, camera_pos, out)

          return params.grad, points.grad, None, camera_pos.grad
        
  return _module_function


@beartype
def evaluate_sh_at(sh_params:torch.Tensor,  # M, K (degree + 1)^2,  (usually K=3, for RGB)
                positions:torch.Tensor,     # M, 3 (packed gaussian or xyz)

                indexes:torch.Tensor,        # N, 1 (indexes to gaussians) 0 to M
                camera_pos:torch.Tensor # 3
                ) -> torch.Tensor:    # N, K
    degree = check_sh_degree(sh_params)

    _module_function = sh_function(degree=degree, 
                                   dimension=sh_params.shape[1], 
                                   dtype=sh_params.dtype)
    return _module_function.apply(sh_params.contiguous(), positions.contiguous(), indexes.contiguous(), camera_pos.contiguous())










    
    