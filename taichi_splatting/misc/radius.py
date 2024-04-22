
from functools import cache
from beartype import beartype
import taichi as ti
import torch

from taichi_splatting.misc.autograd import restore_grad

from taichi_splatting.taichi_lib import get_library  
from taichi_splatting.taichi_lib.conversions import torch_taichi


@cache
def compute_radius_func(torch_dtype=torch.float32, gaussian_scale:float=3.0):
  dtype = torch_taichi[torch_dtype]

  lib = get_library(dtype)


  @ti.kernel
  def radius_kernel(  
    gaussians2d: ti.types.ndarray(lib.GaussianConic.vec, ndim=1),  # (N, 6) - packed 2d gaussians
    radii: ti.types.ndarray(dtype, ndim=1),  # (N, 1) - output radii
  ):

    for i in range(gaussians2d.shape[0]):
      uv_cov = lib.GaussianConic.get_cov(gaussians2d[i])
      radii[i] = lib.radii_from_cov(uv_cov) * gaussian_scale



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians2d:torch.Tensor) -> torch.Tensor:
      device = gaussians2d.device
      radii = torch.empty(gaussians2d.shape[0], dtype=torch_dtype, device=device)

      radius_kernel(gaussians2d, radii)
      ctx.save_for_backward(gaussians2d, radii)

      return radii

    @staticmethod
    def backward(ctx, grad_radius):
      gaussians2d, radii = ctx.saved_tensors

      with restore_grad(gaussians2d, radii):
        radii.grad = grad_radius.contiguous()
        radius_kernel.grad(gaussians2d, radii)

        return gaussians2d.grad

  return _module_function



@beartype
def compute_radius(gaussians2d:torch.Tensor, gaussian_scale:float=3.0):
  """ 
  Compute radii from packed 2d gaussians
  
  Parameters:
    gaussians2d: torch.Tensor (N, 6) - packed 2d gaussians
    gaussian_scale: float - number of standard deviations 

  Returns:
    radii: torch.Tensor (N, 1) - radius for each gaussian
    
  """

  _module_function = compute_radius_func(gaussians2d.dtype, gaussian_scale)
  return _module_function.apply(gaussians2d.contiguous())


