
from functools import cache
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.camera_params import CameraParams
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.misc.autograd import restore_grad

from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.conversions import torch_taichi

# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()

import warnings
warnings.filterwarnings('ignore', '(.*)that is not a leaf Tensor is being accessed(.*)') 


@cache
def gaussian3d_surfel_function(torch_dtype=torch.float32):
  dtype = torch_taichi[torch_dtype]
  lib = get_library(dtype)



  @ti.kernel
  def gaussian3d_surfel_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)
    indexes: ti.types.ndarray(ti.i64, ndim=1),  # (N) indexes of points to render from 0 to M
    
    camera_t_world: ti.template(),  # (4, 4)
    points: ti.types.ndarray(lib.GaussianSurfel.vec, ndim=1),  # (N, 10)

  ):

    for i in range(indexes.shape[0]):
      idx = indexes[i]

      pos = camera_t_world @ lib.vec4(*position[idx], 1)
      rot = camera_t_world[:3, :3] @ lib.quat_to_mat(ti.math.normalize(rotation[idx]))
      scale = ti.exp(log_scale[idx].xy)

      # flip the surfel if the normal is pointing away from the camera
      back_facing = rot[2, 2].z > 0
      S = lib.diag(lib.vec2(-scale.x if back_facing else scale.x, scale.y))

      points[i] = lib.GaussianSurfel.to_vec(
          pos=pos,
          axes = S @ rot[:2, :],
          alpha=lib.sigmoid(alpha_logit[idx][0]),
      )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, log_scaling, rotation, alpha_logit,
                indexes, camera_t_world):
      dtype, device = position.dtype, position.device

      n = indexes.shape[0]
      points = torch.empty((n, lib.GaussianSurfel.vec.n), dtype=dtype, device=device)

      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)
      gaussian3d_surfel_kernel(*gaussian_tensors, indexes, points, camera_t_world)
      
      ctx.indexes = indexes
      
      ctx.mark_non_differentiable(indexes)
      ctx.save_for_backward(*gaussian_tensors, points)
      
      return points

    @staticmethod
    def backward(ctx, dpoints, ddepth):

      gaussian_tensors = ctx.saved_tensors[:4]
      T_camera_world, points = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors, T_camera_world, points):
        points.grad = dpoints.contiguous()
        
        gaussian3d_surfel_kernel.grad(
          *gaussian_tensors,  ctx.indexes, points)

        return (*[tensor.grad for tensor in gaussian_tensors], None, )

  return _module_function

@beartype
def apply(position:torch.Tensor, log_scaling:torch.Tensor,
          rotation:torch.Tensor, alpha_logit:torch.Tensor,
          indexes:torch.Tensor,
          camera_t_world:torch.Tensor) -> torch.Tensor:
  
  _module_function = gaussian3d_surfel_function(position.dtype)
  return _module_function.apply(
    position.contiguous(),
    log_scaling.contiguous(),
    rotation.contiguous(),
    alpha_logit.contiguous(),
    indexes.contiguous(),
    camera_t_world.contiguous())

@beartype
def project_to_conic(gaussians:Gaussians3D, indexes:torch.Tensor, camera_params:CameraParams ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ 
  Convert Gaussians3D to planar representation (post-activation) for rendering.
  
  Parameters:
    gaussians3D: 3D gaussian representation tensorclass

  Returns:
    points:    torch.Tensor (N, 10)  - packed 2D gaussians in image space
  """

  return apply(
      *gaussians.shape_tensors(), indexes, camera_params.camera_t_world
  )




