
from functools import cache
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.misc.autograd import restore_grad

from .params import CameraParams
from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.conversions import torch_taichi

# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()

import warnings
warnings.filterwarnings('ignore', '(.*)that is not a leaf Tensor is being accessed(.*)') 


@cache
def project_to_image_function(torch_dtype=torch.float32):
  dtype = torch_taichi[torch_dtype]
  lib = get_library(dtype)


  @ti.kernel
  def project_perspective_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)

    indexes: ti.types.ndarray(ti.i64, ndim=1),  # (N) indexes of points to render from 0 to M

    T_image_camera: ti.types.ndarray(ndim=2),  # (3, 3) camera projection
    T_camera_world: ti.types.ndarray(ndim=2),  # (4, 4)
    
    points: ti.types.ndarray(lib.GaussianConic.vec, ndim=1),  # (N, 6)
    depth_var: ti.types.ndarray(lib.vec3, ndim=1),  # (N, 3)
    blur_cov:ti.f32
  ):

    for i in range(indexes.shape[0]):
      idx = indexes[i]

      camera_image = lib.mat3_from_ndarray(T_image_camera)
      camera_world = lib.mat4_from_ndarray(T_camera_world)

      uv, point_in_camera = lib.project_perspective_camera_image(
          position[idx], camera_world, camera_image)
    
      
      cov_in_camera = lib.gaussian_covariance_in_camera(
          camera_world, ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]))

      uv_cov = lib.upper(lib.project_perspective_gaussian(
          camera_image, point_in_camera, cov_in_camera))
      
      # add small fudge factor blur to avoid numerical issues
      uv_cov += lib.vec3([blur_cov, 0, blur_cov]) 
      uv_conic = lib.inverse_cov(uv_cov)

      depth_var[i] = lib.vec3(point_in_camera.z, cov_in_camera[2, 2], point_in_camera.z ** 2)
      points[i] = lib.GaussianConic.to_vec(
          uv=uv.xy,
          uv_conic=uv_conic,
          alpha=lib.sigmoid(alpha_logit[idx][0]),
      )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, log_scaling, rotation, alpha_logit,
                indexes,
                T_image_camera, T_camera_world,
                blur_cov):
      dtype, device = T_image_camera.dtype, T_image_camera.device

      n = indexes.shape[0]

      points = torch.empty((n, lib.GaussianConic.vec.n), dtype=dtype, device=device)
      depth_vars = torch.empty((n, 3), dtype=dtype, device=device)

      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)


      project_perspective_kernel(*gaussian_tensors, 
            indexes,
            T_image_camera, T_camera_world,
            points, depth_vars,
            blur_cov)
      
      ctx.indexes = indexes
      ctx.blur_cov = blur_cov
      
      ctx.mark_non_differentiable(indexes)
      ctx.save_for_backward(*gaussian_tensors,
         T_image_camera, T_camera_world, points, depth_vars)
      
      return points, depth_vars

    @staticmethod
    def backward(ctx, dpoints, ddepth_vars):

      gaussian_tensors = ctx.saved_tensors[:4]
      T_image_camera, T_camera_world, points, depth_vars = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors,  T_image_camera, T_camera_world, points, depth_vars):
        points.grad = dpoints.contiguous()
        depth_vars.grad = ddepth_vars.contiguous()
        
        project_perspective_kernel.grad(
          *gaussian_tensors,  
          ctx.indexes,
          T_image_camera, T_camera_world, 
          points, depth_vars,
          ctx.blur_cov)

        return (*[tensor.grad for tensor in gaussian_tensors], 
                None, T_image_camera.grad, T_camera_world.grad,
                None)

  return _module_function

@beartype
def apply(position:torch.Tensor, log_scaling:torch.Tensor,
          rotation:torch.Tensor, alpha_logit:torch.Tensor,
          indexes:torch.Tensor,
          T_image_camera:torch.Tensor, T_camera_world:torch.Tensor,
          blur_cov:float=0.3):
  
  _module_function = project_to_image_function(position.dtype)
  return _module_function.apply(
    position.contiguous(),
    log_scaling.contiguous(),
    rotation.contiguous(),
    alpha_logit.contiguous(),
    indexes.contiguous(),
        
    T_image_camera.contiguous(), 
    T_camera_world.contiguous(),
    
    blur_cov)

@beartype
def project_to_image(gaussians:Gaussians3D, indexes:torch.Tensor, camera_params: CameraParams, 
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ 
  Project 3D gaussians to 2D gaussians in image space using perspective projection.
  Use EWA approximation for the projection of the gaussian covariance,
  as described in Zwicker, et al. "EWA splatting." 2003.
  
  Parameters:
    gaussians3D: 3D gaussian representation tensorclass
    camera_params: CameraParams

  Returns:
    points:    torch.Tensor (N, 6)  - packed 2D gaussians in image space
    depth_var: torch.Tensor (N, 3)  - depth, depth variance and depth^2 of gaussians
  """

  return apply(
      *gaussians.shape_tensors(),
      indexes,
      camera_params.T_image_camera, 
      camera_params.T_camera_world,

      blur_cov = camera_params.blur_cov
  )




