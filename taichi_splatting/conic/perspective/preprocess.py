
from functools import cache
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.misc.autograd import restore_grad

from taichi_splatting.camera_params import CameraParams
import taichi_splatting.taichi_lib.f32 as lib
from taichi_splatting.conic.grid_query import cov_tile_ranges, obb_grid_query

# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()

import warnings
warnings.filterwarnings('ignore', '(.*)that is not a leaf Tensor is being accessed(.*)') 


@cache
def preprocess_conic_function(tile_size:int=16, gaussian_scale:float=3.0):

  @ti.kernel
  def preprocess_conic_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)

    T_image_camera: ti.types.ndarray(ndim=2),  # (3, 3) camera projection
    T_camera_world: ti.types.ndarray(ndim=2),  # (4, 4)

    image_size: ti.math.ivec2,   
    
    points: ti.types.ndarray(lib.GaussianConic.vec, ndim=1),  # (N, 6)
    depth: ti.types.ndarray(lib.dtype, ndim=1),  # (N,)
    tile_count: ti.types.ndarray(ti.i32, ndim=1),  # (N,)

    blur_cov:lib.dtype
  ):

    for idx in range(position.shape[0]):

      camera_image = lib.mat3_from_ndarray(T_image_camera)
      camera_world = lib.mat4_from_ndarray(T_camera_world)

      uv, point_in_camera = lib.project_perspective_camera_image(
          position[idx], camera_world, camera_image)
    
      cov_in_camera = lib.gaussian_covariance_in_camera(
          camera_world, ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]))

      uv_cov = lib.upper(lib.project_perspective_conic(
          camera_image, point_in_camera, cov_in_camera))
      
      # add small fudge factor blur to avoid numerical issues
      uv_cov += lib.vec3([blur_cov, 0, blur_cov]) 
      uv_conic = lib.inverse_cov(uv_cov)

      min_bound, max_bound = cov_tile_ranges(uv.xy, uv_cov, 
                                             image_size, gaussian_scale, tile_size)

      r = (max_bound - min_bound)
      n = r.x * r.y

      if n == 0:
        continue

      depth[idx] = point_in_camera.z
      points[idx] = lib.GaussianConic.to_vec(
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
      depth = torch.empty(n, dtype=dtype, device=device)

      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)


      preprocess_conic_kernel(*gaussian_tensors, 
            indexes,
            T_image_camera, T_camera_world,
            points, depth,
            blur_cov)
      
      ctx.indexes = indexes
      ctx.blur_cov = blur_cov
      
      ctx.mark_non_differentiable(indexes)
      ctx.save_for_backward(*gaussian_tensors,
         T_image_camera, T_camera_world, points, depth)
      
      return points, depth

    @staticmethod
    def backward(ctx, dpoints, ddepth):

      gaussian_tensors = ctx.saved_tensors[:4]
      T_image_camera, T_camera_world, points, depth = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors,  T_image_camera, T_camera_world, points, depth):
        points.grad = dpoints.contiguous()
        depth.grad = ddepth.contiguous()
        
        preprocess_conic_kernel.grad(
          *gaussian_tensors,  
          ctx.indexes,
          T_image_camera, T_camera_world, 
          points, depth,
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
  
  _module_function = preprocess_conic_function(position.dtype)
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
def project_to_conic(gaussians:Gaussians3D, indexes:torch.Tensor, camera_params: CameraParams, 
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
    depth: torch.Tensor (N,)  - depth of each point in camera space
  """

  return apply(
      *gaussians.shape_tensors(),
      indexes,
      camera_params.T_image_camera, 
      camera_params.T_camera_world,

      blur_cov = camera_params.blur_cov
  )




