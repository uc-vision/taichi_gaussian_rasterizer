
from functools import cache
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch

from tensordict import tensorclass

from taichi_splatting.data_types import Gaussians3D, RasterConfig
from taichi_splatting.mapper.grid_query import tile_ranges
from taichi_splatting.misc.autograd import restore_grad

from taichi_splatting.camera_params import CameraParams
import taichi_splatting.taichi_lib.f32 as lib


# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()

import warnings
warnings.filterwarnings('ignore', '(.*)that is not a leaf Tensor is being accessed(.*)') 


@tensorclass
class ProcessedConic:
  points : torch.Tensor # (N, 6) float32

  tile_range : torch.Tensor # (N, 4) short
  tile_count : torch.Tensor # (N,) int32
  obb : torch.Tensor # (N, 6) float32
  depths : torch.Tensor # (N,) float32
  

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

    
    points: ti.types.ndarray(lib.GaussianConic.vec, ndim=1),  # (N, 6)
    depth: ti.types.ndarray(lib.dtype, ndim=1),  # (N,)

    tile_count: ti.types.ndarray(ti.i32, ndim=1),  # (N,)
    tile_range: ti.types.ndarray(lib.short_vec4, ndim=1),  # (N, 4)
    obb: ti.types.ndarray(lib.OBBox.vec, ndim=1),  # (N, 6)

    image_size: ti.math.ivec2,   
    depth_range: ti.math.vec2,
    blur_cov:lib.dtype,

  ):

    ti.loop_config(block_dim=256)
    for idx in range(position.shape[0]):

      camera_image = lib.mat3_from_ndarray(T_image_camera)
      camera_world = lib.mat4_from_ndarray(T_camera_world)

      uv, point_in_camera = lib.project_perspective_camera_image(
          position[idx], camera_world, camera_image)
      
      if (point_in_camera.z < depth_range[0] or point_in_camera.z > depth_range[1]):
        tile_count[idx] = 0
        continue
    
      cov_in_camera = lib.gaussian_covariance_in_camera(
          camera_world, ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]))

      uv_cov = lib.upper(lib.project_perspective_conic(
          camera_image, point_in_camera, cov_in_camera))
      
      # add small fudge factor blur to avoid numerical issues
      uv_cov += lib.vec3([blur_cov, 0, blur_cov]) 

      radius = lib.radii_from_cov(uv_cov) * gaussian_scale
      min_tile, max_tile =  tile_ranges(min_bound = ti.max(0.0, uv - radius), max_bound = uv + radius,
        image_size = image_size, tile_size = tile_size)

      n = (max_tile.x - min_tile.x) * (max_tile.y - min_tile.y)

      tile_count[idx] = n

      if n == 0:
        continue

      obb[idx] = lib.OBBox.to_vec(uv,
        lib.cov_inv_basis(uv_cov, gaussian_scale))

      tile_range[idx] = lib.short_vec4(min_tile, max_tile)
      tile_count[idx] = n

      depth[idx] = point_in_camera.z
      points[idx] = lib.GaussianConic.to_vec(
          uv=uv,
          uv_conic=lib.inverse_cov(uv_cov),
          alpha=lib.sigmoid(alpha_logit[idx][0]),
      )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, log_scaling, rotation, alpha_logit,
                T_image_camera, T_camera_world,

                image_size,
                depth_range,
                blur_cov):
      dtype, device = T_image_camera.dtype, T_image_camera.device

      n = position.shape[0]

      points = torch.empty((n, lib.GaussianConic.vec.n), dtype=dtype, device=device)
      depth = torch.empty(n, dtype=dtype, device=device)
      tile_counts = torch.zeros(n, dtype=torch.int32, device=device)
      tile_ranges = torch.empty((n, 4), dtype=torch.int16, device=device)
      obb = torch.empty((n, 6), dtype=dtype, device=device)

      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)


      preprocess_conic_kernel(*gaussian_tensors, 
            T_image_camera, T_camera_world,
            points, 
            depth,
            tile_counts,
            tile_ranges,
            obb,

            image_size,
            depth_range,
            blur_cov)
      
      ctx.image_size = image_size
      ctx.depth_range = depth_range
      ctx.blur_cov = blur_cov

      
      ctx.mark_non_differentiable(tile_counts, tile_ranges, obb)
      ctx.save_for_backward(*gaussian_tensors,
         T_image_camera, T_camera_world, points, depth, tile_counts, tile_ranges, obb)
      
      return points, depth, tile_counts, tile_ranges, obb

    @staticmethod
    def backward(ctx, dpoints, ddepth, _):

      gaussian_tensors = ctx.saved_tensors[:4]
      T_image_camera, T_camera_world, points, depth, tile_counts, tile_ranges, obb = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors,  T_image_camera, T_camera_world, points, depth):
        points.grad = dpoints.contiguous()
        depth.grad = ddepth.contiguous()
        
        preprocess_conic_kernel.grad(
          *gaussian_tensors,  
          T_image_camera, T_camera_world, 
          points, 
          depth, 
          
          tile_counts,
          tile_ranges,
          obb,


          ctx.depth_range,
          ctx.image_size,
          ctx.blur_cov)

        return (*[tensor.grad for tensor in gaussian_tensors], 
                T_image_camera.grad, T_camera_world.grad,
                None, None)

  return _module_function

@beartype
def apply(position:torch.Tensor, log_scaling:torch.Tensor,
          rotation:torch.Tensor, alpha_logit:torch.Tensor,
          T_image_camera:torch.Tensor, 
          T_camera_world:torch.Tensor,

          config:RasterConfig,

          image_size:Tuple[Integral, Integral],
          depth_range:Tuple[float, float],
          blur_cov:float=0.3):
  
  _module_function = preprocess_conic_function( 
        tile_size=config.tile_size, gaussian_scale=config.gaussian_scale)
  
  return _module_function.apply(
    position.contiguous(),
    log_scaling.contiguous(),
    rotation.contiguous(),
    alpha_logit.contiguous(),
        
    T_image_camera.contiguous(), 
    T_camera_world.contiguous(),

    ti.math.ivec2(image_size),
    ti.math.vec2(depth_range),
    blur_cov)

@beartype
def preprocess_conic(gaussians:Gaussians3D, camera_params: CameraParams, 
                     config:RasterConfig) -> ProcessedConic:
  """ 
  Project 3D gaussians to 2D gaussians in image space using perspective projection.
  Use EWA approximation for the projection of the gaussian covariance,
  as described in Zwicker, et al. "EWA splatting." 2003.
  
  Parameters:
    gaussians3D: 3D gaussian representation tensorclass
    camera_params: CameraParams

  Returns:
    points:   ProcessedConic
  """

  points, depth, tile_counts, tile_ranges, obb = apply(
      *gaussians.shape_tensors(),
      camera_params.T_image_camera, 
      camera_params.T_camera_world,

      config,

      image_size=camera_params.image_size,
      depth_range=camera_params.depth_range,
      blur_cov = camera_params.blur_cov
  )

  return ProcessedConic(points, 
            tile_range=tile_ranges, tile_count=tile_counts, 
            obb=obb, depths=depth)




