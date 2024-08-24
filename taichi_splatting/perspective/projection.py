
from functools import cache
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.data_types import Gaussians3D, RasterConfig
from taichi_splatting.optim.autograd import restore_grad

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
def project_to_image_function(torch_dtype=torch.float32, clamp_margin=0.15, blur_cov=0.0):
  dtype = torch_taichi[torch_dtype]
  lib = get_library(dtype)


  @ti.kernel
  def project_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)

    # projection parameters
    T_camera_world: ti.types.ndarray(lib.mat3x4, ndim=1),  # (M, 3, 4)
    projection: ti.types.ndarray(lib.vec4, ndim=1),  # (M, 4) camera projection (fx, fy, cx, cy)
    image_size: lib.vec2,  # (2) image size
    depth_range: lib.vec2,  # (2) near and far plane

    # outputs
    points: ti.types.ndarray(lib.Gaussian2D.vec, ndim=1),  # (N, 6)
    depth: ti.types.ndarray(lib.vec1, ndim=1),  # (N, 1)

    # other parameters
    gaussian_scale: dtype

  ):
    for idx in range(position.shape[0]):
      mean, z, cov = lib.project_gaussian(
        T_camera_world[idx], projection[idx], image_size,
        position[idx], ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]), clamp_margin=clamp_margin)

      if ti.static(blur_cov > 0):
        cov += lib.vec3([blur_cov, 0, blur_cov])

      sigma, v1, v2 = lib.eig(cov)
      sx, sy = sigma * gaussian_scale
      lower, upper = lib.ellipse_bounds(mean, v1 * sx, v2 * sy)

      in_view = ((z > depth_range[0]) and (z < depth_range[1]) and 
                 (upper > 0).all() and (lower < image_size).all())
                  
      if not in_view:
        depth[idx] = 0.

      else:

        depth[idx] = z
        points[idx] = lib.Gaussian2D.to_vec(
            mean  = mean,
            axis  = v1,
            sigma = sigma,
            alpha = lib.sigmoid(alpha_logit[idx][0]),
        )



  @ti.kernel
  def indexed_project_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)

    indexes: ti.types.ndarray(ti.i64, ndim=1),  # (N) indexes of points to render from 0 to M
    T_camera_world: ti.types.ndarray(lib.mat3x4, ndim=1),  # (M, 3, 4)

    projection: ti.types.ndarray(lib.vec4, ndim=1),  # (M, 4) camera projection (fx, fy, cx, cy)
    image_size: lib.vec2,  # (2) image size
    
    points: ti.types.ndarray(lib.Gaussian2D.vec, ndim=1),  # (N, 6)
    depth: ti.types.ndarray(lib.vec1, ndim=1),  # (N, 1)
  ):

    for i in range(indexes.shape[0]):
      idx = indexes[i]

      mean, z, cov = lib.project_gaussian(
        T_camera_world[idx], projection[idx], image_size,
        position[idx], ti.math.normalize(rotation[idx]), ti.exp(log_scale[idx]), clamp_margin)
      
      if ti.static(blur_cov > 0):
        cov += lib.vec3([blur_cov, 0, blur_cov])

      sigma, v1, _ = lib.eig(cov)

      depth[i] = z
      points[i] = lib.Gaussian2D.to_vec(
          mean=mean,
          axis = v1,
          sigma = sigma,
          alpha=lib.sigmoid(alpha_logit[idx][0]),
      )



  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, log_scaling, rotation, alpha_logit,
                T_camera_world,
                projection, image_size, depth_range,
                gaussian_scale):
      dtype, device = projection.dtype, projection.device

      n = position.shape[0]

      points = torch.empty((n, lib.Gaussian2D.vec.n), dtype=dtype, device=device)
      depth = torch.empty((n, 1), dtype=dtype, device=device)

      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)


      project_kernel(*gaussian_tensors, 
            T_camera_world, projection, 
            lib.vec2(image_size), lib.vec2(depth_range),
            points, depth,  # outputs
            gaussian_scale)
      
      
      ctx.indexes = torch.nonzero(depth[:, 0]).squeeze(1)

      points = points[ctx.indexes]
      depth = depth[ctx.indexes]


      ctx.image_size = image_size
      ctx.depth_range = depth_range

      ctx.gaussian_scale = gaussian_scale
      ctx.mark_non_differentiable(ctx.indexes)

      ctx.save_for_backward(*gaussian_tensors,
         T_camera_world, 
         projection, 
         points, depth)
            
      return points, depth, ctx.indexes

    @staticmethod
    def backward(ctx, dpoints, ddepth, dindexes):

      gaussian_tensors = ctx.saved_tensors[:4]
      T_camera_world, projection, points, depth = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors,  projection, T_camera_world, points, depth):
        points.grad = dpoints.contiguous()
        depth.grad = ddepth.contiguous()

      
        indexed_project_kernel.grad(
          *gaussian_tensors,  
          ctx.indexes,
          T_camera_world, 
          projection, lib.vec2(ctx.image_size),
          points, depth)


        return (*[tensor.grad for tensor in gaussian_tensors], 
                T_camera_world.grad, projection.grad,
                None, None, None)

  return _module_function

@beartype
def apply(position:torch.Tensor, log_scaling:torch.Tensor,
          rotation:torch.Tensor, alpha_logit:torch.Tensor,
          T_camera_world:torch.Tensor,
          projection:torch.Tensor, 
          
          image_size:Tuple[Integral, Integral],
          depth_range:Tuple[float, float],

          gaussian_scale:float=3.0,
          blur_cov:float=0.0,
          
          clamp_margin:float=0.15

          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  
  _module_function = project_to_image_function(position.dtype, clamp_margin, blur_cov)
  n = position.shape[0]

  return _module_function.apply(
    position.contiguous(),
    log_scaling.contiguous(),
    rotation.contiguous(),
    alpha_logit.contiguous(),

    T_camera_world[:3].unsqueeze(0).expand(n, 3, 4).contiguous(),
    projection.unsqueeze(0).expand(n, 4).contiguous(),
    image_size,

    depth_range,
    gaussian_scale)

@beartype
def project_to_image(gaussians:Gaussians3D,  camera_params: CameraParams, config:RasterConfig,
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """ 
  Project 3D gaussians to 2D gaussians in image space using perspective projection.
  Use EWA approximation for the projection of the gaussian covariance,
  as described in Zwicker, et al. "EWA splatting." 2003.
  
  Parameters:
    gaussians3D: 3D gaussian representation tensorclass
    camera_params: CameraParams
    gaussian_scale: float - scale of the gaussian 'radius' to use for culling

  Returns:
    points:    torch.Tensor (N, 6)  - packed 2D gaussians in image space
    depths:    torch.Tensor (N, 1)  - depth, depth variance and depth^2 of gaussians
    indexes:   torch.Tensor (N)     - indexes of points that are in view
  """


  return apply(
      *gaussians.shape_tensors(),
      camera_params.T_camera_world,

      camera_params.projection,
      camera_params.image_size,
      camera_params.depth_range,

      config.gaussian_scale,
      config.blur_cov,
      config.clamp_margin
  )




