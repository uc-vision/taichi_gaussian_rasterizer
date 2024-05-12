
from functools import cache
from beartype import beartype
import taichi as ti
import torch
from taichi_splatting.camera_params import CameraParams, expand_proj
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
def gaussian3d_surfel_function(torch_dtype=torch.float32, gaussian_scale:float=3.0):
  dtype = torch_taichi[torch_dtype]
  lib = get_library(dtype)

  @ti.func
  def bounds_xy(points:ti.template()):
    lower = ti.Vector([ti.min(*points[:, 0]), ti.min(*points[:, 1])])
    upper = ti.Vector([ti.max(*points[:, 0]), ti.max(*points[:, 1])])
    return lower, upper

  @ti.kernel
  def preprocess_surfel_kernel(  
    position: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3) 
    log_scale: ti.types.ndarray(lib.vec3, ndim=1),  # (M, 3)
    rotation: ti.types.ndarray(lib.vec4,  ndim=1),  # (M, 4)
    alpha_logit: ti.types.ndarray(lib.vec1, ndim=1),  # (M)
    indexes: ti.types.ndarray(ti.i64, ndim=1),  # (N) indexes of points to render from 0 to M
  
    camera_t_world_arr: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)
    image_t_camera_arr: ti.types.ndarray(ti.f32, ndim=2),  # (4, 4)

    points: ti.types.ndarray(lib.GaussianSurfel.vec, ndim=1),  # (N, 10)
    tile_counts: ti.types.ndarray(ti.i32, ndim=1),  # (N)

  ):
    
    for i in range(indexes.shape[0]):
      idx = indexes[i]

      camera_t_world = lib.mat4_from_ndarray(camera_t_world_arr)
      image_t_camera = lib.mat4_from_ndarray(image_t_camera_arr)

      pos = lib.transform_point(camera_t_world, position[idx])
      rot = image_t_camera[:3, :3] @ lib.quat_to_mat(ti.math.normalize(rotation[idx]))

      scale = ti.exp(log_scale[idx].xy)

      # flip the surfel if the normal is pointing away from the camera
      back_facing = rot[2, 2] > 0
      
      tx = rot[:, 0] * scale.x * (-1 if back_facing else 1)
      ty = rot[:, 1] * scale.y


      camera_t_surface = lib.surfel_homography(tx, ty, pos)
      image_t_surface = image_t_camera @ camera_t_surface

      projected = [lib.project_perspective_vec(p, image_t_surface) for p in ti.static([
          lib.vec3(-gaussian_scale, -gaussian_scale,  0),
          lib.vec3(-gaussian_scale,  gaussian_scale,   0),
          lib.vec3(gaussian_scale,   gaussian_scale,    0),
          lib.vec3(gaussian_scale,  -gaussian_scale,   0),
      ])]
    
      corners = lib.mat4x3(*projected)
      min_depth = ti.min(corners[:, 2])

      if min_depth < 0:
        continue


      points[i] = lib.GaussianSurfel(
          pos=pos, tx = tx, ty = ty,
          alpha=lib.sigmoid(alpha_logit[idx][0]),
      )


  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, position, log_scaling, rotation, alpha_logit,
                indexes, camera_t_world, camera_t_image):
      dtype, device = position.dtype, position.device

      n = indexes.shape[0]
      points = torch.empty((n, lib.GaussianSurfel.vec.n), dtype=dtype, device=device)


      gaussian_tensors = (position, log_scaling, rotation, alpha_logit)
      preprocess_surfel_kernel(*gaussian_tensors, indexes, 
                               camera_t_world, camera_t_image, points)
      
      ctx.indexes = indexes
      
      ctx.mark_non_differentiable(indexes)
      ctx.save_for_backward(*gaussian_tensors, camera_t_world, camera_t_image, points)
      
      return points

    @staticmethod
    def backward(ctx, dpoints):

      gaussian_tensors = ctx.saved_tensors[:4]
      camera_t_world, camera_t_image, points = ctx.saved_tensors[4:]

      with restore_grad(*gaussian_tensors, camera_t_world, camera_t_image, points):
        points.grad = dpoints.contiguous()
        
        preprocess_surfel_kernel.grad(
          *gaussian_tensors,  ctx.indexes, camera_t_world, camera_t_image, points)

        return (*[tensor.grad for tensor in gaussian_tensors], None, )

  return _module_function

@beartype
def apply(position:torch.Tensor, log_scaling:torch.Tensor,
          rotation:torch.Tensor, alpha_logit:torch.Tensor,
          indexes:torch.Tensor,
          camera_t_world:torch.Tensor,
          camera_t_image:torch.Tensor) -> torch.Tensor:
  
  _module_function = gaussian3d_surfel_function(position.dtype)
  return _module_function.apply(
    position.contiguous(),
    log_scaling.contiguous(),
    rotation.contiguous(),
    alpha_logit.contiguous(),
    indexes.contiguous(),
    camera_t_world.contiguous(),
    camera_t_image.contiguous())

@beartype
def preprocess_surfel(gaussians:Gaussians3D, indexes:torch.Tensor, camera_params:CameraParams) -> torch.Tensor:
  """ 
  Convert Gaussians3D to planar representation (post-activation) for rendering.
  
  Parameters:
    gaussians3D: 3D gaussian representation tensorclass

  Returns:
    points:    torch.Tensor (N, 10)  - packed 2D gaussians in image space
  """

  return apply(
      *gaussians.shape_tensors(), indexes, camera_params.T_camera_world, expand_proj(camera_params.T_image_camera))
  




