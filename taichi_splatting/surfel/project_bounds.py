
from functools import cache
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
import torch


from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.conversions import torch_taichi

# Ignore this from taichi/pytorch integration 
# taichi/lang/kernel_impl.py:763: UserWarning: The .grad attribute of a Tensor 
# that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated 
# during autograd.backward()

import warnings
warnings.filterwarnings('ignore', '(.*)that is not a leaf Tensor is being accessed(.*)') 


@cache
def _surfel_bounds(torch_dtype=torch.float32, gaussian_scale:float=3.0):
  dtype = torch_taichi[torch_dtype]
  lib = get_library(dtype)


  @ti.kernel
  def surface_bounds_kernel(  
    projection_arr : ti.types.ndarray(dtype, ndim=2),  # (M, 4, 4)  
    points: ti.types.ndarray(lib.GaussianSurfel.vec, ndim=1),  # (N, 10)
    bounds: ti.types.ndarray(lib.Quad.vec, ndim=1),
    depth: ti.types.ndarray(ti.f32, ndim=1),  # N,
  ):
    
    for i in range(points.shape[0]):
      surfel = lib.GaussianSurfel.from_vec(points[i])

      projection = lib.mat4_from_ndarray(projection_arr)
      image_t_surface = projection @ surfel.world_t_surface()


      corners = [lib.project_perspective(p, image_t_surface)[0] for p in ti.static([
          lib.vec3(-gaussian_scale, -gaussian_scale,  0),
          lib.vec3(-gaussian_scale, gaussian_scale,   0),
          lib.vec3(gaussian_scale, gaussian_scale,    0),
          lib.vec3(gaussian_scale, -gaussian_scale,   0),
      ])]


      bounds[i] = lib.Quad.to_vec(*corners)
      pos, depth[i] = lib.project_perspective(surfel.pos, projection)



  def f(surfel:torch.Tensor, image_t_camera:torch.Tensor):
    n = surfel.shape[0]
    bounds = torch.zeros(n, lib.Quad.vec.n, dtype=torch.float32)
    depth = torch.zeros(n, dtype=torch.float32)

    surface_bounds_kernel(image_t_camera, surfel, bounds, depth)

    return bounds, depth
  
  return f

@beartype
def surfel_bounds(points:torch.Tensor, projection:torch.Tensor, gaussian_scale:float=3.0):
    assert projection.shape == (4, 4), f"Expected full projection (4, 4) but got {projection.shape}"

    compute_bounds = _surfel_bounds(points.dtype, gaussian_scale)
    return compute_bounds(points, projection)
