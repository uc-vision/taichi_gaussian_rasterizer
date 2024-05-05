from functools import cache
from typing import NamedTuple
import taichi as ti
import torch

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.taichi_lib import get_library
from taichi_splatting.taichi_lib.conversions import torch_taichi, taichi_torch

RasterOut = NamedTuple('RasterOut', 
    [('image', torch.Tensor), 
     ('image_weight', torch.Tensor) ])



@cache
def rasterize_func(config:RasterConfig, dtype=ti.f32):

  lib = get_library(dtype)
  torch_dtype = taichi_torch[dtype]

  @ti.kernel
  def render_surfel_kernel(
    output_image:ti.types.ndarray(dtype=ti.math.vec3, ndim=2), 
    depth_image:ti.types.ndarray(dtype=ti.f32, ndim=2),

    surfels : ti.types.ndarray(dtype=lib.GaussianSurfel.vec, ndim=1), 
    features:ti.types.ndarray(ti.math.vec3, ndim=1), 
    projection_arr : ti.types.ndarray(dtype=ti.f32, ndim=2)
    ):

    for y, x in ti.ndrange(*output_image.shape):
      projection = lib.mat4_from_ndarray(projection_arr)

      for i in range(surfels.shape[0]):

        surfel = lib.GaussianSurfel.from_vec(surfels[i])
        image_t_surface =  projection @ surfel.world_t_surface() 
        
        g, depth = lib.eval_surfel(image_t_surface, lib.vec2(x, y), config.beta)

        if depth > 0:
          output_image[y, x] += (features[i] * g)
          depth_image[y, x] = ti.min(depth, depth_image[y, x])


  def f(surfels:torch.Tensor, features:torch.Tensor, 
        image_size:tuple[int, int], projection:torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
    
    device = surfels.device
    image_shape = tuple(reversed(image_size))
    output_image = torch.zeros((*image_shape, 3), dtype=torch_dtype, device=device)
    depth_image = torch.zeros(image_shape, dtype=torch_dtype, device=device)

    render_surfel_kernel(output_image, depth_image, surfels, features, projection)
    return output_image, depth_image
  
  return f
  

def rasterize_surfels(surfels:torch.Tensor, features:torch.Tensor, 
        image_size:tuple[int, int], projection:torch.Tensor,
        config:RasterConfig

        ) -> RasterOut:
  
  assert surfels.shape[0] == features.shape[0], f"Expected same number of surfels and features but got {surfels.shape[0]} and {features.shape[0]}"
  assert projection.shape == (4, 4), f"Expected full projection (4, 4) but got {projection.shape}"

  
  _module_function = rasterize_func(config, dtype=torch_taichi[surfels.dtype])
  image, depth = _module_function(surfels.contiguous(), features.contiguous(), 
                          image_size, projection.contiguous())
  
  return RasterOut(image, depth)