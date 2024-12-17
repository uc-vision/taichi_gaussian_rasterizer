from functools import cache
from typing import Optional
import numpy as np
import torch
from numbers import Integral
from beartype import beartype
from beartype.typing import Tuple, NamedTuple

import taichi as ti

from taichi_splatting.mapper.tile_mapper import map_to_tiles
from taichi_splatting.taichi_queue import queued
from taichi_splatting.taichi_lib.conversions import torch_taichi

from .forward import RasterConfig, forward_kernel, query_visibility_kernel
from .backward import backward_kernel

RasterOut = NamedTuple('RasterOut', [
  ('image', torch.Tensor),
  ('image_weight', torch.Tensor), 
  ('split_heuristic', Optional[torch.Tensor]),
  ('visibility', Optional[torch.Tensor])
])

@cache
def query_visibility_function(config: RasterConfig, dtype=torch.float32):
  query_kernel = query_visibility_kernel(config, dtype=torch_taichi[dtype])

  def _query_visibility(gaussians: torch.Tensor, 
                       overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                       image_size: Tuple[Integral, Integral]) -> torch.Tensor:
    visibility = torch.zeros((gaussians.shape[0]), dtype=dtype, device=gaussians.device)
    query_kernel(gaussians, tile_overlap_ranges, overlap_to_point, visibility, ti.math.ivec2(image_size))

    return visibility
  
  return _query_visibility

@beartype
@torch.no_grad()
def query_visibility_with_tiles(gaussians2d: torch.Tensor, 
                     overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                     image_size: Tuple[Integral, Integral], 
                     config: RasterConfig, dtype=torch.float32) -> torch.Tensor:
  """ Query visibility for a single image with tile overlap information. """
  query_kernel = query_visibility_function(config, dtype=dtype)
  return query_kernel(gaussians2d, 
                      overlap_to_point, tile_overlap_ranges.view(-1, 2),
                      image_size)

@beartype
def query_visibility(gaussians2d: torch.Tensor, depth: torch.Tensor, image_size: Tuple[Integral, Integral], 
                     config: RasterConfig, dtype=torch.float32) -> torch.Tensor:
  """ Query visibility for a single image.
  Args:
     gaussians2d: (N, 7)  packed gaussians, N is the number of gaussians
     depth: (N, 1)  depths, N is the number of gaussians
     image_size: (2, ) tuple of ints, (width, height)
     config: Config - configuration parameters for rasterization
     dtype: torch.dtype, default is torch.float32
  Returns:
       visibility: (N, ) torch tensor, where N is the number of gaussians
  """


  assert gaussians2d.shape[0] == depth.shape[0], \
    f"Size mismatch: got {gaussians2d.shape}, {depth.shape}"
  
  overlap_to_point, tile_overlap_ranges = map_to_tiles(gaussians2d, depth, image_size=image_size, config=config)
  return query_visibility_with_tiles(gaussians2d, overlap_to_point, tile_overlap_ranges,
                                    image_size, config, dtype)


@cache
def render_function(config: RasterConfig,
                   points_requires_grad: bool,
                   features_requires_grad: bool,
                   feature_size: int,
                   dtype=torch.float32):
  
  forward = forward_kernel(config, feature_size=feature_size, dtype=torch_taichi[dtype])
  backward = backward_kernel(config, points_requires_grad, features_requires_grad, 
                           feature_size, dtype=torch_taichi[dtype])
  
  forward = queued(forward)
  backward = queued(backward)

  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians: torch.Tensor, features: torch.Tensor,
                 overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                 image_size: Tuple[Integral, Integral]) -> torch.Tensor:

      shape = (image_size[1], image_size[0])
      image_feature = torch.empty((*shape, features.shape[1]), dtype=dtype, device=features.device)
      image_alpha = torch.empty(shape, dtype=dtype, device=features.device)

      if config.compute_split_heuristic:
        split_heuristic = torch.zeros((gaussians.shape[0]), dtype=dtype, device=features.device)
      else:
        split_heuristic = torch.empty((0), dtype=dtype, device=features.device)

      if config.compute_visibility:
        visibility = torch.zeros((gaussians.shape[0]), dtype=dtype, device=features.device)
      else:
        visibility = torch.empty((0), dtype=dtype, device=features.device)


      forward(gaussians, features, tile_overlap_ranges, overlap_to_point,
             image_feature, image_alpha, visibility)

      # Non differentiable parameters
      ctx.overlap_to_point = overlap_to_point
      ctx.tile_overlap_ranges = tile_overlap_ranges
      ctx.image_size = image_size
      ctx.split_heuristic = split_heuristic
      ctx.visibility = visibility

      ctx.mark_non_differentiable(image_alpha, overlap_to_point, tile_overlap_ranges, 
                                visibility, split_heuristic)
      ctx.save_for_backward(gaussians, features, image_feature)
    
      return image_feature, image_alpha, split_heuristic, visibility

    @staticmethod
    def backward(ctx, grad_image_feature: torch.Tensor,
                grad_alpha: torch.Tensor, grad_point_heuristics: torch.Tensor,
                grad_visibility: torch.Tensor):
      
      gaussians, features, image_feature = ctx.saved_tensors
      grad_gaussians = torch.zeros_like(gaussians)
      grad_features = torch.zeros_like(features)
  
      backward(gaussians, features, ctx.tile_overlap_ranges, ctx.overlap_to_point,
              image_feature, grad_image_feature.contiguous(),
              grad_gaussians, grad_features, ctx.split_heuristic)

      return grad_gaussians, grad_features, None, None, None, None
    
  return _module_function

@beartype
def rasterize_with_tiles(gaussians2d: torch.Tensor, features: torch.Tensor,
                      overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                      image_size: Tuple[Integral, Integral], config: RasterConfig) -> RasterOut:
  """
  Rasterize an image given 2d gaussians, features and tile overlap information.
  Consider using rasterize instead to also compute tile overlap information.

  Parameters:
      gaussians2d: (N, 7)  packed gaussians, N is the number of gaussians
      features: (N, F)   features, F is the number of features
      tile_overlap_ranges: (TH * TW, 2) M is the number of tiles, 
        maps tile index to range of overlap indices (0..K]
      overlap_to_point: (K, )  K is the number of overlaps, 
        maps overlap index to point index (0..N]
      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

  Returns:
      RasterOut - namedtuple with fields:
        image: (H, W, F) torch tensor, where H, W are the image height and width, F is the features
        image_weight: (H, W) torch tensor, where H, W are the image height and width
        split_heuristic: (N, ) torch tensor, where N is the number of gaussians  
        visibility: (N, ) torch tensor, where N is the number of gaussians
  """
  _module_function = render_function(config, gaussians2d.requires_grad,
                                   features.requires_grad, features.shape[1],
                                   dtype=gaussians2d.dtype)

  image, image_weight, split_heuristic, visibility = _module_function.apply(
    gaussians2d, features, overlap_to_point, tile_overlap_ranges, image_size)
  
  return RasterOut(image, image_weight, split_heuristic, visibility)

def rasterize(gaussians2d: torch.Tensor, depth: torch.Tensor,
            features: torch.Tensor, image_size: Tuple[Integral, Integral],
            config: RasterConfig, use_depth16: bool = False) -> RasterOut:
  """
  Rasterize an image given 2d gaussians, features.

  Parameters:
      gaussians2d: (N, 7)  packed gaussians, N is the number of gaussians
      depth: (N, 1)   depths, N is the number of gaussians
      features: (N, F)   features, F is the number of features
      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

  Returns:
      RasterOut - namedtuple with fields:
        image: (H, W, F) torch tensor, where H, W are the image height and width, F is the features
        image_weight: (H, W) torch tensor, where H, W are the image height and width
        split_heuristic: (N, ) torch tensor, where N is the number of gaussians  
        visibility: (N, ) torch tensor, where N is the number of gaussians
  """
  assert gaussians2d.shape[0] == depth.shape[0] == features.shape[0], \
    f"Size mismatch: got {gaussians2d.shape}, {depth.shape}, {features.shape}"

  overlap_to_point, tile_overlap_ranges = map_to_tiles(
    gaussians2d, depth, image_size=image_size, config=config, use_depth16=use_depth16)
  
  return rasterize_with_tiles(
    gaussians2d, features,
    tile_overlap_ranges=tile_overlap_ranges.view(-1, 2),
    overlap_to_point=overlap_to_point,
    image_size=image_size,
    config=config
  )
