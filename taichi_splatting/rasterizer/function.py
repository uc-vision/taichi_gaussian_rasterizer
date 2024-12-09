
from functools import cache
from typing import Optional
from taichi_splatting.mapper.tile_mapper import map_to_tiles
from taichi_splatting.taichi_queue import queued


from .forward import RasterConfig, forward_kernel
from .backward import backward_kernel

from numbers import Integral
from beartype.typing import Tuple, NamedTuple

import torch
from beartype import beartype
from taichi_splatting.taichi_lib.conversions import torch_taichi

RasterOut = NamedTuple('RasterOut', 
    [('image', torch.Tensor), 
     ('image_weight', torch.Tensor),
     ('point_heuristics', Optional[torch.Tensor]),
     ('visibility', Optional[torch.Tensor])])

@cache
def render_function(config:RasterConfig,
                    points_requires_grad:bool,
                    features_requires_grad:bool, 
                    feature_size:int,
                    dtype=torch.float32):
  
    
  forward = forward_kernel(config, feature_size=feature_size, dtype=torch_taichi[dtype])
  backward = backward_kernel(config, points_requires_grad,
                             features_requires_grad, 
                             feature_size, dtype=torch_taichi[dtype])
  
  forward = queued(forward)
  backward = queued(backward)


  class _module_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gaussians: torch.Tensor, features: torch.Tensor,
                overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
                image_size: Tuple[Integral, Integral]
                ) -> torch.Tensor:
        

      shape = (image_size[1], image_size[0])
      image_feature = torch.empty((*shape, features.shape[1]),
                                  dtype=dtype, device=features.device)
      image_alpha = torch.empty(shape, dtype=dtype, device=features.device)
      image_last_valid = torch.empty(shape, dtype=torch.int32, device=features.device)

      if config.compute_point_heuristics:
        point_heuristics = torch.zeros((gaussians.shape[0], 2), dtype=dtype, device=features.device)
      else:
        point_heuristics = torch.empty((0,2), dtype=dtype, device=features.device)

      if config.compute_visibility:
        visibility = torch.zeros((gaussians.shape[0]), dtype=dtype, device=features.device)
      else:
        visibility = torch.empty((0), dtype=dtype, device=features.device)

      forward(gaussians, features, 
        tile_overlap_ranges, overlap_to_point,
        image_feature, image_alpha, image_last_valid, visibility.unsqueeze(1))

      # Non differentiable parameters
      ctx.overlap_to_point = overlap_to_point
      ctx.tile_overlap_ranges = tile_overlap_ranges
      ctx.image_last_valid = image_last_valid
      ctx.image_alpha = image_alpha
      ctx.image_size = image_size
      ctx.point_heuristics = point_heuristics
      ctx.visibility = visibility

      ctx.mark_non_differentiable(image_alpha, image_last_valid, overlap_to_point, tile_overlap_ranges, visibility, point_heuristics)
      ctx.save_for_backward(gaussians, features)
    
            
      return image_feature, image_alpha, point_heuristics, visibility

    @staticmethod
    def backward(ctx, grad_image_feature:torch.Tensor, 
                 grad_alpha:torch.Tensor, grad_point_heuristics:torch.Tensor,
                 grad_visibility:torch.Tensor):
        
        gaussians, features = ctx.saved_tensors

        grad_gaussians = torch.zeros_like(gaussians)
        grad_features = torch.zeros_like(features)
  
        backward(gaussians, features, 
          ctx.tile_overlap_ranges, ctx.overlap_to_point,
          ctx.image_alpha, ctx.image_last_valid,
          grad_image_feature.contiguous(),
          grad_gaussians, grad_features, ctx.point_heuristics)


        return grad_gaussians, grad_features, None, None, None, None
    
      
  return _module_function


@beartype
def rasterize_with_tiles(gaussians2d: torch.Tensor, features: torch.Tensor,
              overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
              image_size: Tuple[Integral, Integral], config: RasterConfig
              ) -> RasterOut:
  """
  Rasterize an image given 2d gaussians, features and tile overlap information.
  Consider using rasterize instead to also compute tile overlap information.

  Parameters:
      gaussians2d: (N, 6)  packed gaussians, N is the number of gaussians
      features: (N, F)   features, F is the number of features
  compute_visibility:bool = False,

      tile_overlap_ranges: (TH * TW, 2) M is the number of tiles, 
        maps tile index to range of overlap indices (0..K]
      overlap_to_point: (K, )  K is the number of overlaps, 
        maps overlap index to point index (0..N]

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

    Returns:
      RasterOut - namedtuple with fields:
        image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
        image_weight: (H, W) torch tensor, where H, W are the image height and width
        point_heuristics: (N, 2) torch tensor, where N is the number of gaussians  
        visibility: (N, ) torch tensor, where N is the number of gaussians

  """
  _module_function = render_function(config, gaussians2d.requires_grad,
                                      features.requires_grad,
                                      features.shape[1],
                                      dtype=gaussians2d.dtype)

  image, image_weight, point_heuristics, visibility = _module_function.apply(gaussians2d, features, 
          overlap_to_point, tile_overlap_ranges, 
          image_size)
  
  return RasterOut(image, image_weight, point_heuristics, visibility)


def rasterize(gaussians2d:torch.Tensor, depth:torch.Tensor, 
                          features:torch.Tensor, image_size:Tuple[Integral, Integral],
                          config:RasterConfig,
                          use_depth16:bool = False) -> RasterOut:
    
    
  """
  Rasterize an image given 2d gaussians, features. 

  Parameters:
      gaussians2d: (N, 6)  packed gaussians, N is the number of gaussians
      depth:       (N, 1)   depths, N is the number of gaussians
      features:    (N, F)   features, F is the number of features

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization

    Returns:
      RasterOut - namedtuple with fields:
        image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
        image_weight: (H, W) torch tensor, where H, W are the image height and width
        point_heuristics: (N, 2) torch tensor, where N is the number of gaussians  
        visibility: (N, ) torch tensor, where N is the number of gaussians
  """

  assert gaussians2d.shape[0] == depth.shape[0] == features.shape[0], \
    f"Size mismatch: got {gaussians2d.shape}, {depth.shape}, {features.shape}"

  overlap_to_point, tile_overlap_ranges = map_to_tiles(gaussians2d, depth, 
    image_size=image_size, config=config, use_depth16=use_depth16)
  
  return rasterize_with_tiles(gaussians2d, features, 
    tile_overlap_ranges=tile_overlap_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
    image_size=image_size, config=config)
