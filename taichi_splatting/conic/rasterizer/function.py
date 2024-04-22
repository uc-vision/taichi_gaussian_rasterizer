
from functools import cache
from taichi_splatting.mapper.tile_mapper import map_to_tiles
from taichi_splatting.taichi_lib.concurrent import WARP_SIZE


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
     ('point_split_heuristics', torch.Tensor) ])

@cache
def render_function(config:RasterConfig,
                    points_requires_grad:bool,
                    features_requires_grad:bool, 
                    compute_split_heuristics:bool,
                    feature_size:int,
                    dtype=torch.float32):
  
    
  forward = forward_kernel(config, feature_size=feature_size, dtype=torch_taichi[dtype])
  backward = backward_kernel(config, points_requires_grad,
                             features_requires_grad, 
                             compute_split_heuristics, feature_size, dtype=torch_taichi[dtype])

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

      point_split_heuristics = torch.zeros( (gaussians.shape[0], 2) if compute_split_heuristics else (0, 2), 
                                 dtype=dtype, device=features.device)
      

      forward(gaussians, features, 
        tile_overlap_ranges, overlap_to_point,
        image_feature, image_alpha, image_last_valid)

      # Non differentiable parameters
      ctx.overlap_to_point = overlap_to_point
      ctx.tile_overlap_ranges = tile_overlap_ranges
      ctx.image_last_valid = image_last_valid
      ctx.image_alpha = image_alpha
      ctx.image_size = image_size
      ctx.point_split_heuristics = point_split_heuristics

      ctx.mark_non_differentiable(image_alpha, image_last_valid, point_split_heuristics, overlap_to_point, tile_overlap_ranges)
      ctx.save_for_backward(gaussians, features)
            
      return image_feature, image_alpha, point_split_heuristics

    @staticmethod
    def backward(ctx, grad_image_feature:torch.Tensor, 
                 grad_alpha:torch.Tensor, grad_point_split_heuristics:torch.Tensor):
        gaussians, features = ctx.saved_tensors

        grad_gaussians = torch.zeros_like(gaussians)
        grad_features = torch.zeros_like(features)
  
        # with restore_grad(gaussians, features):

        backward(gaussians, features, 
          ctx.tile_overlap_ranges, ctx.overlap_to_point,
          ctx.image_alpha, ctx.image_last_valid,
          grad_image_feature.contiguous(),
          grad_gaussians, grad_features, ctx.point_split_heuristics)

        return grad_gaussians, grad_features, None, None, None, None
  return _module_function





@beartype
def rasterize_with_tiles(gaussians2d: torch.Tensor, features: torch.Tensor,
              overlap_to_point: torch.Tensor, tile_overlap_ranges: torch.Tensor,
              image_size: Tuple[Integral, Integral], config: RasterConfig,
              compute_split_heuristics:bool=False
              ) -> RasterOut:
  """
  Rasterize an image given 2d gaussians, features and tile overlap information.
  Consider using rasterize instead to also compute tile overlap information.

  Parameters:
      gaussians2d: (N, 6)  packed gaussians, N is the number of gaussians
      features: (N, F)   features, F is the number of features

      tile_overlap_ranges: (TH * TW, 2) M is the number of tiles, 
        maps tile index to range of overlap indices (0..K]
      overlap_to_point: (K, )  K is the number of overlaps, 
        maps overlap index to point index (0..N]

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization
      compute_split_heuristics: bool - whether to compute the visibility for each point in the image

    Returns:
      RasterOut - namedtuple with fields:
        image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
        image_weight: (H, W) torch tensor, where H, W are the image height and width
        point_split_heuristics: (N, ) torch tensor, where N is the number of gaussians  
  """
  _module_function = render_function(config, gaussians2d.requires_grad,
                                      features.requires_grad,
                                      compute_split_heuristics, 
                                      features.shape[1],
                                      dtype=gaussians2d.dtype)

  image, image_weight, point_split_heuristics = _module_function.apply(gaussians2d, features, 
          overlap_to_point, tile_overlap_ranges, 
          image_size)
  
  return RasterOut(image, image_weight, point_split_heuristics)


def rasterize(gaussians2d:torch.Tensor, encoded_depths:torch.Tensor, 
                          features:torch.Tensor, image_size:Tuple[Integral, Integral],
                          config:RasterConfig, compute_split_heuristics:bool=False) -> RasterOut:
    
    
  """
  Rasterize an image given 2d gaussians, features. 

  Parameters:
      gaussians2d: (N, 6)  packed gaussians, N is the number of gaussians
      encoded_depths: (N )  encoded depths, N is the number of gaussians
      features: (N, F)   features, F is the number of features

      image_size: (2, ) tuple of ints, (width, height)
      config: Config - configuration parameters for rasterization
      compute_split_heuristics: bool - whether to compute the visibility for each point in the image

    Returns:
      RasterOut - namedtuple with fields:
        image: (H, W, F) torch tensor, where H, W are the image height and width, F is the number of features
        image_weight: (H, W) torch tensor, where H, W are the image height and width
        point_split_heuristics: (N, ) torch tensor, where N is the number of gaussians  
  """

  assert gaussians2d.shape[0] == encoded_depths.shape[0] == features.shape[0], \
    f"Size mismatch: got {gaussians2d.shape}, {encoded_depths.shape}, {features.shape}"

  # render with padding to tile_size, later crop back to original size
  overlap_to_point, tile_overlap_ranges = map_to_tiles(gaussians2d, encoded_depths, 
    image_size=image_size, config=config)
  
  return rasterize_with_tiles(gaussians2d, features, 
    tile_overlap_ranges=tile_overlap_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
    image_size=image_size, config=config, compute_split_heuristics=compute_split_heuristics)
