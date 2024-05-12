from functools import cache
from numbers import Integral
from typing import Tuple
from beartype import beartype
import numpy as np
import torch
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.mapper.tile_mapper import make_tile_mapper

from .grid_query import obb_grid_query


def gaussian_scale(config:RasterConfig):
  x = np.linspace(0, 4, 1000)
  y = np.exp(-(0.5 * x**2) ** config.beta)

  return x[np.argmax(y < 0.5 * config.alpha_threshold)]
  

@cache
def tile_mapper(config:RasterConfig):
  
  scale = gaussian_scale(config)
  grid_query = obb_grid_query(tile_size=config.tile_size, gaussian_scale=scale)
  return make_tile_mapper(grid_query, config)


@beartype
def map_to_tiles(primitives : torch.Tensor, 
                depth:torch.Tensor, 
                depth_range:Tuple[float, float],

                image_size:Tuple[Integral, Integral],
                config:RasterConfig
                
                ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps primitives to tiles, sorted by depth (front to back):
    Parameters:
    primitives: (N, K) representation of primitives (size K)
    depths: (N)  torch.Tensor of encoded depths (float32)
    depth_range: (near, far) tuple of floats

    image_size: (2, ) tuple of ints, (width, height)
    tile_config: configuration for tile mapper (tile_size etc.)

    Returns:
    overlap_to_point: (J, ) torch tensor, where J is the number of overlaps, maps overlap index to point index
    tile_ranges: (H, W, 2) torch tensor, where (H, W) is the tile shape, maps tile a range in overlap_to_point
    """

  mapper = tile_mapper(config)
  return mapper(primitives, depth, depth_range, image_size)

