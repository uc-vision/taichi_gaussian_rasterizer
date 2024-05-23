from functools import cache
import math
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2, vec2
import torch
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.taichi_lib.f32 import (Gaussian2D)
from taichi_splatting.cuda_lib import full_cumsum, segmented_sort_pairs

from taichi_splatting.taichi_lib.grid_query import make_grid_query
from taichi_splatting.taichi_lib.conversions import torch_taichi

def pad_to_tile(image_size: Tuple[Integral, Integral], tile_size: int):
  def pad(x):
    return int(math.ceil(x / tile_size) * tile_size)
 
  return tuple(pad(x) for x in image_size)


@cache
def tile_mapper(config:RasterConfig, depth_type):

  ti_depth_type = torch_taichi[depth_type]

  tile_size = config.tile_size
  grid_ops = make_grid_query(
    tile_size=tile_size, 
    gaussian_scale=config.gaussian_scale, 
    tight_culling=config.tight_culling)
  
  grid_query = grid_ops.grid_query
  

  @ti.kernel
  def count_tile_overlaps_kernel(
      gaussians: ti.types.ndarray(Gaussian2D.vec, ndim=1), 
      image_size: ivec2,
      tile_counts: ti.types.ndarray(ti.i32, ndim=2), 
  ):
      ti.loop_config(block_dim=1024)

      for idx in range(gaussians.shape[0]):
          lower, upper = grid_ops.gaussian_tile_bounds(gaussians[idx], image_size)
          for tile_uv in ti.grouped(ti.ndrange((lower.x, upper.x), (lower.y, upper.y))):
              ti.atomic_add(tile_counts[tile_uv.y, tile_uv.x], 1)


  def count_tile_overlaps(gaussians:torch.Tensor, image_size:Tuple[Integral, Integral]):
    tile_counts = torch.zeros((image_size[1] // tile_size, image_size[0] // tile_size), 
                              dtype=torch.int32, device=gaussians.device)
    
    count_tile_overlaps_kernel(gaussians, ivec2(image_size), tile_counts)
    return tile_counts




  @ti.kernel
  def partition_tiles_kernel(
      depths: ti.types.ndarray(ti_depth_type, ndim=1),  # (M)

      gaussians : ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M)
      tile_offsets: ti.types.ndarray(ti.i32, ndim=2),  # (H, W)

      image_size: ivec2,

      # output - precise tile counts
      tile_counts : ti.types.ndarray(ti.i32, ndim=2),  # (H, W)

      overlap_depths: ti.types.ndarray(ti_depth_type, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

  ):

    ti.loop_config(block_dim=128)
    for point_idx in range(gaussians.shape[0]):
      query = grid_query(gaussians[point_idx], image_size)
      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):
        depth = depths[point_idx]

        if query.test_tile(tile_uv):
          tile = tile_uv + query.min_tile

          tile_idx = ti.atomic_add(tile_counts[tile.y, tile.x], 1)
          key_idx = tile_offsets[tile.y, tile.x] + tile_idx

          overlap_depths[key_idx] = depth # sort based on tile_id, depth
          overlap_to_point[key_idx] = point_idx # map overlap index back to point index




  def f(gaussians : torch.Tensor, depths:torch.Tensor, 
        image_size:Tuple[Integral, Integral]):
    
    image_size = pad_to_tile(image_size, tile_size)

    with torch.no_grad():
      # compute an approximate count of gaussians per tile using bounding box only
      max_overlap_counts = count_tile_overlaps(gaussians, image_size)      

          # compute offsets into an overlap array for each tile
      overlap_sums, total_overlaps = full_cumsum(max_overlap_counts.view(-1))
      overlap_offsets = overlap_sums[:-1].view(max_overlap_counts.shape)

      # allocate overlap array and sort key (depth)
      overlap_depths = torch.empty((total_overlaps,), dtype=depth_type, device=gaussians.device)
      overlap_to_point = torch.empty((total_overlaps,), dtype=torch.int32, device=gaussians.device)

      # allocate space for precise tile counts
      overlap_counts = torch.zeros_like(max_overlap_counts)

      # fill in indices and depths into overlap keys and array
      partition_tiles_kernel(depths, gaussians, overlap_offsets, 
                             ivec2(image_size), overlap_counts, overlap_depths, overlap_to_point)
      
      # sort the ranges of each tile by depth (in parallel)
      tile_start = overlap_offsets
      tile_end = tile_start + overlap_counts

      _, overlap_to_point = segmented_sort_pairs(overlap_depths, overlap_to_point, 
                            tile_start.view(-1).to(torch.int64), tile_end.view(-1).to(torch.int64))
    
      tile_overlap_ranges = torch.stack((tile_start, tile_end), dim=2)
      return overlap_to_point, tile_overlap_ranges
      
  return f


@beartype
def map_to_tiles(gaussians : torch.Tensor, 
                 encoded_depth:torch.Tensor, 
                 image_size:Tuple[Integral, Integral],
                 config:RasterConfig
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps guassians to tiles, sorted by depth (front to back):
    Parameters:
     gaussians: (N, 6) torch.Tensor of packed gaussians, N is the number of gaussians
     encoded_depth: (N)  torch.Tensor (i16 or i32) of encoded depths
     image_size: (2, ) tuple of ints, (width, height)
     tile_config: configuration for tile mapper (tile_size etc.)

    Returns:
     overlap_to_point: (K, ) torch tensor, where K is the number of overlaps, maps overlap index to point index
     tile_ranges: (2, M) torch tensor, where M is the number of tiles, maps tile index to range of overlap indices
    """


  mapper = tile_mapper(config, encoded_depth.dtype)
  return mapper(gaussians, encoded_depth.contiguous(), 
                image_size=image_size)