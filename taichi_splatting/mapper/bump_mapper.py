from functools import cache
import math
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import taichi as ti
from taichi.math import ivec2
import torch
from taichi_splatting import cuda_lib
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.taichi_lib.concurrent import add, block_reduce_i32, warp_allocate

from taichi_splatting.taichi_lib.f32 import (GaussianConic)
from taichi_splatting.taichi_lib.grid_query import make_grid_query
from taichi_splatting.taichi_lib.conversions import torch_taichi

def pad_to_tile(image_size: Tuple[Integral, Integral], tile_size: int):
  def pad(x):
    return int(math.ceil(x / tile_size) * tile_size)
 
  return tuple(pad(x) for x in image_size)


@cache
def tile_mapper(config:RasterConfig, depth_type=torch.int32):

  if depth_type == torch.int32:
    max_tile = 65535
    key_type = torch.int64
    end_sort_bit = 48

    @ti.func
    def make_sort_key(depth:ti.i32, tile_id:ti.i32) -> ti.i64:
        return ti.cast(depth, ti.i64) | (ti.cast(tile_id, ti.i64) << 32)
  
    @ti.func
    def get_tile_id(key:ti.i64) -> ti.i32:
      return ti.cast(key >> 32, ti.i32)


  elif depth_type == torch.int16:
    max_tile = 65535
    key_type = torch.int32
    end_sort_bit = 32

    @ti.func
    def make_sort_key(depth:ti.i16, tile_id:ti.i32):
        depthu = ti.cast(depth, ti.i32) + 32767

        key_u32 = ti.cast(depthu, ti.u32) | (ti.cast(tile_id, ti.u32) << 16)
        return ti.bit_cast(key_u32, ti.i32)
  
    @ti.func
    def get_tile_id(key:ti.i32):
      key_u32 = ti.bit_cast(key, ti.u32)
      return ti.cast(key_u32 >> 16, ti.i32)

  else:
    raise ValueError(f"depth_type {depth_type} not supported")
  
  tile_size = config.tile_size
  grid_ops = make_grid_query(
    tile_size=tile_size, 
    gaussian_scale=config.gaussian_scale, 
    tight_culling=config.tight_culling)
  
  grid_query = grid_ops.grid_query
  
  
  @ti.kernel
  def tile_overlaps_kernel(
      gaussians: ti.types.ndarray(GaussianConic.vec, ndim=1),  
      image_size: ivec2
  ) -> ti.i32:
      
      ti.loop_config(block_dim=1024)
      counts = 0
      for idx in range(gaussians.shape[0]):
          lower, upper = grid_ops.gaussian_tile_bounds(gaussians[idx], image_size)        
          tile_range = upper - lower

          counts += tile_range.x * tile_range.y
      return counts
 

  @ti.kernel
  def find_ranges_kernel(
      sorted_keys: ti.types.ndarray(torch_taichi[key_type], ndim=1),  # (M)
      # output tile_ranges (tile id -> start, end)
      tile_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),   
  ):  
    ti.loop_config(block_dim=1024)
    for idx in range(sorted_keys.shape[0]):
        tile_id = get_tile_id(sorted_keys[idx])

        if idx == sorted_keys.shape[0] - 1:
          tile_ranges[tile_id][1] = idx             # end last tile range

        else:
          next_tile_id = get_tile_id(sorted_keys[idx + 1])
          if tile_id != next_tile_id:

            tile_ranges[tile_id][1] = idx + 1       # end tile range
            tile_ranges[next_tile_id][0] = idx + 1  # start next tile range
            
    

  @ti.kernel
  def generate_sort_keys_kernel(
      depth: ti.types.ndarray(torch_taichi[depth_type], ndim=1),  # (M)
      gaussians : ti.types.ndarray(GaussianConic.vec, ndim=1),  # (M)

      image_size: ivec2,

      # outputs
      overlap_sort_key: ti.types.ndarray(torch_taichi[key_type], ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
  ) -> ti.i32:
    tiles_wide = image_size.x // tile_size
    total_overlaps = 0

    ti.loop_config(block_dim=256)
    for idx in range(gaussians.shape[0]):

      encoded_depth = depth[idx]
      query = grid_query(gaussians[idx], image_size)

      count = 0
      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):  
          count += query.test_tile(tile_uv)

      # block_total = block_reduce_i32(count, add, ti.atomic_add, 0)

      # bump allocate spaces for overlaps for this gaussian
      offset = warp_allocate(total_overlaps, count)
      # offset = ti.atomic_add(total_overlaps, count)
 
      for tile_uv in ti.grouped(ti.ndrange(*query.tile_span)):  
        if query.test_tile(tile_uv):

          tile = tile_uv + query.min_tile      
          tile_id = tile.x + tile.y * tiles_wide

          # sort based on tile_id, depth
          overlap_sort_key[offset] = make_sort_key(encoded_depth, tile_id) 
          overlap_to_point[offset] = idx # map overlap index back to point index
          offset += 1

    return total_overlaps

  def sort_cull_tiles(depths:torch.Tensor, gaussians:torch.Tensor, max_overlaps:int, image_size):

    overlap_key = torch.empty((max_overlaps, ), dtype=key_type, device=gaussians.device)
    overlap_to_point = torch.empty((max_overlaps, ), dtype=torch.int32, device=gaussians.device)

    total_overlaps = generate_sort_keys_kernel(depths, gaussians, image_size,
                                overlap_key, overlap_to_point)
    

    return cuda_lib.radix_sort_pairs(
       overlap_key[:total_overlaps], overlap_to_point[:total_overlaps], end_bit=end_sort_bit)
  


  def f(gaussians : torch.Tensor, depths:torch.Tensor, image_size:Tuple[Integral, Integral]):

    image_size = pad_to_tile(image_size, tile_size)
    tile_shape = (image_size[1] // tile_size, image_size[0] // tile_size)

    assert tile_shape[0] * tile_shape[1] < max_tile, \
      f"tile dimensions {tile_shape} for image size {image_size} exceed maximum tile count (16 bit id), try increasing tile_size" 

    with torch.no_grad():
      max_overlaps = tile_overlaps_kernel(gaussians, ivec2(image_size))

      # This needs to be initialised to zeros (not empty)
      # as sometimes there are no overlaps for a tile
      tile_ranges = torch.zeros((*tile_shape, 2), dtype=torch.int32, device=gaussians.device)

      if max_overlaps > 0:
        overlap_key, overlap_to_point = sort_cull_tiles(
          depths, gaussians, max_overlaps, image_size)
        
        if overlap_key.shape[0] > 0:
          find_ranges_kernel(overlap_key, tile_ranges.view(-1, 2))

      return overlap_to_point, tile_ranges
      
  return f


@beartype
def map_to_tiles(gaussians : torch.Tensor, encoded_depth:torch.Tensor, 
                 image_size:Tuple[Integral, Integral],
                 config:RasterConfig
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
  """ maps guassians to tiles, sorted by depth (front to back):
    Parameters:
     gaussians: (N, 6) torch.Tensor of packed gaussians, N is the number of gaussians
     encoded_depths: (N)  torch.Tensor of encoded depths (int32)
     image_size: (2, ) tuple of ints, (width, height)
     tile_config: configuration for tile mapper (tile_size etc.)

    Returns:
     overlap_to_point: (K, ) torch tensor, where K is the number of overlaps, maps overlap index to point index
     tile_ranges: (M, 2) torch tensor, where M is the number of tiles, maps tile index to range of overlap indices
    """

  
  mapper = tile_mapper(config, depth_type=encoded_depth.dtype)
  return mapper(gaussians, encoded_depth.contiguous(), image_size)