
from functools import cache
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib import get_library



@cache
def forward_kernel(config: RasterConfig, feature_size: int, dtype=ti.f32):

  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D

  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  gaussian_pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf


  @ti.kernel
  def _forward_kernel(
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, 6)
      point_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)
      
      # (TH, TW, 2) the start/end (0..K] index of ranges in the overlap_to_point array
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      # (K) ranges of points mapping to indexes into points list
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
      
      # outputs
      image_feature: ti.types.ndarray(feature_vec, ndim=2),  # (H, W, F)
      # needed for backward
      image_alpha: ti.types.ndarray(dtype, ndim=2),       # H, W
      image_last_valid: ti.types.ndarray(ti.i32, ndim=2),  # H, W
  ):

    camera_height, camera_width = image_feature.shape

    # round up
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size

    # put each tile_size * tile_size tile in the same CUDA thread group (block)
    # tile_id is the index of the tile in the (tiles_wide x tiles_high) grid
    # tile_idx is the index of the pixel in the tile
    # pixels are blocked first by tile_id, then by tile_idx into (8x4) warps
    
    ti.loop_config(block_dim=(tile_area))

    
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):

      pixel = tiling.tile_transform(tile_id, tile_idx, tile_size, (1, 1), tiles_wide)
      pixelf = ti.cast(pixel, dtype) + 0.5

      # The initial value of accumulated alpha (initial value of accumulated multiplication)
      T_i = dtype(1.0)
      accum_feature = feature_vec(0.)

      # open the shared memory
      tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
      tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)

      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset

      num_point_groups = (tile_point_count + ti.static(tile_area - 1)) // tile_area
      pixel_saturated = False
      last_point_idx = start_offset


      # Loop through the range in groups of tile_area
      for point_group_id in range(num_point_groups):

        ti.simt.block.sync()

        # The offset of the first point in the group
        group_start_offset = start_offset + point_group_id * tile_area

        # each thread in a block loads one point into shared memory
        # then all threads in the block process those points sequentially
        load_index = group_start_offset + tile_idx

        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]

  
          tile_point[tile_idx] = points[point_idx]
          tile_feature[tile_idx] = point_features[point_idx]


        ti.simt.block.sync()

        max_point_group_offset: ti.i32 = ti.min(
            tile_area, tile_point_count - point_group_id * tile_area)

        # in parallel across a block, render all points in the group
        for in_group_idx in range(max_point_group_offset):
          if pixel_saturated:
            break

          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])
          gaussian_alpha = gaussian_pdf(pixelf, mean, axis, sigma)

          alpha = point_alpha * gaussian_alpha

            
          # from paper: we skip any blending updates with 𝛼 < 𝜖 (configurable as alpha_threshold)
          if alpha < ti.static(config.alpha_threshold):
            continue

          alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))
          # from paper: before a Gaussian is included in the forward rasterization
          # pass, we compute the accumulated opacity if we were to include it
          # and stop front-to-back blending before it can exceed config.saturate_threshold.

          next_T_i = T_i * (1 - alpha)
          if next_T_i < ti.static(1 - config.saturate_threshold):
            pixel_saturated = True
            continue  # somehow faster than directly breaking
          last_point_idx = group_start_offset + in_group_idx + 1

          # weight = alpha * T_i
          accum_feature += tile_feature[in_group_idx] * alpha * T_i
          T_i = next_T_i


        # end of point group loop
      # end of point group id loop

      if pixel.x < camera_width and pixel.y < camera_height:
        image_feature[pixel.y, pixel.x] = accum_feature

        # No need to accumulate a normalisation factor as it is exactly 1 - T_i
        image_alpha[pixel.y, pixel.x] = 1. - T_i    
        image_last_valid[pixel.y, pixel.x] = last_point_idx

    # end of pixel loop

  return _forward_kernel




