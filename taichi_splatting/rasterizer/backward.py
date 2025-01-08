from functools import cache
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib.concurrent import warp_add_vector_32, warp_add_vector_64

from taichi_splatting.taichi_lib import get_library

from taichi_splatting.rasterizer.tiling import WARP_SIZE


@cache
def backward_kernel(config: RasterConfig,
                   points_requires_grad: bool,
                   features_requires_grad: bool, 
                   feature_size: int,
                   dtype=ti.f32):
  
  # Load library functions
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D
  vec2 = lib.vec2
  vec1 = lib.vec1

  # Configure data types
  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size
  
  thread_pixels = config.pixel_stride[0] * config.pixel_stride[1]
  block_area = tile_area // thread_pixels
  
  assert block_area >= WARP_SIZE, \
    f"pixel_stride {config.pixel_stride} and tile_size {config.tile_size} must allow at least one warp sized ({WARP_SIZE}) tile"

  # each thread is responsible for a small tile of pixels
  pixel_tile = tuple([ (i, 
            (i % config.pixel_stride[0],
            i // config.pixel_stride[0]))
              for i in range(thread_pixels) ])

  # types for each thread to keep state in it's tile of pixels
  thread_features = ti.types.matrix(thread_pixels, feature_size, dtype=dtype)
  thread_vector = ti.types.vector(thread_pixels, dtype=dtype)

  # Select implementations based on dtype
  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64
  pdf_with_grad = lib.gaussian_pdf_antialias_with_grad if config.antialias else lib.gaussian_pdf_with_grad
  # pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf


  @ti.kernel
  def _backward_kernel(
      # Input tensors
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),              # [N, 7] 2D gaussian parameters
      point_features: ti.types.ndarray(feature_vec, ndim=1),         # [N, F] gaussian features

      # Tile data structures
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),  # [T] start/end range of overlapping points
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),            # [P] mapping from overlap index to point index

      # Image buffers
      image_feature: ti.types.ndarray(feature_vec, ndim=2),          # [H, W, F] output features

      # Input image gradients
      grad_image_feature: ti.types.ndarray(feature_vec, ndim=2),     # [H, W, F] gradient of output features

      # Output point gradients
      grad_points: ti.types.ndarray(Gaussian2D.vec, ndim=1),         # [N, 7] gradient of gaussian parameters
      grad_features: ti.types.ndarray(feature_vec, ndim=1),          # [N, F] gradient of gaussian features

      # Output point heuristics
      point_heuristic: ti.types.ndarray(vec2, ndim=1),              # [N] point densify heuristic
  ):
    camera_height, camera_width = grad_image_feature.shape
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size


    ti.loop_config(block_dim=(block_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, block_area):
      pixel_base = tiling.tile_transform(tile_id, tile_idx, 
                                   tile_size, config.pixel_stride, tiles_wide)

      # Shared memory arrays
      tile_point = ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
      tile_point_id = ti.simt.block.SharedArray((block_area, ), dtype=ti.i32)
      tile_feature = ti.simt.block.SharedArray((block_area, ), dtype=feature_vec)

      tile_grad_point = (ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
                         if ti.static(points_requires_grad) else None)
      
      tile_grad_feature = (ti.simt.block.SharedArray((block_area,), dtype=feature_vec)
                          if ti.static(features_requires_grad) else None)

      tile_point_heuristics = (ti.simt.block.SharedArray((block_area,), dtype=vec2) 
                              if ti.static(config.compute_point_heuristic) else None)
      

      # Per-thread state for each pixel in tile
      grad_pixel_feature = thread_features(0.)
      remaining_features = thread_features(0.)
      total_weight = thread_vector(1.0)

      # Initialize per-pixel state
      for i, offset in ti.static(pixel_tile):
        pixel = ti.math.ivec2(offset) + pixel_base
        
        if pixel.y < camera_height and pixel.x < camera_width:
          remaining_features[i,:] = image_feature[pixel.y, pixel.x]
          grad_pixel_feature[i,:] = grad_image_feature[pixel.y, pixel.x]
          total_weight[i] = 0.0


      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset
      num_point_groups = tiling.round_up(tile_point_count, block_area)

      for point_group_id in range(num_point_groups):
        # Check if all pixels in tile are saturated
        if ti.simt.block.sync_all_nonzero(ti.i32(total_weight.min() >= ti.static(config.saturate_threshold))):
          break

        # Load points into shared memory
        group_start_offset = start_offset + point_group_id * block_area
        load_index = group_start_offset + tile_idx

        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]
          tile_point[tile_idx] = points[point_idx]
          tile_feature[tile_idx] = point_features[point_idx]
          tile_point_id[tile_idx] = point_idx

          if ti.static(points_requires_grad):
            tile_grad_point[tile_idx] = Gaussian2D.vec(0.0)
          if ti.static(features_requires_grad):
            tile_grad_feature[tile_idx] = feature_vec(0.0)
          if ti.static(config.compute_point_heuristic):
            tile_point_heuristics[tile_idx] = vec2(0.0)


        ti.simt.block.sync()

        remaining_points = tile_point_count - point_group_id 

        # Process all points in group for each pixel in tile
        for in_group_idx in range(min(block_area, remaining_points)):
          if ti.simt.warp.all_nonzero(ti.u32(0xffffffff), ti.i32(total_weight.min() >= ti.static(config.saturate_threshold))):
            break
          
          grad_point = Gaussian2D.vec(0.0)
          gaussian_point_heuristics = vec2(0.0)
          grad_feature = feature_vec(0.0)

          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])
          has_grad = False

          # Process all pixels in tile for current point
          for i, offset in ti.static(pixel_tile):
            pixel_saturated = total_weight[i] >= ti.static(config.saturate_threshold)

            pixelf = ti.cast(pixel_base + ti.math.ivec2(offset), dtype) + 0.5
            gaussian_alpha, dp_dmean, dp_daxis, dp_dsigma = pdf_with_grad(pixelf, mean, axis, sigma)
            alpha = point_alpha * gaussian_alpha

            if alpha > ti.static(config.alpha_threshold) and not pixel_saturated:
              has_grad = True
              
              alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))
              feature = tile_feature[in_group_idx]     

              T_i = (1.0 - total_weight[i])  # transmisivity (remaining weight)
              weight = alpha * T_i           # pre-multiplied alpha 

              # Update pixel state
              total_weight[i] += weight               
              remaining_features[i,:] -= feature * weight

              # Compute feature difference
              feature_diff = feature * T_i - remaining_features[i,:] / (1.0 - alpha)
              alpha_grad_from_feature = feature_diff * grad_pixel_feature[i,:]
              alpha_grad = alpha_grad_from_feature.sum()

              alpha_alpha_grad = point_alpha * alpha_grad
              pos_grad = alpha_alpha_grad * dp_dmean

              # Accumulate gradients
              if ti.static(points_requires_grad):
                grad_point += Gaussian2D.to_vec(
                  pos_grad, alpha_alpha_grad * dp_daxis, 
                  alpha_alpha_grad * dp_dsigma, 
                  gaussian_alpha * alpha_grad)
              
              if ti.static(config.compute_point_heuristic):
                gaussian_point_heuristics += vec2(
                  alpha_alpha_grad ** 2,
                  ti.abs(pos_grad).sum()
                )

              if ti.static(features_requires_grad):
                grad_feature += weight * grad_pixel_feature[i,:]

          # Accumulate gradients across warps if any pixel had gradients
          if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(has_grad)):
            if ti.static(points_requires_grad):
              warp_add_vector(tile_grad_point[in_group_idx], grad_point)

            if ti.static(features_requires_grad):
              warp_add_vector(tile_grad_feature[in_group_idx], grad_feature)

            if ti.static(config.compute_point_heuristic):
              warp_add_vector(tile_point_heuristics[in_group_idx], gaussian_point_heuristics)

        ti.simt.block.sync()

        # Write accumulated gradients to global memory
        
        if (load_index < end_offset):
          point_idx = tile_point_id[tile_idx]
          
          if ti.static(points_requires_grad):
            ti.atomic_add(grad_points[point_idx], tile_grad_point[tile_idx])

          if ti.static(features_requires_grad):
            ti.atomic_add(grad_features[point_idx], tile_grad_feature[tile_idx])

          if ti.static(config.compute_point_heuristic):
            ti.atomic_add(point_heuristic[point_idx], tile_point_heuristics[tile_idx])


  return _backward_kernel








