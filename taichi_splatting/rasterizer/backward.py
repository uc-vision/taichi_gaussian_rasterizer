from functools import cache
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib.concurrent import warp_add_vector_32, warp_add_vector_64

from taichi_splatting.taichi_lib import get_library


@cache
def backward_kernel(config: RasterConfig,
                   points_requires_grad: bool,
                   features_requires_grad: bool, 
                   feature_size: int,
                   dtype=ti.f32,
                   eps=1e-8):
  
  # Load library functions
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D
  vec2 = lib.vec2

  # Configure data types
  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  # Select implementations based on dtype
  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64
  pdf_with_grad = lib.gaussian_pdf_antialias_with_grad if config.antialias else lib.gaussian_pdf_with_grad
  pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf


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
      image_alpha: ti.types.ndarray(dtype, ndim=2),                  # [H, W] alpha values

      # Input image gradients
      grad_image_feature: ti.types.ndarray(feature_vec, ndim=2),     # [H, W, F] gradient of output features

      # Output point gradients
      grad_points: ti.types.ndarray(Gaussian2D.vec, ndim=1),         # [N, 7] gradient of gaussian parameters
      grad_features: ti.types.ndarray(feature_vec, ndim=1),          # [N, F] gradient of gaussian features

      # Output point heuristics
      point_heuristics: ti.types.ndarray(vec2, ndim=1),              # [N, 2] point statistics
  ):
    camera_height, camera_width = grad_image_feature.shape
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size

    ti.loop_config(block_dim=(tile_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):
      pixel = tiling.tile_transform(tile_id, tile_idx, 
                                   tile_size, (1, 1), tiles_wide)
      pixelf = ti.cast(pixel, dtype) + 0.5

      grad_pixel_feature = feature_vec(0.) 
      # Initialize accumulators for pixel
      remaining_features = feature_vec(0.0)
      total_weight = 1.0

      # Check bounds and initialize remaining features
      in_bounds = pixel.y < camera_height and pixel.x < camera_width
      if in_bounds:
        remaining_features = image_feature[pixel.y, pixel.x]
        grad_pixel_feature = grad_image_feature[pixel.y, pixel.x]
        total_weight = 0.0

      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset
      num_point_groups = (tile_point_count + tile_area - 1) // tile_area



      # Open shared memory arrays
      tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
      tile_point_id = ti.simt.block.SharedArray((tile_area, ), dtype=ti.i32)
      tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)

      tile_grad_point = (ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
                         if ti.static(points_requires_grad) else None)
      
      tile_grad_feature = (ti.simt.block.SharedArray((tile_area,), dtype=feature_vec)
                          if ti.static(features_requires_grad) else None)

      tile_point_heuristics = (ti.simt.block.SharedArray((tile_area,), dtype=vec2) 
                              if ti.static(config.compute_point_heuristics) else None)

      for point_group_id in range(num_point_groups):
        if ti.simt.block.sync_all_nonzero(ti.i32(total_weight >= config.saturate_threshold)):
          break

        # Load points into shared memory
        group_start_offset = start_offset + point_group_id * tile_area
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
          if ti.static(config.compute_point_heuristics):
            tile_point_heuristics[tile_idx] = vec2(0.0)

        ti.simt.block.sync()

        remaining_points = tile_point_count - point_group_id
        
        # Process all points in group for pixel
        for in_group_idx in range(min(tile_area, remaining_points)):

          saturated = total_weight >= ti.static(config.saturate_threshold)
          if ti.simt.warp.all_nonzero(ti.u32(0xffffffff), ti.i32(saturated)):
            break

          grad_point = Gaussian2D.vec(0.0)
          gaussian_point_heuristics = vec2(0.0)
          grad_feature = feature_vec(0.0)


          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])
          gaussian_alpha = pdf(pixelf, mean, axis, sigma)
          alpha = gaussian_alpha * point_alpha

          has_grad = alpha > config.alpha_threshold and not saturated
          if has_grad:

            # Compute gaussian gradients
            _, dp_dmean, dp_daxis, dp_dsigma = pdf_with_grad(pixelf, mean, axis, sigma)
            weight = alpha * (1.0 - total_weight)

            # Accumulate total hits and subtract accumulated features
            total_weight += weight            
            remaining_features -= tile_feature[in_group_idx] * weight


            # Compute feature difference between point  and remaining features (from points behind this one)
            feature_diff = tile_feature[in_group_idx] * weight - remaining_features / (1.0 - total_weight + eps)


            alpha_grad_from_feature = feature_diff * grad_pixel_feature
            alpha_grad = alpha_grad_from_feature.sum()
            




            # Compute gradients
            if ti.static(points_requires_grad):
              grad_point = alpha_grad * Gaussian2D.to_vec(point_alpha * dp_dmean, 
                    point_alpha * dp_daxis, point_alpha * dp_dsigma, gaussian_alpha * alpha_grad)
            
            if ti.static(config.compute_point_heuristics):
              gaussian_point_heuristics = vec2(weight, lib.l1_norm( alpha_grad * point_alpha * dp_dmean))

            if ti.static(features_requires_grad):
              grad_feature = weight * grad_pixel_feature


          # Check if any thread in the warp has a gradient
          if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(has_grad)):
            # Accumulate gradients in shared memory across the warp
            if ti.static(points_requires_grad):
              warp_add_vector(tile_grad_point[in_group_idx], grad_point)
  
            if ti.static(features_requires_grad):
              warp_add_vector(tile_grad_feature[in_group_idx], grad_feature)

            if ti.static(config.compute_point_heuristics):
              warp_add_vector(tile_point_heuristics[in_group_idx], gaussian_point_heuristics)

        ti.simt.block.sync()

        if load_index < end_offset:
          point_idx = tile_point_id[tile_idx]
          
          # Write gradients to global memory
          if ti.static(points_requires_grad):
            ti.atomic_add(grad_points[point_idx], tile_grad_point[tile_idx])

          if ti.static(features_requires_grad):
            ti.atomic_add(grad_features[point_idx], tile_grad_feature[tile_idx])

          if ti.static(config.compute_point_heuristics):
            ti.atomic_add(point_heuristics[point_idx], tile_point_heuristics[tile_idx])

  return _backward_kernel








