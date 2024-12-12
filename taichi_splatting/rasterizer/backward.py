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
                   dtype=ti.f32):
  
  lib = get_library(dtype)
  Gaussian2D = lib.Gaussian2D
  vec2 = lib.vec2

  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  # Match backward.py pattern for pixel tiling
  thread_pixels = config.pixel_stride[0] * config.pixel_stride[1]
  block_area = tile_area // thread_pixels

  # Create batch types for accumulation
  thread_features = ti.types.matrix(thread_pixels, feature_size, dtype=dtype)
  thread_i32 = ti.types.vector(thread_pixels, dtype=ti.i32)

  # Create pixel tile mapping
  pixel_tile = tuple([ (i, 
                       (i % config.pixel_stride[0],
                        i // config.pixel_stride[0]))
                      for i in range(thread_pixels) ])

  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64
  gaussian_pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf

  @ti.func
  def decode_hit(hit: ti.u32):
    return hit >> 6, hit & 0x3F


  @ti.kernel
  def _backward_kernel(
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),
      point_features: ti.types.ndarray(feature_vec, ndim=1),
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),

      image_hits: ti.types.ndarray(ti.u32, ndim=3),

      # input gradients
      grad_image_feature: ti.types.ndarray(feature_vec, ndim=2),  # (H, W, F)

      # output gradients
      grad_points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, C)
      grad_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)

      point_heuristics: ti.types.ndarray(vec2, ndim=1),  # (M)
  ):
    camera_height, camera_width = grad_image_feature.shape
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size

    ti.loop_config(block_dim=(block_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, block_area):
      pixel_base = tiling.tile_transform(tile_id, tile_idx, 
                                       tile_size, config.pixel_stride, tiles_wide)

      # Initialize accumulators for all pixels in tile
      accum_features = thread_features(0.0)
      next_hit = thread_i32(0.)
      num_next_hit = thread_i32(0)

      hit_idx = thread_i32(0)

      # Initialize RNG states and check bounds for all pixels in tile
      for i, offset in ti.static(pixel_tile):
        pixel = pixel_base + ti.Vector(offset)
        in_bounds = pixel.y < camera_height and pixel.x < camera_width
        if in_bounds:
          next_hit[i], num_next_hit[i] = decode_hit(image_hits[pixel.y, pixel.x, 0])


      start_offset, end_offset = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset
      num_point_groups = (tile_point_count + ti.static(block_area - 1)) // block_area

      # open shared memory
      tile_point_id = ti.simt.block.SharedArray((block_area, ), dtype=ti.i32)
      tile_point = ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
      tile_feature = ti.simt.block.SharedArray((block_area, ), dtype=feature_vec)

      tile_grad_point = (ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
                        if ti.static(points_requires_grad) else None)
      
      tile_grad_feature = (ti.simt.block.SharedArray((block_area,), dtype=feature_vec)
                          if ti.static(features_requires_grad) else None)

      tile_point_heuristics = (ti.simt.block.SharedArray((block_area,), dtype=vec2) 
                              if ti.static(config.compute_point_heuristics) else None)
      

      for point_group_id in range(num_point_groups):
        finished = next_hit.sum() == 0
        if not ti.simt.block.sync_any_nonzero(finished):
          break 

        # Load points into shared memory
        group_start_offset = start_offset + point_group_id * block_area
        load_index = group_start_offset + tile_idx

        if load_index < end_offset:
          point_idx = overlap_to_point[load_index]
          tile_point[tile_idx] = points[point_idx]
          tile_feature[tile_idx] = point_features[point_idx]

          if ti.static(points_requires_grad):
            tile_grad_point[tile_idx] = Gaussian2D.vec(0.0)

          if ti.static(features_requires_grad):
            tile_grad_feature[tile_idx] = feature_vec(0.0)


          if ti.static(config.compute_point_heuristics):
            tile_point_heuristics[tile_idx] = vec2(0.0)

        ti.simt.block.sync()

        max_point_group_offset = ti.min(
            block_area, tile_point_count - point_group_id * block_area)

        # Process all points in group for each pixel in tile
        for in_group_idx in range(max_point_group_offset):
          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

          grad_point = Gaussian2D.vec(0.0)
          grad_feature = feature_vec(0.0)
          gaussian_point_heuristics = vec2(0.0)

          has_grad = False

          for i, offset in ti.static(pixel_tile):
            if next_hit[i] == tile_point_id[in_group_idx]:
              has_grad = True
              pixel = pixel_base + ti.Vector(offset) 
              pixelf = ti.cast(pixel, dtype) + 0.5

              gaussian_alpha, dp_dmean, dp_daxis, dp_dsigma = gaussian_pdf(pixelf, mean, axis, sigma)

              weight = num_next_hit[i]/config.samples 
              feature_diff = tile_feature[in_group_idx] - accum_features[i, :]

              alpha_grad_from_feature = feature_diff * grad_image_feature[pixel.y, pixel.x]
              alpha_grad = alpha_grad_from_feature.sum()

              pos_grad = alpha_grad * point_alpha * dp_dmean
              axis_grad = alpha_grad * point_alpha * dp_daxis
              sigma_grad = alpha_grad * point_alpha * dp_dsigma
              alpha_grad = gaussian_alpha * alpha_grad

              grad_point += Gaussian2D.to_vec(
                  pos_grad, axis_grad, sigma_grad, alpha_grad
              )

              if ti.static(features_requires_grad):
                grad_feature += weight * grad_image_feature[pixel.y, pixel.x]

              if ti.static(config.compute_point_heuristics):
                gaussian_point_heuristics += vec2(
                  weight,  # visibility
                  lib.l1_norm(pos_grad)  # sensitivity
                )

              accum_features[i, :] += (
                  tile_feature[in_group_idx] * weight
              )

              # Step to next hit
              hit_idx += 1
              next_hit[i], num_next_hit[i] = decode_hit(image_hits[pixel.y, pixel.x, hit_idx])

              if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(has_grad)):
                # Accumulate gradients in shared memory with warp sums
                if ti.static(points_requires_grad):
                  warp_add_vector(tile_grad_point[in_group_idx], grad_point)
                
                if ti.static(features_requires_grad):
                  warp_add_vector(tile_grad_feature[in_group_idx], grad_feature)

                if ti.static(config.compute_point_heuristics):
                  warp_add_vector(tile_point_heuristics[in_group_idx], gaussian_point_heuristics)

          # After processing points in group, accumulate to global memory
          ti.simt.block.sync()

          if load_index < end_offset:
            point_idx = overlap_to_point[load_index]
            if ti.static(points_requires_grad):
              ti.atomic_add(grad_points[point_idx], tile_grad_point[tile_idx])

            if ti.static(features_requires_grad):
              ti.atomic_add(grad_features[point_idx], tile_grad_feature[tile_idx])

            if ti.static(config.compute_point_heuristics):
              ti.atomic_add(point_heuristics[point_idx], tile_point_heuristics[tile_idx])

        # end of point group loop
      # end of tile loop

  return _backward_kernel







