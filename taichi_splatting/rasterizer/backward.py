from functools import cache
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.rasterizer import tiling
from taichi_splatting.taichi_lib.concurrent import WARP_SIZE, block_reduce_i32, warp_add_vector_32, warp_add_vector_64

from taichi_splatting.taichi_lib import get_library
from taichi.math import ivec2

from taichi_splatting.taichi_queue import queued




@cache
def backward_kernel(config: RasterConfig,
                    points_requires_grad: bool,
                    features_requires_grad: bool, 
                    feature_size: int,
                    dtype=ti.f32):
  
  lib = get_library(dtype)
  Gaussian2D, vec2 = lib.Gaussian2D, lib.vec2
  warp_add_vector = warp_add_vector_32 if dtype == ti.f32 else warp_add_vector_64

  feature_vec = ti.types.vector(feature_size, dtype=dtype)
  tile_size = config.tile_size
  tile_area = tile_size * tile_size

  thread_pixels = config.pixel_stride[0] * config.pixel_stride[1]
  block_area = tile_area // thread_pixels
  
  assert block_area >= WARP_SIZE, \
    f"pixel_stride {config.pixel_stride} and tile_size, {config.tile_size} must allow at least one warp sized ({WARP_SIZE}) tile"

  # each thread is responsible for a small tile of pixels
  pixel_tile = tuple([ (i, 
            (i % config.pixel_stride[0],
            i // config.pixel_stride[0]))
              for i in range(thread_pixels) ])

  # types for each thread to keep state in it's tile of pixels
  thread_features = ti.types.matrix(thread_pixels, feature_size, dtype=dtype)
  thread_vector = ti.types.vector(thread_pixels, dtype=dtype)
  thread_index = ti.types.vector(thread_pixels, dtype=ti.i32)


  gaussian_pdf = lib.gaussian_pdf_antialias_with_grad if config.antialias else lib.gaussian_pdf_with_grad

  @ti.kernel
  def _backward_kernel(
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, 6)
      point_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)
      
      # (TH, TW, 2) the start/end (0..K] index of ranges in the overlap_to_point array
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      # (K) ranges of points mapping to indexes into points list
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
      
      # saved from forward
      image_alpha: ti.types.ndarray(dtype, ndim=2),       # H, W
      image_last_valid: ti.types.ndarray(ti.i32, ndim=2),  # H, W

      # input gradients
      grad_image_feature: ti.types.ndarray(feature_vec, ndim=2),  # (H, W, F)

      # output gradients
      grad_points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, C)
      grad_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)

      point_heuristics: ti.types.ndarray(vec2, ndim=1),  # (M)
  ):

    camera_height, camera_width = image_alpha.shape

    # round up
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size

    # see forward.py for explanation of tile_id and tile_idx and blocking
    ti.loop_config(block_dim=(block_area))
    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, block_area):
      pixel_base = tiling.tile_transform(tile_id, tile_idx, 
                        tile_size, config.pixel_stride, tiles_wide)


      # open the shared memory
      tile_point_id = ti.simt.block.SharedArray((block_area, ), dtype=ti.i32)
      tile_point = ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
      tile_feature = ti.simt.block.SharedArray((block_area, ), dtype=feature_vec)

      tile_grad_point = (ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
        if ti.static(points_requires_grad) else None)
      
      tile_grad_feature = (ti.simt.block.SharedArray((block_area,), dtype=feature_vec)
        if ti.static(features_requires_grad) else None)

      tile_point_heuristics = (ti.simt.block.SharedArray((block_area,), dtype=vec2) 
        if ti.static(config.compute_point_heuristics) else None)

      
      last_point_pixel = thread_index(-1)
      T_i = thread_vector(1.0)
      grad_pixel_feature = thread_features(0.0)
      #pixel_feature = thread_features(0.0)

  
      for i, offset in ti.static(pixel_tile):
        pixel = ivec2(offset) + pixel_base

        if pixel.y < camera_height and pixel.x < camera_width:
          last_point_pixel[i] = image_last_valid[pixel.y, pixel.x]
          T_i[i] = 1.0 - image_alpha[pixel.y, pixel.x]
          grad_pixel_feature[i, :] = grad_image_feature[pixel.y, pixel.x]
          #pixel_feature[i, :] = image_feature[pixel.y, pixel.x]

      last_point_thread = last_point_pixel.max()
      w_i = thread_features(0.0)

      #  T_i = \prod_{j=1}^{i-1} (1 - a_j) \\
      #  \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} \\
      #  \sum_{j=i+1}^{n} c_j a_j T(j) \\
      #  \text{let } w_i = \sum_{j=i+1}^{n} c_j a_j T(j) \\
      #  w_n = 0 \\
      #  w_{i-1} = w_i + c_i a_i T(i) \\
      #  \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i \\

      # fine tune the end offset to the actual number of points renderered
      end_offset = block_reduce_i32(last_point_thread, ti.max, ti.atomic_max, 0)

      start_offset, _ = tile_overlap_ranges[tile_id]
      tile_point_count = end_offset - start_offset

      num_point_groups = (tile_point_count + ti.static(block_area - 1)) // block_area

      # Loop through the range in groups of block_area
      for point_group_id in range(num_point_groups):
        ti.simt.block.sync() 
        
        # load points and features into block shared memory
        group_offset_base = point_group_id * block_area

        block_end_idx = end_offset - group_offset_base
        block_start_idx = ti.max(block_end_idx - block_area, 0)

        load_index = block_end_idx - tile_idx - 1
        if load_index >= block_start_idx:
          point_idx = overlap_to_point[load_index]

          tile_point_id[tile_idx] = point_idx
          tile_point[tile_idx] = points[point_idx]
          tile_feature[tile_idx] = point_features[point_idx]

          if ti.static(points_requires_grad):
            tile_grad_point[tile_idx] = Gaussian2D.vec(0.0)

          if ti.static(features_requires_grad):
            tile_grad_feature[tile_idx] = feature_vec(0.0)
          
          if ti.static(config.compute_point_heuristics):
            tile_point_heuristics[tile_idx] = vec2(0.0)

        point_group_size = ti.min(
          block_area, tile_point_count - group_offset_base)
                    

        ti.simt.block.sync()

        for in_group_idx in range(point_group_size):
          point_index = end_offset - (group_offset_base + in_group_idx)

          mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

          grad_point = Gaussian2D.vec(0.0)
          grad_feature = feature_vec(0.0)
          gaussian_point_heuristics = vec2(0.0)

          has_grad = False
          for i, offset in ti.static(pixel_tile):
            pixelf = ti.cast(pixel_base, dtype) + vec2(offset) + 0.5

            gaussian_alpha, dp_dmean, dp_daxis, dp_dsigma = gaussian_pdf(pixelf, mean, axis, sigma)
            
            alpha = point_alpha * gaussian_alpha
            pixel_grad = (alpha >= ti.static(config.alpha_threshold)) and (point_index <= last_point_pixel[i])      
      
            if pixel_grad:
              has_grad = True
              
              alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))
              T_i[i] /= (1. - alpha)

              feature = tile_feature[in_group_idx]
              weight = alpha * T_i[i]

              grad_feature += weight * grad_pixel_feature[i, :]      
              feature_diff = (feature * T_i[i] - w_i[i, :] / (1. - alpha))

              
                # \frac{dC}{da_i} = c_i T(i) - \frac{1}{1 - a_i} w_i
              alpha_grad_from_feature = feature_diff * grad_pixel_feature[i, :]

              # w_{i-1} = w_i + c_i a_i T(i)
              w_i[i, :] += feature * weight
              alpha_grad: dtype = alpha_grad_from_feature.sum()

              pos_grad = alpha_grad * point_alpha * dp_dmean
              axis_grad = alpha_grad * point_alpha * dp_daxis
              sigma_grad = alpha_grad * point_alpha * dp_dsigma
              grad_point +=  Gaussian2D.to_vec(pos_grad, axis_grad, sigma_grad, gaussian_alpha * alpha_grad)

              if ti.static(config.compute_point_heuristics):
                gaussian_point_heuristics += vec2(
                  # (feature_diff**2).sum() * weight, # pruning cost heuristic
                  # (dp_dmean.norm() * feature_diff.norm() * weight),
                  dp_dmean.norm(2) * weight,
                  ti.abs(pos_grad).sum() # point split heuristic 
                )

          if ti.simt.warp.any_nonzero(ti.u32(0xffffffff), ti.i32(has_grad)):

            # Accumulating gradients in block shared memory does not appear to be faster
            # on it's own, but combined with warp sums it seems to be fast
            if ti.static(points_requires_grad):
              warp_add_vector(tile_grad_point[in_group_idx], grad_point)
            
            if ti.static(features_requires_grad):
              warp_add_vector(tile_grad_feature[in_group_idx], grad_feature)

            if ti.static(config.compute_point_heuristics):
              warp_add_vector(tile_point_heuristics[in_group_idx], gaussian_point_heuristics)
        # end of point group loop

        ti.simt.block.sync()

        # finally accumulate gradients in global memory
        if load_index >= block_start_idx:
          point_offset = tile_point_id[tile_idx] 
          if ti.static(points_requires_grad):
            ti.atomic_add(grad_points[point_offset], tile_grad_point[tile_idx])

          if ti.static(features_requires_grad):
            ti.atomic_add(grad_features[point_offset], tile_grad_feature[tile_idx])

          if ti.static(config.compute_point_heuristics):
            ti.atomic_add(point_heuristics[point_offset], tile_point_heuristics[tile_idx])

      # end of point group id loop
    # end of pixel loop

  @ti.kernel
  def _backward_kernel_no_alpha(
      points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, 6)
      point_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)
      
      # (TH, TW, 2) the start/end (0..K] index of ranges in the overlap_to_point array
      tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
      # (K) ranges of points mapping to indexes into points list
      overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
      
      # saved from forward
      image_alpha: ti.types.ndarray(dtype, ndim=2),       # H, W
      image_last_valid: ti.types.ndarray(ti.i32, ndim=2),  # H, W

      # input gradients
      grad_image_feature: ti.types.ndarray(feature_vec, ndim=2),  # (H, W, F)

      # output gradients
      grad_points: ti.types.ndarray(Gaussian2D.vec, ndim=1),  # (M, C)
      grad_features: ti.types.ndarray(feature_vec, ndim=1),  # (M, F)

      point_heuristics: ti.types.ndarray(vec2, ndim=1),  # (M)
  ):

    camera_height, camera_width = image_alpha.shape

    # round up
    tiles_wide = (camera_width + tile_size - 1) // tile_size 
    tiles_high = (camera_height + tile_size - 1) // tile_size

    # see forward.py for explanation of tile_id and tile_idx and blocking
    ti.loop_config(block_dim=(block_area))

    for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):

      pixel = tiling.tile_transform(tile_id, tile_idx, tile_size, (1, 1), tiles_wide)
      # pixelf = ti.cast(pixel, dtype) + 0.5

      if pixel.y < camera_height and pixel.x < camera_width and image_last_valid[pixel.y, pixel.x] >= 0:
        gaussian_idx = image_last_valid[pixel.y, pixel.x] - 1
        grad_feature = grad_image_feature[pixel.y, pixel.x]

        # mean, axis, sigma, _ = Gaussian2D.unpack(points[gaussian_idx])
        # feature = point_features[gaussian_idx]
      
        # alpha_grad_from_feature = feature * grad_feature
        # alpha_grad: dtype = alpha_grad_from_feature.sum()

        grad_point = Gaussian2D.vec(0.)

        if ti.static(points_requires_grad):
          ti.atomic_add(grad_points[gaussian_idx], grad_point)

        if ti.static(features_requires_grad):
          ti.atomic_add(grad_features[gaussian_idx], grad_feature)


      # end of point group id loop
    # end of pixel loop


  return _backward_kernel if config.use_alpha_blending else _backward_kernel_no_alpha




