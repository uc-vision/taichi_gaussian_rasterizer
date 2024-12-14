from dataclasses import replace
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

    pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf

    @ti.kernel
    def _forward_kernel(
        points: ti.types.ndarray(Gaussian2D.vec, ndim=1),
        point_features: ti.types.ndarray(feature_vec, ndim=1),
        tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
        overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
        image_feature: ti.types.ndarray(feature_vec, ndim=2),
        image_alpha: ti.types.ndarray(ti.f32, ndim=2),
    ):
        camera_height, camera_width = image_feature.shape
        tiles_wide = (camera_width + tile_size - 1) // tile_size 
        tiles_high = (camera_height + tile_size - 1) // tile_size

        ti.loop_config(block_dim=(tile_area))
        for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, tile_area):
            pixel = tiling.tile_transform(tile_id, tile_idx, 
                            tile_size, (1, 1), tiles_wide)
            pixelf = ti.cast(pixel, dtype) + 0.5

            # Initialize accumulators for all pixels in tile
            in_bounds = pixel.y < camera_height and pixel.x < camera_width

            accum_features = feature_vec(0.0)
            total_weight = 0.0 if in_bounds else 1.0

            start_offset, end_offset = tile_overlap_ranges[tile_id]
            tile_point_count = end_offset - start_offset
            num_point_groups = tiling.round_up(tile_point_count, tile_area)

            # open shared memory
            tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
            tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)
            tile_point_id = ti.simt.block.SharedArray((tile_area, ), dtype=ti.i32)

            for point_group_id in range(num_point_groups):
                if ti.simt.block.sync_all_nonzero(ti.i32(total_weight >= ti.static(config.saturate_threshold))):
                    break

                # Load points into shared memory
                group_start_offset = start_offset + point_group_id * tile_area
                load_index = group_start_offset + tile_idx

                if load_index < end_offset:
                    point_idx = overlap_to_point[load_index]
                    tile_point[tile_idx] = points[point_idx]
                    tile_feature[tile_idx] = point_features[point_idx]
                    tile_point_id[tile_idx] = point_idx
                ti.simt.block.sync()

                remaining_points = tile_point_count - point_group_id

                # Process all points in group for each pixel in tile
                for in_group_idx in range(min(tile_area, remaining_points)):
                    if total_weight >= ti.static(config.saturate_threshold):
                        break
                        
                    mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

                    gaussian_alpha = pdf(pixelf, mean, axis, sigma)
                    alpha = point_alpha * gaussian_alpha
                    alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

                    if alpha > config.alpha_threshold:
                        weight = alpha * (1.0 - total_weight)

                        accum_features += tile_feature[in_group_idx] * weight
                        total_weight += weight
                        
            # Write final results
            if in_bounds:
                image_feature[pixel.y, pixel.x] = accum_features 
                image_alpha[pixel.y, pixel.x] = total_weight

    return _forward_kernel





