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
    bernoulli = lib.bernoulli
    xoshiro128 = lib.xoshiro128
    wang_hash = lib.wang_hash

    feature_vec = ti.types.vector(feature_size, dtype=dtype)
    tile_size = config.tile_size
    tile_area = tile_size * tile_size

    hit_vector = ti.types.vector(config.samples, dtype=ti.u32)
    gaussian_pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf



    @ti.kernel
    def _forward_kernel(
        points: ti.types.ndarray(Gaussian2D.vec, ndim=1),
        point_features: ti.types.ndarray(feature_vec, ndim=1),
        tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
        overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
        image_feature: ti.types.ndarray(feature_vec, ndim=2),
        image_alpha: ti.types.ndarray(ti.f32, ndim=2),

        image_hits: ti.types.ndarray(hit_vector, ndim=2),
        seed: ti.uint32
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
            remaining_samples = ti.i32(config.samples) * ti.i32(in_bounds)
            hit_index = ti.i32(0)
            hits = hit_vector(0)

            rng_states = wang_hash(ti.u32(pixel.x), ti.u32(pixel.y), seed)

            start_offset, end_offset = tile_overlap_ranges[tile_id]
            tile_point_count = end_offset - start_offset
            num_point_groups = tiling.round_up(tile_point_count, tile_area)

            # open shared memory
            tile_point = ti.simt.block.SharedArray((tile_area, ), dtype=Gaussian2D.vec)
            tile_feature = ti.simt.block.SharedArray((tile_area, ), dtype=feature_vec)
            tile_point_id = ti.simt.block.SharedArray((tile_area, ), dtype=ti.i32)

            for point_group_id in range(num_point_groups):
                if not ti.simt.block.sync_any_nonzero(remaining_samples):
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
                    if ti.simt.warp.all_nonzero(ti.u32(0xffffffff), ti.i32(remaining_samples == 0)):
                        break
                        
                    mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

                    gaussian_alpha = gaussian_pdf(pixelf, mean, axis, sigma)
                    alpha = point_alpha * gaussian_alpha
                    alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

                    u, rng_states = xoshiro128(rng_states)
                    new_hits = min(remaining_samples, 
                                  bernoulli(u, alpha, config.samples))
                    
                    if new_hits > 0:
                        accum_features += (
                            tile_feature[in_group_idx] * new_hits / config.samples
                        )


                        encoded = tiling.encode_hit(tile_point_id[in_group_idx], new_hits)
                        hits[hit_index] = encoded
                        hit_index += 1

                        remaining_samples -= new_hits


            # Write final results
            if pixel.y < camera_height and pixel.x < camera_width:
                image_feature[pixel.y, pixel.x] = accum_features

                alpha = (ti.static(config.samples) - remaining_samples) / ti.static(config.samples)
                image_alpha[pixel.y, pixel.x] = alpha

                image_hits[pixel.y, pixel.x] = hits
    return _forward_kernel




