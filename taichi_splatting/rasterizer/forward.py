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

    # Match backward.py pattern for pixel tiling
    thread_pixels = config.pixel_stride[0] * config.pixel_stride[1]
    block_area = tile_area // thread_pixels

    # Create batch types for accumulation
    thread_features = ti.types.matrix(thread_pixels, feature_size, dtype=dtype)
    thread_u32 = ti.types.vector(thread_pixels, dtype=ti.u32)
    thread_i32 = ti.types.vector(thread_pixels, dtype=ti.i32)

    thread_hits = ti.types.matrix(thread_pixels, config.samples, dtype=ti.u32)
    hit_vector = ti.types.vector(config.samples, dtype=ti.u32)

    # Create pixel tile mapping
    pixel_tile = tuple([ (i, 
                (i % config.pixel_stride[0],
                i // config.pixel_stride[0]))
                  for i in range(thread_pixels) ])

    gaussian_pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf

    @ti.kernel
    def _forward_kernel(
        points: ti.types.ndarray(Gaussian2D.vec, ndim=1),
        point_features: ti.types.ndarray(feature_vec, ndim=1),
        tile_overlap_ranges: ti.types.ndarray(ti.math.ivec2, ndim=1),
        overlap_to_point: ti.types.ndarray(ti.i32, ndim=1),
        image_feature: ti.types.ndarray(feature_vec, ndim=2),
        image_hits: ti.types.ndarray(hit_vector, ndim=2),

        seed: ti.uint32
    ):
        camera_height, camera_width = image_feature.shape
        tiles_wide = (camera_width + tile_size - 1) // tile_size 
        tiles_high = (camera_height + tile_size - 1) // tile_size

        ti.loop_config(block_dim=(block_area))
        for tile_id, tile_idx in ti.ndrange(tiles_wide * tiles_high, block_area):
            pixel_base = tiling.tile_transform(tile_id, tile_idx, 
                            tile_size, config.pixel_stride, tiles_wide)

            # Initialize accumulators for all pixels in tile
            accum_features = thread_features(0.0)
            remaining_samples = thread_i32(config.samples)
            rng_states = thread_u32(0)
            hit_index = thread_i32(0)
            hits = thread_hits(0)

            # Initialize RNG states and check bounds for all pixels in tile
            for i, offset in ti.static(pixel_tile):
                pixel = pixel_base + ti.Vector(offset)
                in_bounds = pixel.y < camera_height and pixel.x < camera_width
                rng_states[i] = wang_hash(ti.u32(pixel.x), ti.u32(pixel.y), seed)
                remaining_samples[i] = config.samples * ti.i32(in_bounds)

            start_offset, end_offset = tile_overlap_ranges[tile_id]
            tile_point_count = end_offset - start_offset
            num_point_groups = (tile_point_count + ti.static(block_area - 1)) // block_area

            # open shared memory
            tile_point = ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
            tile_feature = ti.simt.block.SharedArray((block_area, ), dtype=feature_vec)

            for point_group_id in range(num_point_groups):
                remaining = remaining_samples.sum()
                if not ti.simt.block.sync_any_nonzero(remaining):
                    break

                # Load points into shared memory
                group_start_offset = start_offset + point_group_id * block_area
                load_index = group_start_offset + tile_idx

                if load_index < end_offset:
                    point_idx = overlap_to_point[load_index]
                    tile_point[tile_idx] = points[point_idx]
                    tile_feature[tile_idx] = point_features[point_idx]

                ti.simt.block.sync()

                max_point_group_offset = ti.min(
                    block_area, tile_point_count - point_group_id * block_area)

                # Process all points in group for each pixel in tile
                for in_group_idx in range(max_point_group_offset):
                    mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

                    for i, offset in ti.static(pixel_tile):
                        if remaining_samples[i] > 0:
                            pixel = pixel_base + ti.Vector(offset)
                            pixelf = ti.cast(pixel, dtype) + 0.5

                            gaussian_alpha = gaussian_pdf(pixelf, mean, axis, sigma)
                            alpha = point_alpha * gaussian_alpha
                            alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

                            u, rng_states[i] = xoshiro128(rng_states[i])
                            new_hits = min(remaining_samples[i], 
                                         bernoulli(u, alpha, config.samples))
                            
                            if new_hits > 0:
                                accum_features[i, :] += (
                                    tile_feature[in_group_idx] * new_hits / config.samples
                                )
                                index =  group_start_offset + in_group_idx + 1
                                encoded = ti.u32(index) << ti.u32(6) | ti.u32(new_hits)
                                hits[i, hit_index[i]] = encoded
                                hit_index[i] += 1

                                remaining_samples[i] -= new_hits


            # Write final results
            for i, offset in ti.static(pixel_tile):
                pixel = pixel_base + ti.Vector(offset)
                if pixel.y < camera_height and pixel.x < camera_width:
                    image_feature[pixel.y, pixel.x] = accum_features[i, :]
                    image_hits[pixel.y, pixel.x] = hits[i, :]
    return _forward_kernel




