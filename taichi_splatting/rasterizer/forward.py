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
    hit_vector = ti.types.vector(config.samples, dtype=ti.u32)

    # Create pixel tile mapping
    pixel_tile = tuple(
        (i % config.pixel_stride[0], i // config.pixel_stride[0])
        for i in range(thread_pixels))

    gaussian_pdf = lib.gaussian_pdf_antialias if config.antialias else lib.gaussian_pdf


    @ti.dataclass
    class PixelState:
      pixel: ti.math.ivec2
      rng_state: ti.u32
      remaining_samples: ti.i32
      accum_features: feature_vec
      hit_info: hit_vector
      hit_index: ti.i32

      @ti.func
      def accumulate(self, mean:lib.vec2, axis:lib.vec2, sigma:lib.dtype, point_alpha:lib.dtype, 
                     point_feature: feature_vec, point_idx: ti.i32):
        
        pixelf = ti.cast(self.pixel, dtype) + 0.5

        gaussian_alpha = gaussian_pdf(pixelf, mean, axis, sigma)
        alpha = point_alpha * gaussian_alpha
        alpha = ti.min(alpha, ti.static(config.clamp_max_alpha))

        u, self.rng_state = xoshiro128(self.rng_state)
        new_hits = min(self.remaining_samples, 
                      bernoulli(u, alpha, config.samples))
        
        if new_hits > 0:
            self.accum_features += (
                point_feature * new_hits / config.samples
            )
            encoded = ti.u32(point_idx) << ti.u32(6) | ti.u32(new_hits)
            self.hit_info[self.hit_index] = encoded
            self.hit_index += 1

            self.remaining_samples -= new_hits

        return new_hits
        

    @ti.func
    def initialize_pixel_state(pixel:ti.math.ivec2, camera_height:ti.i32, camera_width:ti.i32, seed:ti.u32):
        in_bounds = pixel.y < camera_height and pixel.x < camera_width
        rng_state = wang_hash(ti.u32(pixel.x), ti.u32(pixel.y), seed)
        remaining_samples = config.samples * ti.i32(in_bounds)
        return PixelState(pixel, rng_state, remaining_samples, feature_vec(0.0), hit_vector(0), 0)


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

            # Initialize pixel states for all pixels 
            pixel_states = [initialize_pixel_state(
                pixel_base + ti.math.ivec2(offset), camera_height, camera_width, seed)
                for offset in ti.static(pixel_tile)]
            
            remaining = thread_pixels * config.samples

            start_offset, end_offset = tile_overlap_ranges[tile_id]
            tile_point_count = end_offset - start_offset
            num_point_groups = (tile_point_count + ti.static(block_area - 1)) // block_area

            # open shared memory
            tile_point = ti.simt.block.SharedArray((block_area, ), dtype=Gaussian2D.vec)
            tile_feature = ti.simt.block.SharedArray((block_area, ), dtype=feature_vec)

            for point_group_id in range(num_point_groups):
            
                    
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
                    if remaining > 0:
                      mean, axis, sigma, point_alpha = Gaussian2D.unpack(tile_point[in_group_idx])

                      for state in ti.static(pixel_states):
                        remaining -= state.accumulate(mean, axis, sigma, point_alpha, 
                          tile_feature[in_group_idx], group_start_offset + in_group_idx + 1)


            # Write final results
            for state in ti.static(pixel_states):
                pixel = state.pixel
                if pixel.y < camera_height and pixel.x < camera_width:
                    image_feature[pixel.y, pixel.x] = state.accum_features
                    image_hits[pixel.y, pixel.x] = state.hit_info
                    
    return _forward_kernel




