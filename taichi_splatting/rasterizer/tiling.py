import taichi as ti
from taichi.math import ivec2

from taichi_splatting.taichi_lib.concurrent import WARP_SIZE

@ti.func
def interleave(x:ti.i32):
    x = ( x | ( x << 4 ) ) & 0x0F0F0F0F
    x = ( x | ( x << 2 ) ) & 0x33333333
    x = ( x | ( x << 1 ) ) & 0x55555555
    return x


@ti.func
def deinterleave(x:ti.i32):
  x &= 0x55555555                   
  x = (x ^ (x >>  1)) & 0x33333333 
  x = (x ^ (x >>  2)) & 0x0f0f0f0f
  return x


@ti.func
def morton_tile(x:ti.i32, y:ti.i32):
    order = interleave(x) | ( interleave(y) << 1 )
    return order

@ti.func
def morton_tile_inv(order:ti.i32):
  x = deinterleave(order)
  y = deinterleave(order >> 1)
  return x, y


@ti.func
def warp_transform(i:ti.i32, tile_size:ti.template(), pixel_stride:ti.template()):
  # divide tile into 32 (8x4) sized chunks for a warp
  w = ti.static(8 * pixel_stride[0])
  
  warp_id = i // WARP_SIZE
  warp_offset = i % WARP_SIZE

  warps_wide = ti.static(tile_size // w)

  warp_u = (warp_id % warps_wide)
  warp_v = (warp_id // warps_wide)

  return ivec2(
    (warp_u * 8 + warp_offset % 8) * pixel_stride[0],
    (warp_v * 4 + warp_offset // 8) * pixel_stride[1])



@ti.func
def tile_transform(tile_id:ti.i32, tile_idx:ti.i32, 
                   tile_size:ti.template(),
                   pixel_stride:ti.template(), 
                   tiles_wide:ti.template()):
    
    u, v = warp_transform(tile_idx, tile_size, pixel_stride)

    tile_u = tile_id % tiles_wide
    tile_v = tile_id // tiles_wide

    pixel = ivec2(tile_u, tile_v) * tile_size + ivec2(u, v) 
    return pixel


@ti.func
def round_up(a: ti.i32, b: ti.i32) -> ti.i32:
    return (a + (b - 1)) // b

@ti.func
def encode_hit(point_id: ti.i32, hits: ti.i32) -> ti.u32:
    return ti.u32(point_id) << ti.u32(6) | ti.u32(hits)


@ti.func
def decode_hit(hit: ti.u32):
    id = -1
    count = 0
  
    if hit != 0:
      id = ti.cast(hit >> 6, ti.i32)
      count = ti.cast(hit & 0x3F, ti.i32)

    return id, count
