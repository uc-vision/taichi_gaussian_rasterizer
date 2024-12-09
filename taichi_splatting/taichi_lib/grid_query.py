
import taichi as ti
from taichi.math import ivec2, vec2, mat2, vec3

from taichi_splatting.taichi_lib.f32 import (Gaussian2D, 
    ellipse_bounds)


@ti.func
def tile_ranges(
    min_bound: vec2,
    max_bound: vec2,

    image_size: ti.math.ivec2,
    tile_size: ti.template()
):
    max_tile = (image_size - 1) // tile_size 


    min_tile_bound = ti.floor(min_bound / tile_size, ti.i32)
    min_tile_bound = ti.max(min_tile_bound, 0)

    max_tile_bound = ti.ceil(max_bound / tile_size, ti.i32) 
    max_tile_bound = ti.min(ti.max(max_tile_bound, min_tile_bound + 1),
                        max_tile + 1)
     
    return min_tile_bound, max_tile_bound

@ti.func
def separates_bbox(inv_basis:mat2, lower:vec2, upper:vec2) -> bool:
  rel_points = ti.Matrix.cols(
      [lower, vec2(upper.x, lower.y), upper, vec2(lower.x, upper.y)])
  
  local_points = (inv_basis @ rel_points)

  separates = False
  for i in ti.static(range(2)):
    min_val = ti.min(*local_points[i, :])
    max_val = ti.max(*local_points[i, :])
    if (min_val > 1. or max_val < -1.):
      separates = True

  return separates


def make_grid_query(tile_size:int=16, alpha_threshold:float=1. / 255.):

  @ti.dataclass
  class OBBGridQuery:
    inv_basis: mat2

    rel_min_bound: vec2

    min_tile: ivec2
    tile_span: ivec2

    @ti.func
    def test_tile(self, tile_uv: ivec2):
      lower = self.rel_min_bound + tile_uv * tile_size
      return not separates_bbox(self.inv_basis, lower, lower + tile_size)
      
    @ti.func 
    def count_tiles(self) -> ti.i32:
      count = 0
      
      for tile_uv in ti.grouped(ti.ndrange(*self.tile_span)):
        if self.test_tile(tile_uv):
          count += 1

      return count

  @ti.func 
  def obb_grid_query(v: Gaussian2D.vec, image_size:ivec2) -> OBBGridQuery:
      mean, axis1, sigma, alpha = Gaussian2D.unpack(v)

      gaussian_scale = ti.sqrt(2 * ti.log(alpha / alpha_threshold))
      scale = sigma * gaussian_scale

      axis2 = vec2(-axis1.y, axis1.x)
      min_bound, max_bound = ellipse_bounds(mean, axis1 * scale.x, axis2 * scale.y)
      
      # transform from (relative) image to ellipse space
      inv_basis = ti.Matrix.rows([axis1 / scale.x, axis2 / scale.y])

      min_tile, max_tile = tile_ranges(min_bound, max_bound, image_size, tile_size)
      return OBBGridQuery(
        inv_basis = inv_basis,
        rel_min_bound = min_tile * tile_size - mean,

        min_tile = min_tile,
        tile_span = max_tile - min_tile)

  return obb_grid_query
