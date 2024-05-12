
from functools import cache
import taichi as ti
from taichi.math import ivec2


from taichi_splatting.mapper.grid_query import GridQuery, tile_ranges
from taichi_splatting.taichi_lib.f32 import (Quad, mat4x3)




@ti.func
def quad_tile_ranges(
    quad: Quad,
    image_size: ti.math.ivec2,
    tile_size: ti.template()
):
    # avoid zero radii, at least 1 pixel
    min_bound, max_bound = quad.bounds()

    return tile_ranges(
      min_bound = min_bound,
      max_bound = max_bound,
      image_size = image_size,
      tile_size = tile_size
    )

@cache
def quad_grid_query(tile_size:int=16):

  @ti.dataclass
  class QuadGridQuery:
    planes: mat4x3

    min_tile: ivec2
    tile_span: ivec2

    @ti.func
    def test_tile(self, tile_uv: ivec2):
      return True
      
    @ti.func 
    def count_tiles(self) -> ti.i32:
      count = 0
      
      for tile_uv in ti.grouped(ti.ndrange(*self.tile_span)):
        if self.test_tile(tile_uv):
          count += 1

      return count

  @ti.func 
  def obb_grid_query(v: Quad.vec, image_size:ivec2) -> QuadGridQuery:
      quad = Quad.from_vec(v)

      min_tile, max_tile = quad_tile_ranges(quad, image_size, tile_size)

      return QuadGridQuery(
        # Find tiles which intersect the oriented box
        planes = quad.planes(),

        min_tile = min_tile,
        tile_span = max_tile - min_tile)


  return GridQuery(
    make_query = obb_grid_query,
    primitive = Quad
  )

