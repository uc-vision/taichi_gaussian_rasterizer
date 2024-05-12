
from dataclasses import dataclass
from typing import Callable
import taichi as ti
from taichi.math import vec2


@dataclass(frozen=True)
class GridQuery:
  """
    Pair of primitive type and a query object, both taichi structs.
    The query object is used to precompute some data for repeated queries on grid cells,
    Supports two queries (taichi functions)
      test_tile(self:QueryObject, tile_uv: ivec2)
      count_tiles(self:QueryObject) -> ti.i32
  """

  make_query: Callable # make_query(v: primitive, image_size:ivec2) -> QueryObject
  primitive: ti.lang.struct.StructType # taichi primitive - e.g. GaussianConic



@ti.func
def tile_ranges(
    min_bound: vec2,
    max_bound: vec2,

    image_size: ti.math.ivec2,
    tile_size: ti.template()
):

    max_tile = (image_size - 1) // tile_size + 1

    min_tile_bound = ti.math.clamp(
       ti.floor(min_bound / tile_size, ti.i32),
       0, max_tile)
    
    max_tile_bound = ti.math.clamp(
       ti.ceil(max_bound / tile_size, ti.i32),
       0, max_tile)

     
    return min_tile_bound, max_tile_bound


