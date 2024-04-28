
from dataclasses import dataclass
from typing import Callable
import numpy as np
import taichi as ti
from taichi.math import ivec2, vec2, mat2, vec3


from taichi_splatting.data_types import RasterConfig
from taichi_splatting.taichi_lib.f32 import (GaussianConic, 
    inverse_cov, cov_inv_basis, radii_from_cov,)


@dataclass(frozen=True)
class GridQuery:
  """
    Pair of primitive type and a query object, both taichi structs.
    The query object is used to precompute some data for repeated queries on grid cells,
    Supports two queries (taichi functions)
      test_tile(self:QueryObject, tile_uv: ivec2)
      count_tiles(self:QueryObject) -> ti.i32
  """

  make_query: Callable # v: primitive, image_size:ivec2 -> QueryObject
  primitive: ti.lang.struct.StructType # taichi primitive - e.g. GaussianConic


@ti.func
def separates_bbox(inv_basis: mat2, lower:vec2, upper:vec2) -> bool:
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


@ti.func
def cov_tile_ranges(
    uv: vec2,
    uv_cov: vec3,
    image_size: ti.math.ivec2,

    gaussian_scale: ti.template(),
    tile_size: ti.template()
):

    # avoid zero radii, at least 1 pixel
    radius = ti.max(radii_from_cov(uv_cov) * gaussian_scale, 1.0)  

    min_bound = ti.max(0.0, uv - radius)
    max_bound = uv + radius

    max_tile = image_size // tile_size

    min_tile_bound = ti.cast(min_bound / tile_size, ti.i32)
    min_tile_bound = ti.min(min_tile_bound, max_tile)

    max_tile_bound = ti.cast(max_bound / tile_size, ti.i32) + 1
    max_tile_bound = ti.min(ti.max(max_tile_bound, min_tile_bound + 1),
                        max_tile)

    return min_tile_bound, max_tile_bound

@ti.func
def gaussian_tile_bounds(
    gaussian: GaussianConic.vec,
    image_size: ti.math.ivec2,

    gaussian_scale: ti.template(),
    tile_size: ti.template()
):
    uv, uv_conic, _ = GaussianConic.unpack(gaussian)
    uv_cov = inverse_cov(uv_conic)

    return cov_tile_ranges(uv, uv_cov, image_size, gaussian_scale, tile_size)



def conic_grid_query(config:RasterConfig):

  x = np.linspace(0, 4, 1000)
  y = np.exp(-(0.5 * x**2) ** config.beta)

  gaussian_scale =   x[np.argmax(y < 0.5 * config.alpha_threshold)]

  tile_size = config.tile_size


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
  def obb_grid_query(v: GaussianConic.vec, image_size:ivec2) -> OBBGridQuery:
      uv, uv_conic, _ = GaussianConic.unpack(v)
      uv_cov = inverse_cov(uv_conic)

      min_tile, max_tile = cov_tile_ranges(uv, uv_cov, image_size, gaussian_scale, tile_size)
      return OBBGridQuery(
        # Find tiles which intersect the oriented box
        inv_basis = cov_inv_basis(uv_cov, gaussian_scale),
        rel_min_bound = min_tile * tile_size - uv,

        min_tile = min_tile,
        tile_span = max_tile - min_tile)


  @ti.dataclass
  class RangeGridQuery:

    min_tile: ivec2
    tile_span: ivec2

    @ti.func
    def test_tile(self, tile_uv: ivec2):
      return True
    
    @ti.func 
    def count_tiles(self) -> ti.i32:
      return self.tile_span.x * self.tile_span.y
          
  @ti.func 
  def range_grid_query(v: GaussianConic.vec, image_size:ivec2) -> RangeGridQuery:
      min_tile, max_tile = gaussian_tile_bounds(v, image_size, gaussian_scale, tile_size)
      return RangeGridQuery(
        min_tile = min_tile,
        tile_span = max_tile - min_tile)



  return GridQuery(
    make_query = obb_grid_query if config.tight_culling else range_grid_query,
    primitive = GaussianConic
  )
