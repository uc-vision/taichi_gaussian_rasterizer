
from functools import cache
import taichi as ti
from taichi.math import ivec2, vec2, mat2, vec3


from taichi_splatting.mapper.grid_query import GridQuery, tile_ranges
from taichi_splatting.taichi_lib.f32 import (GaussianConic, 
    inverse_cov, cov_inv_basis, radii_from_cov,)


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

    return tile_ranges(
      min_bound = ti.max(0.0, uv - radius),
      max_bound = uv + radius,
      image_size = image_size,
      tile_size = tile_size
    )

@ti.func
def gaussian_tile_ranges(
    gaussian: GaussianConic.vec,
    image_size: ti.math.ivec2,

    gaussian_scale: ti.template(),
    tile_size: ti.template()
):
    uv, uv_conic, _ = GaussianConic.unpack(gaussian)
    uv_cov = inverse_cov(uv_conic)

    return cov_tile_ranges(uv, uv_cov, image_size, gaussian_scale, tile_size)


@ti.func 
def count_obb_tiles(uv:vec2, uv_conic:vec3, image_size:ivec2, tile_size:int=16, gaussian_scale:float=3.0) -> ti.i32:
  uv_cov = inverse_cov(uv_conic)

  min_tile, max_tile = cov_tile_ranges(uv, uv_cov, image_size, gaussian_scale, tile_size)
  inv_basis = cov_inv_basis(uv_cov, gaussian_scale)

  tile_span = max_tile - min_tile
  rel_min_bound = min_tile * tile_size - uv

  count = 0
  
  for tile_uv in ti.grouped(ti.ndrange(*tile_span)):
      lower = rel_min_bound + tile_uv * tile_size
      count += not separates_bbox(inv_basis, lower, lower + tile_size)

  return count



@cache
def cov_obb_query(tile_size:int=16, gaussian_scale:float=3.0):

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
        count += self.test_tile(tile_uv)
          
      return count

  @ti.func 
  def from_cov(uv:ti.math.vec2, uv_cov:ti.math.vec3, image_size:ivec2) -> OBBGridQuery:
      min_tile, max_tile = cov_tile_ranges(uv, uv_cov, image_size, gaussian_scale, tile_size)

      return OBBGridQuery(
        # Find tiles which intersect the oriented box
        inv_basis = cov_inv_basis(uv_cov, gaussian_scale),
        rel_min_bound = min_tile * tile_size - uv,

        min_tile = min_tile,
        tile_span = max_tile - min_tile)


  @ti.func 
  def from_conic(v: GaussianConic.vec, image_size:ivec2) -> OBBGridQuery:
      uv, uv_conic, _ = GaussianConic.unpack(v)
      uv_cov = inverse_cov(uv_conic)

      return from_cov(uv, uv_cov, image_size)

  return from_cov, from_conic
  
def obb_grid_query(tile_size:int=16, gaussian_scale:float=3.0):
  from_cov, from_conic = cov_obb_query(tile_size, gaussian_scale)
  return GridQuery(
    make_query = from_conic,
    primitive = GaussianConic
  )

@cache
def box_grid_query(tile_size:int=16, gaussian_scale:float=3.0):
  
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
  def query_object(v: GaussianConic.vec, image_size:ivec2) -> RangeGridQuery:
      min_tile, max_tile = gaussian_tile_ranges(v, image_size, gaussian_scale, tile_size)
      return RangeGridQuery(
        min_tile = min_tile,
        tile_span = max_tile - min_tile)
  
  return GridQuery(
    make_query = query_object,
    primitive = GaussianConic
  )

