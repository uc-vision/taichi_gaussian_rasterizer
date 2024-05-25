
from types import SimpleNamespace
import taichi as ti
from taichi.math import ivec2, vec2, mat2, vec3

from taichi_splatting.taichi_lib.f32 import (Gaussian2D, 
    inverse_cov, cov_inv_basis, radii_from_cov,)

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


def make_grid_query(tile_size:int=16, gaussian_scale:float=3.0, tight_culling:bool=True):

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
      uv, uv_conic, _ = Gaussian2D.unpack(v)
      uv_cov = inverse_cov(uv_conic)

      min_tile, max_tile = cov_tile_ranges(uv, uv_cov, image_size)
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
  def range_grid_query(v: Gaussian2D.vec, image_size:ivec2) -> RangeGridQuery:
      min_tile, max_tile = gaussian_tile_bounds(v, image_size)
      return RangeGridQuery(
        min_tile = min_tile,
        tile_span = max_tile - min_tile)



  @ti.func
  def cov_tile_ranges(
      uv: vec2,
      uv_cov: vec3,
      image_size: ti.math.ivec2,
  ):

      # avoid zero radii, at least 1 pixel
      radius = ti.max(radii_from_cov(uv_cov) * gaussian_scale, 1.0)  
      return tile_ranges(uv - radius, uv + radius, image_size, tile_size)
    
  
  @ti.func
  def gaussian_tile_bounds(
      gaussian: Gaussian2D.vec,
      image_size: ti.math.ivec2,
  ):
      uv, uv_conic, _ = Gaussian2D.unpack(gaussian)
      uv_cov = inverse_cov(uv_conic)

      return cov_tile_ranges(uv, uv_cov, image_size)

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

  
  

  return SimpleNamespace(
    grid_query = obb_grid_query if tight_culling else range_grid_query,
    obb_grid_query = obb_grid_query,
    range_grid_query = range_grid_query,
    separates_bbox = separates_bbox,
    gaussian_tile_bounds = gaussian_tile_bounds,
    cov_tile_ranges = cov_tile_ranges)