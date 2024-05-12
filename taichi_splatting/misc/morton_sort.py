from typing import Optional, Tuple
import taichi as ti
from taichi.math import vec3, ivec3, clamp

from taichi_splatting.taichi_lib.f32 import AABBox

import torch




# https://stackoverflow.com/questions/
# 1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints 
@ti.func
def spreads_bits32(x:ti.int32) -> ti.int32:
  x &= 0x3ff
  x = (x | (x << 16)) & 0x030000FF
  x = (x | (x <<  8)) & 0x0300F00F
  x = (x | (x <<  4)) & 0x030C30C3
  x = (x | (x <<  2)) & 0x09249249
  return x

@ti.func
def spreads_bits64(x:ti.int64) -> ti.int64:
  x &= 0x1fffff
  x = (x | (x << 32)) & ti.int64(0x1f00000000ffff)
  x = (x | (x << 16)) & ti.int64(0x1f0000ff0000ff)
  x = (x | (x << 8)) & ti.int64(0x100f00f00f00f00f)
  x = (x | (x << 4)) & ti.int64(0x10c30c30c30c30c3)
  x = (x | (x << 2)) & ti.int64(0x1249249249249249)
  return x





@ti.dataclass
class Grid:

  lower:vec3
  upper:vec3

  size:ivec3
  

  @ti.func
  def get_inc(self) -> Tuple[vec3, vec3]:
    inc = (self.upper - self.lower) / ti.cast(self.size, ti.f32)
    return self.lower, inc



  @ti.func 
  def grid_cell(self, p:vec3) -> ivec3:
    lower, inc = self.get_inc()
    v = (p - lower) / inc
    return ti.cast(clamp(v, 0, self.size - 1), ti.i32)


  @ti.func
  def cell_bounds(self, cell:ivec3) -> AABBox:
    lower, inc = self.get_inc()
    return AABBox(lower + cell * inc, lower + (cell + 1) * inc)

  @ti.func
  def in_bounds(self, cell:ivec3) -> bool:
    return (0 <= cell).all() and (cell < self.size).all()



  @ti.func
  def cell_code64(self, cell:ivec3) -> ti.int64:
    cell64 = ti.cast(cell, ti.int64)
    
    return (spreads_bits64(cell64.x) 
      | (spreads_bits64(cell64.y) << 1) 
      | (spreads_bits64(cell64.z) << 2))

  @ti.func
  def morton_code64(self, p:vec3) -> ti.int64:
    cell = self.grid_cell(p)
    return self.cell_code64(cell)


  @ti.func
  def cell_code32(self, cell:ivec3) -> ti.int32:
    return (spreads_bits32(cell.x) 
      | (spreads_bits32(cell.y) << 1) 
      | (spreads_bits32(cell.z) << 2))

  @ti.func
  def morton_code32(self, p:vec3) -> ti.int32:
    cell = self.grid_cell(p)
    return self.cell_code32(cell)



@ti.kernel
def code_points32_kernel(
  grid:Grid,
  points:ti.types.ndarray(vec3, ndim=1), 
  codes:ti.types.ndarray(ti.int32, ndim=1)):
  
  for i in range(points.shape[0]):
    codes[i] = grid.morton_code32(points[i])



@ti.kernel
def code_points64_kernel(
  grid:Grid,
  points:ti.types.ndarray(vec3, ndim=1), 
  codes:ti.types.ndarray(ti.int64, ndim=1)):
  
  for i in range(points.shape[0]):
    codes[i] = grid.morton_code64(points[i])


def grid_from_points(points:torch.Tensor, size:ti.math.ivec3) -> Grid:
  lower = points.min(dim=0).values
  upper = points.max(dim=0).values
  return Grid(vec3(lower), vec3(upper), size)

def morton_argsort32(points:torch.Tensor, grid:Optional[Grid] = None):  
  if grid is None:
    grid = grid_from_points(points, ti.math.ivec3(1024))

  codes = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
  code_points32_kernel(grid, points, codes)
  return torch.argsort(codes)

def morton_sort32(points:torch.Tensor, grid:Optional[Grid] = None):
  return points[morton_argsort32(points, grid)]
    

def morton_argsort64(points:torch.Tensor, grid:Optional[Grid] = None):  
  if grid is None:
    grid = grid_from_points(points, ti.math.ivec3(2**20))

  codes = torch.zeros(points.shape[0], dtype=torch.int64, device=points.device)
  code_points64_kernel(grid, points, codes)
  return torch.argsort(codes)

def morton_sort64(points:torch.Tensor, grid:Optional[Grid] = None):
  return points[morton_argsort64(points, grid)]




if __name__ == "__main__":
  ti.init(arch=ti.gpu)

  points = torch.rand(1000, 3).to(torch.float32).cuda()

  grid = grid_from_points(points, ti.math.ivec3(1024))
  morton_argsort64(points, grid)
  
