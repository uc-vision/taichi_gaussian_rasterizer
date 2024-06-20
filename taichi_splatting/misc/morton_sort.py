from collections import namedtuple
from typing import Optional, Tuple
import taichi as ti
from taichi.math import vec3, uvec3, clamp, uvec3
from taichi_splatting import cuda_lib   

import torch


# https://stackoverflow.com/questions/
# 1024754/how-to-compute-a-3d-morton-number-interleave-the-bits-of-3-ints 
@ti.func
def spreads_bits32(x:ti.uint32) -> ti.uint32:
  x &= 0x3ff
  x = (x | (x << 16)) & 0x030000FF
  x = (x | (x <<  8)) & 0x0300F00F
  x = (x | (x <<  4)) & 0x030C30C3
  x = (x | (x <<  2)) & 0x09249249
  return x

@ti.func
def spreads_bits64(x:ti.uint64) -> ti.uint64:
  x &= 0x1fffff
  x = (x | (x << 32)) & ti.uint64(0x1f00000000ffff)
  x = (x | (x << 16)) & ti.uint64(0x1f0000ff0000ff)
  x = (x | (x << 8)) & ti.uint64(0x100f00f00f00f00f)
  x = (x | (x << 4)) & ti.uint64(0x10c30c30c30c30c3)
  x = (x | (x << 2)) & ti.uint64(0x1249249249249249)
  return x


@ti.dataclass
class Grid:

  lower:vec3
  upper:vec3

  size:uvec3
  
  @ti.func
  def get_inc(self) -> Tuple[vec3, vec3]:
    inc = (self.upper - self.lower) / ti.cast(self.size, ti.f32)
    return self.lower, inc



  @ti.func 
  def grid_cell(self, p:vec3) -> uvec3:
    lower, inc = self.get_inc()
    v = (p - lower) / inc
    return ti.cast(clamp(v, 0, self.size - 1), ti.u32)


  @ti.func
  def in_bounds(self, cell:uvec3) -> bool:
    return (0 <= cell).all() and (cell < self.size).all()



  @ti.func
  def cell_code64(self, cell:uvec3) -> ti.uint64:
    cell64 = ti.cast(cell, ti.uint64)
    
    return (spreads_bits64(cell64.x) 
      | (spreads_bits64(cell64.y) << 1) 
      | (spreads_bits64(cell64.z) << 2))

  @ti.func
  def morton_code64(self, p:vec3) -> ti.uint64:
    cell = self.grid_cell(p)
    return self.cell_code64(cell)


  @ti.func
  def cell_code32(self, cell:uvec3) -> ti.uint32:
    return (spreads_bits32(cell.x) 
      | (spreads_bits32(cell.y) << 1) 
      | (spreads_bits32(cell.z) << 2))

  @ti.func
  def morton_code32(self, p:vec3) -> ti.uint32:
    cell = self.grid_cell(p)
    return self.cell_code32(cell)



@ti.kernel
def code_points32_kernel(
  grid:Grid,
  points:ti.types.ndarray(vec3, ndim=1), 
  codes:ti.types.ndarray(ti.uint32, ndim=1)):
  
  for i in range(points.shape[0]):
    codes[i] = grid.morton_code32(points[i])



@ti.kernel
def code_points64_kernel(
  grid:Grid,
  points:ti.types.ndarray(vec3, ndim=1), 
  codes:ti.types.ndarray(ti.int64, ndim=1)):
  
  for i in range(points.shape[0]):
    codes[i] = ti.cast(grid.morton_code64(points[i]), ti.int64)


def grid_at_resolution(points:torch.Tensor, resolution:float, size:int = 2**20) -> Grid:
  lower = points.min(dim=0).values
  upper = lower + size * resolution

  return Grid(vec3(lower), vec3(upper), size)


SortResult = namedtuple("values", "indices")


def _sort(points:torch.Tensor, resolution:float): 
  grid = grid_at_resolution(points, resolution, size=2**20)

  codes = torch.empty(points.shape[0], dtype=torch.int64, device=points.device)
  code_points64_kernel(grid, points.contiguous(), codes)

  idx = torch.arange(points.shape[0], device=points.device, dtype=torch.int64)
  return cuda_lib.radix_sort_pairs(codes, idx)


def sort(points:torch.Tensor, resolution:float): 
  _, idx = _sort(points, resolution)
  return SortResult(points[idx], idx)

def argsort(points:torch.Tensor, resolution:float):
  _, idx = _sort(points, resolution)
  return idx


def argsort_dedup(points:torch.Tensor, resolution:float): 
  codes, idx = _sort(points, resolution)

  codes, counts = torch.unique_consecutive(codes, return_counts=True)
  unique_idx = torch.cumsum(counts, dim=0) - 1

  return idx[unique_idx]


def sort_dedup(points:torch.Tensor, resolution:float):
  idx = argsort_dedup(points, resolution)
  return SortResult(points[idx], idx)


if __name__ == "__main__":
  ti.init(arch=ti.gpu)
  points_int = torch.randint(0, 10, (1000, 3)).cuda() 
  unique, unique_idx = torch.unique(points_int, dim=0, return_inverse=True)
  
  points = points_int.to(torch.float32) + 0.5

  idx1 = argsort(points, 1.0)
  idx2 = argsort(points * 2 - 4, 2.0)
  
  assert torch.allclose(idx1, idx2)

  idx3 = argsort_dedup(points, 1.0)
