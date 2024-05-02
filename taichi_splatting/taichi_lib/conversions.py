from math import prod
import taichi as ti

import torch


torch_taichi = {
    torch.float16: ti.f16,
    torch.float32: ti.f32,
    torch.float64: ti.f64,
    torch.int32: ti.i32,
    torch.int64: ti.i64,
    torch.int8: ti.i8,
    torch.int16: ti.i16,
    torch.uint8: ti.u8,
    # torch.uint16: ti.u16,
}

taichi_torch = {v:k for k,v in torch_taichi.items()}


def struct_size(ti_struct:ti.lang.struct.StructType):
  size = 0
  for k, v in ti_struct.members.items():

    if isinstance(v, ti.lang.matrix.MatrixType):
      size += prod(v.get_shape())
    elif isinstance(v, ti.lang.matrix.VectorType):
      size += prod(v.get_shape())
    
    elif isinstance(v, ti.lang.struct.StructType):
      size += struct_size(v)
    else:
      size += 1
  return int(size)

