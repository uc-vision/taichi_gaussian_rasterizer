
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from taichi_splatting.conic.perspective import CameraParams

from taichi_splatting.torch_lib import transforms

def fov_to_focal(fov, image_size):
  return image_size / (2 * math.tan(fov / 2))

def look_at(eye, target, up=None):
  if up is None:
    up = torch.tensor([0.0, 1.0, 0.0])


  forward = F.normalize(target - eye, dim=0)
  left = F.normalize(torch.cross(forward, up), dim=0)
  true_up = torch.cross(left, forward, dim=0)

  R = torch.stack([left, true_up, forward])
  return transforms.join_rt(R.T, eye)


def random_camera(pos_scale:float=1., image_size:Optional[Tuple[int, int]]=None, image_size_range:int = (256, 1024)) -> CameraParams:
  q = F.normalize(torch.randn((1, 4)))
  t = torch.randn((3)) * pos_scale

  T_world_camera = transforms.join_rt(transforms.quat_to_mat(q), t)
  T_camera_world = torch.inverse(T_world_camera)

  if image_size is None:
    min_size, max_size = image_size_range
    image_size = [x.item() for x in torch.randint(size=(2,), 
              low=min_size, high=max_size)]

  w, h = image_size
  cx, cy = torch.tensor([w/2, h/2]) + torch.randn(2) * (w / 20) 

  fov = torch.deg2rad(torch.rand(1) * 70 + 30)
  fx = w / (2 * torch.tan(fov / 2))
  fy = h / (2 * torch.tan(fov / 2))

  T_image_camera = torch.tensor([
    [fx, 0,  cx],
    [0,  fy, cy],
    [0,  0,  1]
  ])

  near_plane = 0.1
  assert near_plane > 0

  return CameraParams(
    T_camera_world=T_camera_world,
    T_image_camera=T_image_camera,
    image_size=(w, h),
    near_plane=near_plane,
    far_plane=near_plane * 1000.
  )
