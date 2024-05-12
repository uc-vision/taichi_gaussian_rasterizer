import torch
import math

from taichi_splatting.camera_params import CameraParams
from taichi_splatting.testing.camera import fov_to_focal, look_at
from taichi_splatting.testing.gaussian import gaussian_grid

def test_grid(n, device, scale=2.0, pos=(0.0, 3.5, 0.5)):

  pos = torch.tensor(pos)
  target = torch.tensor([0.0, 0.0, 0.0])

  image_size = (640, 480)
  f = fov_to_focal(math.radians(60.0), image_size[0])

  proj = torch.tensor([
    [f, 0,  image_size[0] / 2],
    [0, f,  image_size[1] / 2],
    [0, 0,  1]
  ])

  world_t_camera = look_at(pos, target)

  camera_params = CameraParams(
     T_camera_world=torch.linalg.inv(world_t_camera),
     T_image_camera=proj,
     image_size=image_size,
     near_plane=0.1, far_plane=100.0
     ).to(device=device)
  

  gaussians = gaussian_grid(n, scale=scale).to(device=device)
  return gaussians, camera_params