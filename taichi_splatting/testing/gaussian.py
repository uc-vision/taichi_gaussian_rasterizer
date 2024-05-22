import math
from typing import Tuple

import torch
import torch.nn.functional as F

from taichi_splatting.data_types import Gaussians2D, Gaussians3D
from taichi_splatting.conic.perspective import CameraParams

from taichi_splatting.torch_lib import transforms
from taichi_splatting.torch_lib.util import inverse_sigmoid


def grid_2d(i, j):
  x, y = torch.meshgrid(torch.arange(i), torch.arange(j), indexing='ij')
  return torch.stack([x, y], dim=-1)


def gaussian_grid(n, scale=2):
  points = (grid_2d(n, n).view(-1, 2).to(torch.float32) - n // 2) * 2 * scale 

  points_3d = torch.stack([*points.unbind(-1), torch.zeros_like(points[..., 0])], dim=-1)
  n = points_3d.shape[0]

  r = torch.tensor([1.0, 0.0, 0.0, 0.0])
  s = torch.tensor([0.2, 4.0, 1e-6]) * scale / math.sqrt(2)

  return Gaussians3D(
    position = points_3d,
    log_scaling = torch.log(s.view(1, 3)).expand(n, -1),
    rotation = r.view(1, 4).expand(n, -1),
    alpha_logit = torch.full((n, 1), fill_value=100.0),
    feature = torch.rand(n, 3),
    batch_size = (n,)
  )




def random_3d_gaussians(n, camera_params:CameraParams, 
  scale_factor:float=1.0, alpha_range=(0.1, 0.9)) -> Gaussians3D:
  
  w, h = camera_params.image_size
  uv_pos = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)

  depth_range = camera_params.far_plane - camera_params.near_plane
  depth = torch.rand(n) * depth_range + camera_params.near_plane   

  position = transforms.unproject_points(uv_pos, depth.unsqueeze(1), camera_params.T_image_world)
  fx = camera_params.T_image_camera[0, 0]

  scale =  (w / math.sqrt(n)) * (depth / fx) * scale_factor
  scaling = (torch.rand(n, 3) + 0.2) * scale.unsqueeze(1) 

  rotation = torch.randn(n, 4) 
  rotation = F.normalize(rotation, dim=1)

  low, high = alpha_range
  alpha = torch.rand(n) * (high - low) + low

  return Gaussians3D(
    position=position,
    log_scaling=torch.log(scaling),
    rotation=rotation,
    alpha_logit=inverse_sigmoid(alpha).unsqueeze(1),
    feature=torch.rand(n, 3),
    batch_size=(n,)
  )


def random_2d_gaussians(n, image_size:Tuple[int, int], num_channels=3, scale_factor=1.0, alpha_range=(0.1, 0.9), depth_range=(0.1, 100.0)):
  w, h = image_size

  position = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)
  depth = torch.rand((n, 1)) * (depth_range[1] - depth_range[0]) + depth_range[0]
  
  density_scale = scale_factor * w / (1 + math.sqrt(n))
  scaling = (torch.rand(n, 2) + 0.2) * density_scale 

  rotation = torch.randn(n, 2) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

  low, high = alpha_range
  alpha = torch.rand(n) * (high - low) + low

  return Gaussians2D(
    position=position,
    z_depth=depth,
    log_scaling=torch.log(scaling),
    rotation=rotation,
    alpha_logit=inverse_sigmoid(alpha),
    feature=torch.rand(n, num_channels),
    batch_size=(n,)
  )
