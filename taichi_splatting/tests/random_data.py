import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F

from taichi_splatting.data_types import Gaussians2D, Gaussians3D
from taichi_splatting.perspective import CameraParams

from taichi_splatting.torch_lib.transforms import join_rt
from taichi_splatting.torch_lib import projection as torch_proj



def random_camera(pos_scale:float=1., image_size:Optional[Tuple[int, int]]=None, image_size_range:int = (256, 1024),  near_plane = 0.1) -> CameraParams:
  assert near_plane > 0

  q = F.normalize(torch.randn((1, 4)))
  t = torch.randn((3)) * pos_scale

  T_world_camera = join_rt(torch_proj.quat_to_mat(q), t)
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

  projection = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

  return CameraParams(
    T_camera_world=T_camera_world,
    projection=projection,
    image_size=(w, h),
    near_plane=near_plane,
    far_plane=near_plane * 1000.
  )





def random_3d_gaussians(n, camera_params:CameraParams, 
  scale_factor:float=1.0, alpha_range=(0.1, 0.9), margin=0.0) -> Gaussians3D:
  
  w, h = camera_params.image_size
  uv_pos = (torch.rand(n, 2) * (1 + margin) - margin * 0.5) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)

  depth = torch_proj.inverse_ndc_depth(torch.rand(n), camera_params.near_plane, camera_params.far_plane)

  position = torch_proj.unproject_points(uv_pos, depth.unsqueeze(1), camera_params.T_image_world)
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
    alpha_logit=torch_proj.inverse_sigmoid(alpha).unsqueeze(1),
    feature=torch.rand(n, 3),
    batch_size=(n,)
  )


def random_2d_gaussians(n, image_size:Tuple[int, int], 
                        num_channels=3, scale_factor=1.0, alpha_range=(0.1, 0.9), depth_range=(0.0, 1.0)) -> Gaussians2D:
  w, h = image_size

  position = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)
  depth = torch.rand((n, 1)) * (depth_range[1] - depth_range[0]) + depth_range[0]
  
  density_scale = scale_factor * w / (1 + math.sqrt(n))
  scaling = (torch.rand(n, 2) + 0.2) * density_scale 

  rotation = torch.randn(n, 2) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

  low, high = alpha_range
  alpha = torch.rand(n, 1) * (high - low) + low


  return Gaussians2D(
    position=position,
    z_depth=depth,
    log_scaling=torch.log(scaling),
    rotation=rotation,
    alpha_logit=torch_proj.inverse_sigmoid(alpha),
    feature=torch.rand(n, num_channels),
    batch_size=(n,)
  )
