import math
from typing import Tuple
import torch
from taichi_splatting.data_types import Gaussians2D, Gaussians3D
from taichi_splatting.perspective import CameraParams

from taichi_splatting.torch_ops.transforms import join_rt
from taichi_splatting.torch_ops import projection as torch_proj


def random_camera(pos_scale:float=1., max_image_size:int = 1024) -> CameraParams:
  q = torch.randn((1, 4))
  t = torch.randn((3)) * pos_scale

  T_world_camera = join_rt(torch_proj.quat_to_mat(q), t)
  T_camera_world = torch.inverse(T_world_camera)

  w, h = [x.item() for x in torch.randint(size=(2,), 
            low=max_image_size // 5, high=max_image_size)]
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


def random_3d_gaussians(n, camera_params:CameraParams, 
  scale_factor:float=1.0, alpha_range=(0.1, 0.9)) -> Gaussians3D:
  
  w, h = camera_params.image_size

  uv_pos = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)

  depth_range = camera_params.far_plane - camera_params.near_plane
  depth = torch.rand(n) * depth_range + camera_params.near_plane   

  position = torch_proj.unproject_points(uv_pos, depth.unsqueeze(1), camera_params.T_image_world)
  fx = camera_params.T_image_camera[0, 0]

  scale =  (w / math.sqrt(n)) * (depth / fx) * scale_factor
  scaling = (torch.rand(n, 3) + 0.2) * scale.unsqueeze(1) 

  rotation = torch.randn(n, 4) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

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



def inv_softplus(x):
  return x + torch.log(-torch.expm1(-x))


def random_2d_gaussians(n, image_size:Tuple[int, int], scale_factor=1.0, alpha_range=(0.1, 0.9), depth_range=(0.1, 100.0), beta_range=(0.5, 1.5)):
  w, h = image_size

  position = torch.rand(n, 2) * torch.tensor([w, h], dtype=torch.float32).unsqueeze(0)
  depth = torch.rand((n, 1)) * (depth_range[1] - depth_range[0]) + depth_range[0]
  
  density_scale = scale_factor * w / (1 + math.sqrt(n))
  scaling = (torch.rand(n, 2) + 0.2) * density_scale 

  rotation = torch.randn(n, 2) 
  rotation = rotation / torch.norm(rotation, dim=1, keepdim=True)

  low, high = alpha_range
  alpha = torch.rand(n) * (high - low) + low

  beta = inv_softplus(torch.rand(n) * (beta_range[1] - beta_range[0]) + beta_range[0] - 0.5) 

  return Gaussians2D(
    position=position,
    depth=depth,
    log_scaling=torch.log(scaling),
    rotation=rotation,
    alpha_logit=torch_proj.inverse_sigmoid(alpha),
    beta=beta,

    feature=torch.rand(n, 3),
    batch_size=(n,)
  )
