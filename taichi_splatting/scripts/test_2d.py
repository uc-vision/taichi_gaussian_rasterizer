import math
import torch
import torch.nn.functional as F
import cv2

from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.perspective.params import CameraParams
from taichi_splatting.renderer import render_gaussians
from taichi_splatting.torch_ops.transforms import join_rt


import taichi as ti

def fov_to_focal(fov, image_size):
  return image_size / (2 * math.tan(fov / 2))

def look_at(eye, target, up=torch.tensor([0., 1., 0.])):
  forward = F.normalize(target - eye, dim=0)
  left = F.normalize(torch.cross(up, forward), dim=0)
  true_up = torch.cross(forward, left)
  R = torch.stack([left, true_up, forward])
  return join_rt(R, eye)


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(0)


def grid_2d(i, j):
  x, y = torch.meshgrid(torch.arange(i), torch.arange(j), indexing='ij')
  return torch.stack([x, y], dim=-1)

def main():
  ti.init(arch=ti.cuda)

  pos = torch.tensor([0.0, 0.0, -15.0])
  target = torch.tensor([0.0, 0.0, 0.0])

  image_size = (512, 512)
  f = fov_to_focal(math.radians(75.0), image_size[0])
  proj = torch.tensor([
    [f, 0, image_size[0] / 2],
    [0, f, image_size[1] / 2],
    [0, 0, 1]
  ])

  world_t_camera = look_at(pos, target)
  print(world_t_camera)

  camera_params = CameraParams(
     T_camera_world=torch.linalg.inv(world_t_camera),
     T_image_camera=proj,
     image_size=image_size,
     near_plane=0.1, far_plane=100.0
     )
  
  points = (grid_2d(5, 5).view(-1, 2).to(torch.float32) - 2) * 4.0
  points_3d = torch.stack([*points.unbind(-1), torch.zeros_like(points[..., 0])], dim=-1)
  n = points_3d.shape[0]

  # position     : torch.Tensor # 3  - xyz
  # log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
  # rotation      : torch.Tensor # 4  - quaternion wxyz
  # alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  # feature      : torch.Tensor # (any rgb (3), spherical harmonics (3x16) etc)

  r = torch.tensor([1.0, 0.0, 0.0, 0.0])

  gaussians = Gaussians3D(
    position = points_3d,
    log_scaling = torch.full_like(points_3d, fill_value=math.log(1.0)),
    rotation = r.view(1, 4).expand(n, -1),
    alpha_logit = torch.full((n, 1), fill_value=100.0),
    feature = torch.rand(n, 3),
    batch_size = (n,)
  )

  device = torch.device('cuda:0')

  image = render_gaussians(gaussians.to(device='cuda:0'), camera_params.to(device)).image

  display_image("image", image)
  


if __name__=="__main__":
   main()