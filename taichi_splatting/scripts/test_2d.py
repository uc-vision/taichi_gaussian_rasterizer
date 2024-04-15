import math
import torch
import torch.nn.functional as F
import cv2

from taichi_splatting.data_types import Gaussians3D, RasterConfig
from taichi_splatting.perspective.params import CameraParams
from taichi_splatting.renderer import render_gaussians
from taichi_splatting.torch_ops.transforms import expand44, join_rt, quat_to_mat

import taichi as ti


def fov_to_focal(fov, image_size):
  return image_size / (2 * math.tan(fov / 2))

def look_at(eye, target, up=torch.tensor([0., 1., 0.])):
  forward = F.normalize(target - eye, dim=0)
  left = F.normalize(torch.cross(forward, up), dim=0)
  true_up = torch.cross(left, forward, dim=0)
  print(true_up)

  R = torch.stack([left, true_up, forward])
  return join_rt(R.T, eye)


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(0)


def grid_2d(i, j):
  x, y = torch.meshgrid(torch.arange(i), torch.arange(j), indexing='ij')
  return torch.stack([x, y], dim=-1)








def project_planes(position, log_scaling, rotation, alpha_logit, indexes,
           T_image_camera, T_camera_world):
  
  position, log_scaling, rotation, alpha_logit = [
     x[indexes] for x in (position, log_scaling, rotation, alpha_logit)] 

  T_camera_world = T_camera_world.squeeze(0)
  T_image_camera = T_image_camera.squeeze(0)

  # point_in_camera = transform44(T_camera_world,  make_homog(position))[:, :3]
  # uv = transform33(T_image_camera, point_in_camera) / point_in_camera[:, 2:3]

  T_image_world = expand44(T_image_camera) @  T_camera_world

  R = quat_to_mat(rotation)

  scale = log_scaling.exp()

  S = torch.eye(3, device=scale.device, dtype=scale.dtype
                  ).unsqueeze(0) * scale.unsqueeze(1)
  
  H = join_rt(R @ S, position)

  WH = T_image_world @ H
  return WH.transpose(-1, -2)



# @ti.kernel
# def render_gaussians(output_image:ti.types.ndarray(type=ti.math.vec3, ndim=2), 
#                      M : ti.types.ndarray(type=ti.math.mat4, ndim=1), features=ti.types.ndarray(ti.math.vec3, ndim=1)):
#   pass



def main():
  ti.init(arch=ti.cuda)

  pos = torch.tensor([0.0, -15.0, 15.0])
  target = torch.tensor([0.0, 0.0, 0.0])

  image_size = (1000, 1000)
  f = fov_to_focal(math.radians(20.0), image_size[0])
  proj = torch.tensor([
    [f, 0, image_size[0] / 2],
    [0, f, image_size[1] / 2],
    [0, 0, 1]
  ])

  world_t_camera = look_at(pos, target)

  device = torch.device('cuda:0')

  camera_params = CameraParams(
     T_camera_world=torch.linalg.inv(world_t_camera),
     T_image_camera=proj,
     image_size=image_size,
     near_plane=0.1, far_plane=100.0
     ).to(device=device)
  
  points = (grid_2d(3, 3).view(-1, 2).to(torch.float32) - 1) * 2
  points_3d = torch.stack([*points.unbind(-1), torch.zeros_like(points[..., 0])], dim=-1)
  n = points_3d.shape[0]

  r = torch.tensor([1.0, 0.0, 0.0, 0.0])
  s = torch.tensor([1.0, 1.0, 0.01]) / math.sqrt(2)

  gaussians = Gaussians3D(
    position = points_3d,
    log_scaling = torch.log(s.view(1, 3)).expand(n, -1),
    rotation = r.view(1, 4).expand(n, -1),
    alpha_logit = torch.full((n, 1), fill_value=100.0),
    feature = torch.rand(n, 3),
    batch_size = (n,)
  ).to(device=device)

  M = project_planes(gaussians.position, gaussians.log_scaling, gaussians.rotation, gaussians.alpha_logit,
                     torch.arange(n), camera_params.T_image_camera, camera_params.T_camera_world)
  
  print(M)


  image = render_gaussians(gaussians, camera_params, config=RasterConfig(beta=50.0)).image
  display_image("image", image)
  


if __name__=="__main__":
   main()