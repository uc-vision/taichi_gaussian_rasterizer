import math
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import cv2

from taichi_splatting.data_types import Gaussians3D, RasterConfig
from taichi_splatting.camera_params import CameraParams
from taichi_splatting.conic.renderer import render_gaussians
from taichi_splatting.torch_lib.transforms import expand44, join_rt, make_homog, quat_to_mat, transform33, transform44

import taichi as ti


def fov_to_focal(fov, image_size):
  return image_size / (2 * math.tan(fov / 2))

def look_at(eye, target, up=None):
  if up is None:
    up = torch.tensor([0.0, 1.0, 0.0])


  forward = F.normalize(target - eye, dim=0)
  left = F.normalize(torch.cross(forward, up), dim=0)
  true_up = torch.cross(left, forward, dim=0)
  print(true_up)

  R = torch.stack([left, true_up, forward])
  return join_rt(R.T, eye)


def numpy_image(image):
  return (image.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

def display_image(name, image):
    
    if isinstance(image, torch.Tensor):
      image = numpy_image(image)

    cv2.imshow(name, image)
    cv2.waitKey(0)


def grid_2d(i, j):
  x, y = torch.meshgrid(torch.arange(i), torch.arange(j), indexing='ij')
  return torch.stack([x, y], dim=-1)



def expand_proj(transform:torch.Tensor):
  # expand 3x3 to 4x4 by padding with identity matrix
  prefix = transform.shape[:-2]

  expanded = torch.zeros((4, 4), dtype=transform.dtype, device=transform.device
                       ).view(*[1 for _ in prefix] , 4, 4).expand(*prefix, 4, 4)
  expanded[..., :3, :3] = transform
  expanded[..., 3, 2] = 1.0
  return expanded



def project_planes(position, log_scaling, rotation, alpha_logit, indexes,
           T_image_camera, T_camera_world):
  
  position, log_scaling, rotation, alpha_logit = [
     x[indexes] for x in (position, log_scaling, rotation, alpha_logit)] 
  
  T_camera_world = T_camera_world.squeeze(0)
  T_image_camera = T_image_camera.squeeze(0)


  T_image_world = expand_proj(T_image_camera) @  T_camera_world
  R = quat_to_mat(rotation)

  scale = log_scaling.exp()

  S = torch.eye(3, device=scale.device, dtype=scale.dtype
                  ).unsqueeze(0) * scale.unsqueeze(1)
  
  world_t_splat = join_rt(R @ S, position)
  image_t_splat = T_image_world @ world_t_splat


  return image_t_splat



@ti.kernel
def render_gaussians_kernel(
  output_image:ti.types.ndarray(dtype=ti.math.vec3, ndim=2), 
  depth_image:ti.types.ndarray(dtype=ti.f32, ndim=2),
                     M : ti.types.ndarray(dtype=ti.math.mat4, ndim=1), features:ti.types.ndarray(ti.math.vec3, ndim=1), beta:ti.f32):

  for y, x in output_image:
    for i in range(M.shape[0]):

      m = M[i]

      hu = ti.Vector([-1, 0, 0, x]) @ m 
      hv = ti.Vector([0, -1, 0, y]) @ m 

      
      u = (hu.y * hv.w - hu.w * hv.y) / (hu.x * hv.y - hu.y * hv.x)
      v = (hu.w * hv.x - hu.x * hv.w) / (hu.x * hv.y - hu.y * hv.x)

      p = m @ ti.Vector([u, v, 1., 1.])

      g = ti.exp(-((u**2 + v**2) / 2)**beta)

      if p.z > 0:
        output_image[y, x] += (features[i] * g)
        depth_image[y, x] = ti.min(p.z, depth_image[y, x])


def gaussian_grid(n, scale=2):
  points = (grid_2d(n, n).view(-1, 2).to(torch.float32) - n // 2) * 2 * scale 
  # points = (grid_2d(1, 1).view(-1, 2).to(torch.float32)) * 2

  points_3d = torch.stack([*points.unbind(-1), torch.zeros_like(points[..., 0])], dim=-1)
  n = points_3d.shape[0]

  r = torch.tensor([1.0, 0.0, 0.0, 0.0])
  s = torch.tensor([1.0, 1.0, 0.00001]) * scale / math.sqrt(2)

  return Gaussians3D(
    position = points_3d,
    log_scaling = torch.log(s.view(1, 3)).expand(n, -1),
    rotation = r.view(1, 4).expand(n, -1),
    alpha_logit = torch.full((n, 1), fill_value=100.0),
    feature = torch.rand(n, 3),
    batch_size = (n,)
  )


def main():
  torch.set_printoptions(precision=4, sci_mode=False)
  ti.init(arch=ti.cuda)

  device = torch.device('cuda:0')
  torch.set_default_device(device)


  pos = torch.tensor([4.0, 4.0, 3.0])
  target = torch.tensor([0.0, 0.0, 0.0])

  image_size = (1000, 1000)
  f = fov_to_focal(math.radians(60.0), image_size[0])

  proj = torch.tensor([
    [f, 0,  image_size[0] / 2],
    [0, f,  image_size[1] / 2],
    [0, 0,  1]
  ])

  world_t_camera = look_at(pos, target)

  
  T_camera_world=torch.linalg.inv(world_t_camera)
  T_image_camera=proj

  camera_params = CameraParams(
     T_camera_world=torch.linalg.inv(world_t_camera),
     T_image_camera=proj,
     image_size=image_size,
     near_plane=0.1, far_plane=100.0
     ).to(device=device)
  

  gaussians = gaussian_grid(5, scale=1.0).to(device=device)


  image_t_splat = project_planes(gaussians.position, gaussians.log_scaling, gaussians.rotation, gaussians.alpha_logit,
                     torch.arange(gaussians.batch_size[0]), T_image_camera, T_camera_world).contiguous()

  def find_threshold(alpha, beta):
    x = np.linspace(0, 4, 10000)
    y = np.exp(-(0.5 * x**2) ** beta)
    return x[np.argmax(y < alpha)]

  config = RasterConfig(beta=10.0, alpha_threshold=0.01)


  r=find_threshold(config.alpha_threshold, config.beta)
  uv = torch.Tensor([[-r, -r, 1, 1], [r, -r, 1, 1], [r, r, 1, 1], [-r, r, 1, 1]]).to(device=device)

  # project corners
  corners = torch.einsum('nij,mj->nmi', image_t_splat, uv) 
  corners = corners[:, :, 0:2] / corners[:, :, 2:3]
  

  output_image = torch.zeros((*reversed(image_size), 3), dtype=torch.float32, device=device)
  depth_image = torch.zeros(image_size, dtype=torch.float32, device=device)

  render_gaussians_kernel(output_image, depth_image, image_t_splat, gaussians.feature, beta=config.beta)


  output_image = numpy_image(output_image)

  for box in corners.cpu().numpy().astype(int):
    for i in range(4):
      a, b = box[i], box[(i + 1) % 4]
      cv2.line(output_image, a, b, (255, 255, 255))


  display_image("output_image", output_image)



  
  # print(config)

  # render = render_gaussians(gaussians, camera_params, config=config, compute_radii=True)
  # image = numpy_image(render.image)

  # for uv, radii in zip(render.gaussians_2d[:, :2], render.radii):
  #   uv = uv.cpu().numpy()
  #   radius = int(radii.item())

  #   corners = uv.reshape(1, -1) + np.array([[-radius, -radius], [radius, -radius], [radius, radius], [-radius, radius]])
  #   corners = corners.astype(int)
  #   for i in range(4):
  #     a, b = corners[i], corners[(i + 1) % 4]
  #     cv2.line(image, tuple(a), tuple(b), (255, 255, 255))

  # display_image("image", image)
  


if __name__=="__main__":
   main()