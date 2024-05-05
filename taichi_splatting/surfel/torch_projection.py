import math
import numpy as np
import torch
import cv2

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.camera_params import CameraParams, expand_proj
from taichi_splatting.surfel.project_bounds import surfel_bounds
from taichi_splatting.testing.camera import fov_to_focal, look_at
from taichi_splatting.testing.gaussian import gaussian_grid
from taichi_splatting.torch_lib.transforms import join_rt,  quat_to_mat
from taichi_splatting.surfel import gaussian3d_to_surfel


import taichi as ti
import taichi_splatting.taichi_lib.f32 as lib




def numpy_image(image):
  return (image.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

def display_image(name, image):
    
    if isinstance(image, torch.Tensor):
      image = numpy_image(image)

    cv2.imshow(name, image)
    cv2.waitKey(0)



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
def render_surfel_kernel(
  output_image:ti.types.ndarray(dtype=ti.math.vec3, ndim=2), 
  depth_image:ti.types.ndarray(dtype=ti.f32, ndim=2),

  surfels : ti.types.ndarray(dtype=lib.GaussianSurfel.vec, ndim=1), 
  projection_arr : ti.types.ndarray(dtype=ti.f32, ndim=2),
  features:ti.types.ndarray(ti.math.vec3, ndim=1), beta:ti.f32):

  for y, x in ti.ndrange(*output_image.shape):
    projection = lib.mat4_from_ndarray(projection_arr)

    for i in range(surfels.shape[0]):

      surfel = lib.GaussianSurfel.from_vec(surfels[i])
      image_t_surface =  projection @ surfel.world_t_surface() 
      
      g, depth = lib.eval_surfel(image_t_surface, lib.vec2(x, y), beta)

      if depth > 0:
        output_image[y, x] += (features[i] * g)
        depth_image[y, x] = ti.min(depth, depth_image[y, x])



def find_threshold(alpha, beta):
  x = np.linspace(0, 4, 10000)
  y = np.exp(-(0.5 * x**2) ** beta)
  return x[np.argmax(y < alpha)]

def main():
  torch.set_printoptions(precision=4, sci_mode=False)
  ti.init(arch=ti.cuda)

  device = torch.device('cuda:0')
  torch.set_default_device(device)


  config = RasterConfig(beta=10.0, alpha_threshold=0.01)
  r=find_threshold(config.alpha_threshold, config.beta)


  gaussians, camera_params = test_grid(5, device)

  surfel = gaussian3d_to_surfel(gaussians, torch.arange(0, gaussians.batch_size[0], device=device), torch.eye(4, device=device)) #camera_params.T_camera_world)
  bounds = surfel_bounds(surfel, camera_params.T_image_world, gaussian_scale=r)
  print(bounds)




  uv = torch.Tensor([[-r, -r, 0, 1], [r, -r, 0, 1], [r, r, 0, 1], [-r, r, 0, 1]]).to(device=device)


  image_t_splat = project_planes(gaussians.position, gaussians.log_scaling, gaussians.rotation, gaussians.alpha_logit,
                     torch.arange(gaussians.batch_size[0]), camera_params.T_image_camera, camera_params.T_camera_world).contiguous()

  
  print("image_t_splat", image_t_splat)

  # project corners
  corners = torch.einsum('nij,mj->nmi', image_t_splat, uv)
  corners = corners[:, :, 0:2] / corners[:, :, 2:3]



  image_shape = tuple(reversed(camera_params.image_size))
  output_image = torch.zeros((*image_shape, 3), dtype=torch.float32, device=device)
  depth_image = torch.zeros(image_shape, dtype=torch.float32, device=device)

  # render_gaussians_kernel(output_image, depth_image, image_t_splat, gaussians.feature, beta=config.beta)
  render_surfel_kernel(output_image, depth_image, surfel, camera_params.T_image_world, gaussians.feature, beta=config.beta)

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