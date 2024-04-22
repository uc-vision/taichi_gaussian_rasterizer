import taichi as ti
from taichi.math import vec4
import torch

from taichi_splatting.data_types import check_packed3d
from taichi_splatting.taichi_lib.f32 import Gaussian3D
from taichi_splatting.torch_ops.transforms import transform44

from .params import CameraParams
 

def planes_from_points(points: torch.Tensor):
   
  normals = torch.cross(points[:, 1] - points[:, 0], 
                              points[:, 2] - points[:, 0])
  normals = normals / torch.norm(normals, dim=1, keepdim=True)

  plane_dists = torch.sum(normals * points[:, 0], dim=1)
  return torch.cat([normals, -plane_dists.unsqueeze(1)], dim=1)


def frustum_points(camera_params: CameraParams):
  near=camera_params.near_plane
  far=camera_params.far_plane
  w, h=camera_params.image_size

  points = [(x * depth, y * depth, depth, 1) 
    for depth in [near, far]
          for x, y in [[0, 0], [w, 0], [w, h], [0, h]] 
  ]
  
  camera_points = torch.tensor(points, dtype=torch.float32, device=camera_params.device)
  world = transform44(camera_params.T_image_world.inverse(), camera_points)
  return world[:, 0:3] / world[:, 3:4]


def frustum_planes(camera_params: CameraParams):
  world = frustum_points(camera_params)
  
  # there are 8 points in total, 4 near, 4 far
  plane_idx = torch.tensor([
      [0, 1, 2], # near
      [6, 5, 4], # far
      [7, 3, 2], # bottom
      [5, 1, 0], # top

      [0, 3, 4], # left
      [6, 2, 1], # right
  ], dtype=torch.int64, device=camera_params.device)

  plane_points = world[plane_idx]
  return planes_from_points(plane_points)


@ti.kernel
def cull_to_planes_kernel(
      gaussians: ti.types.ndarray(Gaussian3D.vec, ndim=1),  # (N, 11)
      output_mask: ti.types.ndarray(ti.u1, ndim=1),  # (N), output
      planes: ti.types.ndarray(vec4, ndim=1),  # (6,)
      gaussian_scale: ti.template(),
  ):
  
  for point_id in range(gaussians.shape[0]):
    position, radius = Gaussian3D.bounding_sphere(gaussians[point_id], gaussian_scale)
    inside = True

    for plane_id in ti.static(range(6)):
      dist = planes[plane_id].dot(vec4(*position, 1))

      # use hard cutoff for near and far planes
      inside = inside and dist >= ti.select(plane_id > 2, -radius, 0)
    
    output_mask[point_id] = inside
                
   
    
def frustum_plane_culling(gaussians: torch.Tensor, 
                    camera_params: CameraParams,  gaussian_scale: float=3.0):
  mask = torch.empty(gaussians.shape[0], dtype=torch.bool, device=gaussians.device)
  check_packed3d(gaussians)

  planes = frustum_planes(camera_params)

  cull_to_planes_kernel(
    gaussians=gaussians,
    output_mask=mask,
    planes = planes,
    gaussian_scale=gaussian_scale
  )

  return mask