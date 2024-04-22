from beartype.typing import Tuple

import torch
import torch.nn.functional as F

from taichi_splatting.conic.perspective import CameraParams
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.torch_lib.transforms import make_homog, quat_to_mat, transform33, transform44






def covariance_in_camera(
    T_camera_world: torch.Tensor,  # 4, 4  
    cov_rotation: torch.Tensor,    # N, 4
    cov_scale: torch.Tensor,       # N, 3
) -> torch.Tensor:                 # N, 3, 3
    """ Construct and rotate the covariance matrix in camera space
    """
    W = T_camera_world[:3, :3]
    R = quat_to_mat(cov_rotation)
    S = torch.eye(3, device=cov_scale.device, dtype=cov_scale.dtype
                  ).unsqueeze(0) * cov_scale.unsqueeze(1)
    
    m = W @ R @ S
    return m @ m.transpose(1, 2)

def get_projective_transform_jacobian(
    projection: torch.Tensor,  # 3, 3
    position: torch.Tensor,    # N, 3
) -> torch.Tensor:             # N, 2, 3
    fx = projection[0, 0]
    fy = projection[1, 1]

    x, y, z = position.T
        # [fx / z, 0, -(fx * x) / (z * z)],
        # [0, fy / z, -(fy * y) / (z * z)]
    
    zero = torch.zeros_like(x)
    return torch.stack([
      fx / z, zero, -(fx * x) / (z * z),
      zero, fy / z, -(fy * y) / (z * z)
    ], dim=1).reshape(-1, 2, 3)

def project_perspective_gaussian(
    projective_transform: torch.Tensor, # 3, 3
    point_in_camera: torch.Tensor,      # N, 3
    cov_in_camera: torch.Tensor         # N, 3, 3
  ) -> torch.Tensor:                    # N, 2, 2
    """ Approximate the 2D gaussian covariance in image space """
    
    J = get_projective_transform_jacobian(
        projective_transform, point_in_camera) # N, 2, 3
    
    # cov_uv = J @ cov_in_camera @ J.transpose()
    cov_uv = torch.einsum('nij,njk,nkl->nil', J, cov_in_camera, J.transpose(1,2))
    return cov_uv

def cov_to_conic(cov: torch.Tensor) -> torch.Tensor:
  """ Convert covariance matrix to conic form
  """
  x, y, z = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
  inv_det = 1 / (x * z - y * y)
  return torch.stack([inv_det * z, -inv_det * y, inv_det * x], -1)


  # inv_cov = torch.inverse(cov)
  # return torch.stack(
  #    [inv_cov[..., 0, 0], 
  #     inv_cov[..., 0, 1], 
  #     inv_cov[..., 1, 1]], dim=-1)


def unpack_activate(vec: torch.Tensor
      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  position = vec[..., 0:3]
  log_scaling = vec[..., 3:6]
  rotation = vec[..., 6:10]
  alpha_logit = vec[..., 10:11]

  return  (position, 
           torch.exp(log_scaling), 
           rotation / torch.norm(rotation, dim=-1, keepdim=True),
           torch.sigmoid(alpha_logit)
  )


def apply(position, log_scaling, rotation, alpha_logit, indexes,
           T_image_camera, T_camera_world, blur_cov=0.3):
  
  position, log_scaling, rotation, alpha_logit = [
     x[indexes] for x in (position, log_scaling, rotation, alpha_logit)] 

  T_camera_world = T_camera_world.squeeze(0)
  T_image_camera = T_image_camera.squeeze(0)

  point_in_camera = transform44(T_camera_world,  make_homog(position))[:, :3]
  uv = transform33(T_image_camera, point_in_camera) / point_in_camera[:, 2:3]


  cov_in_camera = covariance_in_camera(T_camera_world, F.normalize(rotation, dim=-1), log_scaling.exp())
  uv_cov = project_perspective_gaussian(T_image_camera, point_in_camera, cov_in_camera)

  uv_cov += torch.eye(2, device=uv_cov.device, dtype=uv_cov.dtype) * blur_cov

  points = torch.concatenate([uv[:, :2], cov_to_conic(uv_cov), alpha_logit.sigmoid()], axis=-1)
  depths = torch.stack([point_in_camera[:, 2], cov_in_camera[:, 2, 2], point_in_camera[:, 2] ** 2], axis=-1)

  return points, depths.contiguous()


def project_to_image(gaussians:Gaussians3D, camera_params: CameraParams
  ) -> Tuple[torch.Tensor, torch.Tensor]:  

  return apply(*gaussians.shape_tensors(),
          camera_params.T_image_camera, camera_params.T_camera_world, 
          camera_params.blur_cov)
