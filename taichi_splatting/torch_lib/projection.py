from beartype.typing import Tuple

import torch
import torch.nn.functional as F

from taichi_splatting.perspective import CameraParams
from taichi_splatting.data_types import Gaussians3D, RasterConfig
from taichi_splatting.taichi_queue import queued
from taichi_splatting.torch_lib.transforms import make_homog, quat_to_mat, transform44

def radii_from_cov(uv_cov:torch.Tensor):
  x, y, _, z = uv_cov.view(-1, 4).unbind(1)
  d = (x - z)

  max_eig_sq = (x + z + torch.sqrt(d * d + 4.0 * y * y)) / 2.0
  return torch.sqrt(max_eig_sq)



def eig(cov: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """ Compute the eigenvalues and eigenvectors of a 2x2 covariance matrix
  """
  assert cov.shape[-2:] == (2, 2), f"Expected ...x2x2 covariance matrix, got {cov.shape}"

  x, y, z = cov[..., 0,0], cov[..., 0,1], cov[..., 1,1]
  tr = x + z
  det = x * z - y * y

  gap = tr**2 - 4 * det
  sqrt_gap = torch.sqrt(torch.clamp_min(gap, 0))

  lam1 = (tr + sqrt_gap) * 0.5
  lam2 = (tr - sqrt_gap) * 0.5

  v1 = F.normalize(torch.stack([x - lam2, y], -1), dim=-1)
  v2 = torch.stack([-v1[..., 1], v1[..., 0]], -1)

  return torch.stack([lam1, lam2], -1).sqrt(), v1, v2


def ellipse_bounds(mean: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  extent = torch.sqrt(v1**2 + v2**2)
  return mean - extent, mean + extent



def inverse_sigmoid(x:torch.Tensor):
  return torch.log(x / (1 - x))

def project_points(transform, xyz):
  homog = transform44(transform, make_homog(xyz))
  depth = homog[..., 2:3]
  xy = homog[..., 0:2] 
  return (xy / depth), depth

def unproject_points(uv, depth, transform):
  points = torch.concatenate(
     [uv * depth, depth, torch.ones_like(depth)], axis=-1)
  transformed = transform44(torch.inverse(transform), points)
  return transformed[..., 0:3] / transformed[..., 3:4]



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


def project_with_jacobian(
    projection: torch.Tensor,  # 4
    position: torch.Tensor,    # N, 3
    image_size: torch.Tensor,  # 2
    clamp_margin: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:         # (Nx2, Nx1, Nx2x3)
    f = projection[:2]
    c = projection[2:]

    z = position[:, 2]

    uv = (position[:, :2] * f) / z.unsqueeze(1) + c
    t = torch.clamp(uv, -clamp_margin * image_size, (1. + clamp_margin) * (image_size - 1))

    zero = torch.zeros_like(uv[:, 0])

    J = torch.stack([
      f[0] / z, zero, -(t[:, 0] - c[0]) / z,
      zero, f[1] / z, -(t[:, 1] - c[1]) / z
    ], dim=1).reshape(-1, 2, 3)

    return uv, z, J

def project_perspective_gaussian(
    J: torch.Tensor,      # N, 2, 3
    cov_in_camera: torch.Tensor         # N, 3, 3
  ) -> torch.Tensor:                    # N, 2, 2
    """ Approximate the 2D gaussian covariance in image space """
  
    # cov_uv = J @ cov_in_camera @ J.transpose()
    cov_uv = torch.einsum('nij,njk,nkl->nil', J, cov_in_camera, J.transpose(1,2))
    return cov_uv

def cov_to_conic(cov: torch.Tensor) -> torch.Tensor:
  """ Convert covariance matrix to conic form
  """
  x, y, z = cov[..., 0, 0], cov[..., 0, 1], cov[..., 1, 1]
  inv_det = 1 / (x * z - y * y)
  return torch.stack([inv_det * z, -inv_det * y, inv_det * x], -1)

@torch.compile
def ndc_depth(depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
  # ndc from 0 (near) to 1 (far)
  return 1 - (1./depth - 1./far) / (1./near - 1./far)


@torch.compile
def inverse_ndc_depth(ndc_depth: torch.Tensor, near: float, far: float) -> torch.Tensor:
  # inverse ndc from 0 (near) to 1 (far) -> depth
  return 1.0 / ((1.0 - ndc_depth) * (1/near - 1/far) + 1/far)

@torch.compile
def generalized_ndc(depth: torch.Tensor, near: float, far: float, k:float) -> torch.Tensor:
  # k = 1 -> linear
  # k = -1 -> ndc (inverse)

  n = near ** k
  f = far ** k
  return (depth.pow(k) - f) / (f - n)



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


def apply(position, log_scaling, rotation, alpha_logit, 
          T_camera_world,
          projection, image_size, depth_range, 
         blur_cov=0.0, clamp_margin=0.15, alpha_threshold=1. / 255.):
  

  T_camera_world = T_camera_world.squeeze(0)
  projection = projection.squeeze(0)

  point_in_camera = transform44(T_camera_world,  make_homog(position))[:, :3]
  image_size = torch.tensor(image_size, dtype=position.dtype, device=position.device)

  mean, z, J = project_with_jacobian(projection, point_in_camera, image_size, clamp_margin)

  cov_in_camera = covariance_in_camera(T_camera_world, F.normalize(rotation, dim=-1), log_scaling.exp())
  cov = project_perspective_gaussian(J, cov_in_camera)
  cov += torch.eye(2, device=cov.device, dtype=cov.dtype) * blur_cov
  
  sigma, v1, v2 = eig(cov)
  alpha = alpha_logit.sigmoid()
  scale = sigma * torch.sqrt(torch.log(alpha_threshold / alpha))
  lower, upper = ellipse_bounds(mean, v1 * scale[:, 0:1], v2 * scale[:, 1:2])

    
  in_view = ((z > depth_range[0]) & (z < depth_range[1]) 
          & (upper > 0).all(1)
          & (lower < image_size.unsqueeze(0)).all(1)
  )

  
  points = torch.concatenate([mean[:, :2], v1, sigma, alpha], axis=-1)
  vis_idx = in_view.nonzero(as_tuple=True)[0]

  return points[in_view], z[in_view].unsqueeze(1), vis_idx


def project_to_image(gaussians:Gaussians3D, camera_params: CameraParams, config: RasterConfig
  ) -> Tuple[torch.Tensor, torch.Tensor]:  

  return apply(*gaussians.shape_tensors(),
          camera_params.T_image_camera, camera_params.T_camera_world, 
          config.blur_cov, config.clamp_margin, config.alpha_threshold)
