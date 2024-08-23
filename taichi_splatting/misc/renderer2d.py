
from dataclasses import replace
from functools import partial
import math
from numbers import Integral
from typing import Optional
from beartype.typing import Tuple
import torch
import torch.nn.functional as F

from taichi_splatting.data_types import Gaussians2D

from taichi_splatting.rasterizer import rasterize, RasterConfig


def project_gaussians2d(points: Gaussians2D) -> torch.Tensor:
    """
    Torch based "projection" of 2D Gaussian parameters to the packed conic-based
    representation used by the tile-mapper and rasterizer.
    Args:
        points: The Gaussians2D parameters to be "projected".
    Returns:
        torch.Tensor [N, 6] the packed
        conic-based representation of the Gaussians2D object.
         
    """
    alpha = torch.sigmoid(points.alpha_logit) 
    sigma = torch.exp(points.log_scaling)

    v1 = points.rotation / torch.norm(points.rotation, dim=1, keepdim=True)

    return torch.cat([points.position, v1, sigma, alpha.unsqueeze(1)], dim=-1)  
    


def point_basis(points:Gaussians2D):
  scale = torch.exp(points.log_scaling)

  v1 = points.rotation / torch.norm(points.rotation, dim=1, keepdim=True)
  v2 = torch.stack([-v1[..., 1], v1[..., 0]], dim=-1)

  return torch.stack([v1, v2], dim=2) * scale.unsqueeze(-1)



def point_rotation(points:Gaussians2D):

  v1 = points.rotation / torch.norm(points.rotation, dim=1, keepdim=True)
  v2 = torch.stack([-v1[..., 1], v1[..., 0]], dim=-1)

  return torch.stack([v1, v2], dim=2) 

def point_covariance(gaussians):
  basis = point_basis(gaussians)
  return torch.bmm(basis.transpose(1, 2), basis)



def split_by_samples(points: Gaussians2D, samples: torch.Tensor, depth_noise:float=1e-2) -> Gaussians2D:
  num_points, n, _ = samples.shape

  basis = point_basis(points)
  point_samples = (samples.view(-1, 2).unsqueeze(1) @ basis.repeat_interleave(repeats=n, dim=0)).squeeze(1)

  gaussians = points.apply(
    partial(torch.repeat_interleave, repeats=n, dim=0), 
    batch_size=[num_points * n])
  
  return replace(gaussians,
    position = gaussians.position + point_samples,
    z_depth = torch.clamp_min(gaussians.z_depth + torch.randn_like(gaussians.z_depth) * depth_noise, 1e-6),
    batch_size=(num_points * n, ))
   

def split_gaussians2d(points: Gaussians2D, n:int=2, scaling:Optional[float]=None, depth_noise:float=1e-2) -> Gaussians2D:
  """
  Toy implementation of the splitting operation used in gaussian-splatting,
  returns a scaled, randomly sampled splitting of the gaussians.

  Args:
      points: The Gaussians2D parameters to be split
      n: number of gaussians to split into
      scale: scale of the new gaussians relative to the original

  Returns:
      Gaussians2D: the split gaussians 
  """

  samples = torch.randn((points.batch_size[0], n, 2), device=points.position.device) 
  if scaling is None:
    scaling = 1 / math.sqrt(n)

  factor = math.log(math.sqrt(1 / n))
  points = replace(points, 
      log_scaling = points.log_scaling + factor,
      batch_size = points.batch_size)

  return split_by_samples(points, samples, depth_noise)


def sample_gaussians(points: Gaussians2D) -> torch.Tensor:
  samples = torch.randn_like(points.position)
  return (samples.unsqueeze(1) @ point_basis(points)).squeeze(1)


def uniform_split_gaussians2d(points: Gaussians2D, n:int=2, scaling:Optional[float]=None, noise=0.0, depth_noise:float=1e-2) -> Gaussians2D:
  """ Split along most significant axis """
  axis = F.one_hot(torch.argmax(points.log_scaling, dim=1), num_classes=2)
  values = torch.linspace(-1, 1, n, device=points.position.device)

  samples = values.view(1, -1, 1) * axis.view(-1, 1, 2)
  samples += torch.randn_like(samples) * noise

  if scaling is None:
    scaling = math.sqrt(n) / n


  factor = math.log(scaling)
  points = replace(points, 
      log_scaling = points.log_scaling + factor * axis,
      batch_size = points.batch_size)
      
      

  return split_by_samples(points, samples, depth_noise)

def resample_inplace(points: Gaussians2D, scale:float=0.625, depth_noise:float=1e-2) -> Gaussians2D:    
    samples = torch.randn_like(points.position)

    points.position += (samples.unsqueeze(1) @ point_basis(points)).squeeze(1)
    points.z_depth += torch.randn_like(points.z_depth) * depth_noise           

    points.log_scaling += math.log(scale)
    return points



def render_gaussians(
      gaussians: Gaussians2D, 
      image_size: Tuple[Integral, Integral],
      raster_config: RasterConfig = RasterConfig()
    ):
  
  gaussians2d = project_gaussians2d(gaussians)
  
  raster = rasterize(gaussians2d=gaussians2d, 
    depths = torch.clamp(gaussians.z_depth, 0, 1),
    features=gaussians.feature, 
    image_size=image_size, 
    config=raster_config)
  
  return raster


