from dataclasses import dataclass, replace
from beartype.typing import Tuple
from beartype import beartype
import torch

from torch.nn import functional as F
from tensordict import tensorclass


  
@beartype
@dataclass(frozen=True, eq=True, kw_only=True)
class RasterConfig:
  tile_size: int = 16

  # pixel tilin per thread in the backwards pass 
  pixel_stride: Tuple[int, int] = (2, 2)


  # clamp position to within this margin of the image for affine jaocbian
  clamp_margin: float = 0.15  
  
  # Use anti-aliasing implementation
  antialias: bool = False  

  # blur covariance matrix by this factor
  blur_cov: float = 0.3

  clamp_max_alpha: float = 0.99
  alpha_threshold: float = 1. / 255.

  # stop alpha blending at this point
  saturate_threshold: float = 0.9999

  # use alpha blending - if set to false, with saturate_threshold can be used to compute quantile (e.g. median)
  use_alpha_blending: bool = True

  compute_point_heuristic: bool = False # implies compute_visibility
  compute_visibility: bool = False # compute visibility (pixels) for each gaussian



    

def check_packed3d(packed_gaussians: torch.Tensor):
  assert len(packed_gaussians.shape) == 2 and packed_gaussians.shape[1] == 11, f"Expected shape (N, 11), got {packed_gaussians.shape}"  

def check_packed2d(packed_gaussians: torch.Tensor):
  assert len(packed_gaussians.shape) == 2 and packed_gaussians.shape[1] == 6, f"Expected shape (N, 6), got {packed_gaussians.shape}"  


@tensorclass
class Gaussians3D():
  position     : torch.Tensor # 3  - xyz
  log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 4  - quaternion wxyz
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  feature      : torch.Tensor # (any rgb (3), spherical harmonics (3x16) etc)


  def __post_init__(self):
    assert self.position.shape[1] == 3, f"Expected shape (N, 3), got {self.position.shape}"
    assert self.log_scaling.shape[1] == 3, f"Expected shape (N, 3), got {self.log_scaling.shape}"
    assert self.rotation.shape[1] == 4, f"Expected shape (N, 4), got {self.rotation.shape}"
    assert self.alpha_logit.shape[1] == 1, f"Expected shape (N, 1), got {self.alpha_logit.shape}"


  def packed(self):
    return torch.cat([self.position, self.log_scaling, self.rotation, self.alpha_logit], dim=-1)
  
  def shape_tensors(self):
    return (self.position, self.log_scaling, self.rotation, self.alpha_logit)

  @property
  def scale(self):
    return torch.exp(self.log_scaling)

  @property
  def alpha(self):
    return torch.sigmoid(self.alpha_logit)

  
  def replace(self, **kwargs):
    return replace(self, **kwargs, batch_size=self.batch_size)
  
  def concat(self, other):
    return replace(self,
      position=torch.cat([self.position, other.position], dim=0),
      log_scaling=torch.cat([self.log_scaling, other.log_scaling], dim=0),
      rotation=torch.cat([self.rotation, other.rotation], dim=0),
      alpha_logit=torch.cat([self.alpha_logit, other.alpha_logit], dim=0),
      feature=torch.cat([self.feature, other.feature], dim=0),
      batch_size = (self.batch_size[0] + other.batch_size[0], )
    )
  
def inverse_sigmoid(x:torch.Tensor):
  return torch.log(x / (1 - x))


@tensorclass
class Gaussians2D():
  position     : torch.Tensor # 2  - xy
  z_depth        : torch.Tensor # 1  - for sorting
  log_scaling   : torch.Tensor # 2

  rotation      : torch.Tensor # 2  - unit length imaginary number
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  feature      : torch.Tensor # N  - (any rgb, label etc)

  @property
  def opacity(self):
    return self.alpha_logit.sigmoid()
  
  @property
  def scaling(self):
    return torch.exp(self.log_scaling)


  def set_scaling(self, scaling) -> 'Gaussians2D':
    return replace(self, log_scaling=torch.log(scaling))



