from dataclasses import dataclass
from beartype.typing import Tuple
from beartype import beartype
from tensordict import tensorclass
import torch
import torch.nn.functional as F

  
@beartype
@dataclass(frozen=True)
class RasterConfig:
  tile_size: int = 16

  # pixel tilin per thread in the backwards pass 
  pixel_stride: Tuple[int, int] = (2, 2)

  margin_tiles: int = 3

  # cutoff N standard deviations from mean
  gaussian_scale: float = 3.0   
  
  # cull to an oriented box, otherwise an axis aligned bounding box
  tight_culling: bool = True  

  clamp_max_alpha: float = 0.99
  alpha_threshold: float = 1. / 255.
  saturate_threshold: float = 0.9999



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
  beta          : torch.Tensor # 1  - beta (1/2 * gaussian exponent)

  feature      : torch.Tensor # (any rgb (3), spherical harmonics (3x16) etc)


  def __post_init__(self):
    assert self.position.shape[1] == 3, f"Expected shape (N, 3), got {self.position.shape}"
    assert self.log_scaling.shape[1] == 3, f"Expected shape (N, 3), got {self.log_scaling.shape}"
    assert self.rotation.shape[1] == 4, f"Expected shape (N, 4), got {self.rotation.shape}"
    assert self.alpha_logit.shape[1] == 1, f"Expected shape (N, 1), got {self.alpha_logit.shape}"
    assert self.beta.shape[1] == 1, f"Expected shape (N, 1), got {self.beta.shape}"


  def packed(self):
    return torch.cat([self.position, self.log_scaling, self.rotation, self.alpha_logit, self.beta], dim=-1)

  @property
  def scale(self):
    return torch.exp(self.log_scaling)

  @property
  def alpha(self):
    return torch.sigmoid(self.alpha_logit)
  

@tensorclass
class Gaussians2D():
  position     : torch.Tensor # 2  - xy
  depth        : torch.Tensor # 1  - for sorting
  log_scaling   : torch.Tensor # 2  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 2  - unit length imaginary number
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  beta          : torch.Tensor # 1  - beta (1/2 * gaussian exponent)

  feature      : torch.Tensor # N  - (any rgb, label etc)





