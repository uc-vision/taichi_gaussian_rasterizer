from dataclasses import dataclass, replace
from typing import Optional
from beartype.typing import Tuple
from beartype import beartype
from tensordict import tensorclass
import torch


@dataclass 
class Rendering:
  """ Collection of outputs from the renderer, 
  including image map(s) and point statistics for each rendered point.

  depth and depth var are optional, as they are only computed if render_depth=True
  split_heuristics is computed in the backward pass if compute_split_heuristics=True

  radii is computed in the backward pass if compute_radii=True
  """
  image: torch.Tensor        # (H, W, C) - rendered image, C channels of features
  image_weight: torch.Tensor # (H, W, 1) - weight of each pixel (total alpha)

  # Information relevant to points rendered
  points_in_view: torch.Tensor  # (N, 1) - indexes of points in view 
  gaussians_2d: torch.Tensor    # (N, 6)   - 2D gaussians in conic form

  split_heuristics: Optional[torch.Tensor] = None  # (N, 2) - split and prune heuristic
  radii : Optional[torch.Tensor] = None  # (N, 1) - radius of each point

  depth: Optional[torch.Tensor] = None      # (H, W)    - depth map 
  depth_var: Optional[torch.Tensor] = None  # (H, W) - depth variance map

  @property
  def image_size(self) -> Tuple[int, int]:
    h, w, _ = self.image.shape
    return (w, h)

  
@beartype
@dataclass(frozen=True, eq=True)
class RasterConfig:
  tile_size: int = 16

  # pixel tilin per thread in the backwards pass 
  pixel_stride: Tuple[int, int] = (2, 2)
  margin_tiles: int = 3
  
  # cull to an oriented box, otherwise an axis aligned bounding box
  tight_culling: bool = True  

  clamp_max_alpha: float = 0.99
  alpha_threshold: float = 1. / 255.
  saturate_threshold: float = 0.9999

  beta: float = 1.0 # multiplier on gaussian exponent e^-(d ^ (2 * beta))
  depth16: bool = False



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
  
  def requires_grad_(self, requires_grad):
    self.position.requires_grad_(requires_grad)
    self.log_scaling.requires_grad_(requires_grad)
    self.rotation.requires_grad_(requires_grad)
    self.alpha_logit.requires_grad_(requires_grad)
    self.feature.requires_grad_(requires_grad)
    return self
  
  def replace(self, **kwargs):
    return replace(self, **kwargs, batch_size=self.batch_size)

@tensorclass
class Gaussians2D():
  position     : torch.Tensor # 2  - xy
  z_depth        : torch.Tensor # 1  - for sorting
  log_scaling   : torch.Tensor # 2  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 2  - unit length imaginary number
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  feature      : torch.Tensor # N  - (any rgb, label etc)




