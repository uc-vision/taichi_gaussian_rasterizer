

from dataclasses import dataclass
from typing import Optional, Tuple
from beartype import beartype
import torch

from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.conic.bounds import compute_radius
from taichi_splatting.spherical_harmonics import  evaluate_sh_at

from taichi_splatting.conic.rasterizer import rasterize, RasterConfig
from taichi_splatting.conic.perspective.preprocess import  preprocess_conic
from taichi_splatting.camera_params import CameraParams



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

  point_depth: torch.Tensor = None  # (N, 1) - depth of each point in camera

  split_heuristics: Optional[torch.Tensor] = None  # (N, 2) - split and prune heuristic
  radii : Optional[torch.Tensor] = None  # (N, 1) - radius of each point

  depth: Optional[torch.Tensor] = None      # (H, W)    - depth map 
  depth_var: Optional[torch.Tensor] = None  # (H, W) - depth variance map

  @property
  def image_size(self) -> Tuple[int, int]:
    h, w, _ = self.image.shape
    return (w, h)




@beartype
def render_gaussians(
  gaussians: Gaussians3D,
  camera_params: CameraParams, 
  config:RasterConfig = RasterConfig(),      
  use_sh:bool = False,      

  compute_split_heuristics:bool = False,
  compute_radii:bool = False,
  render_depth:bool = False
): #-> Rendering:
  """
  A complete renderer for 3D gaussians. 
  Parameters:
    packed_gaussians: torch.Tensor (N, 11) - packed 3D gaussians
    features: torch.Tensor (N, C) | torch.Tensor(N, 3, (D+1)**2) 
      features for each gaussian OR spherical harmonics coefficients of degree D
    
    camera_params: CameraParams
    config: RasterConfig
    use_sh: bool - whether to use spherical harmonics
    render_depth: bool - whether to render depth and depth variance
    use_depth16: bool - whether to use 16 bit depth encoding (otherwise 32 bit)
    compute_split_heuristics: bool - whether to compute the visibility for each point in the image
  
  Returns:
    images : Rendering - rendered images, with optional depth and depth variance and point weights
    
  """

  conics = preprocess_conic(gaussians, camera_params, config)
  indexes = torch.nonzero(conics.tile_counts).squeeze(1)

  if use_sh:
    features = evaluate_sh_at(gaussians.feature, gaussians.position.detach(), indexes, camera_params.camera_position)

  assert len(features.shape) == 2, f"Features must be (N, C) if use_sh=False, got {features.shape}"


  # raster = rasterize(gaussians2d, 
  #                    depth=point_depth, depth_range=camera_params.depth_range, 
  #                    tile_counts=tile_counts,
  #                    features=features.contiguous(),
  #   image_size=camera_params.image_size, config=config, compute_split_heuristics=compute_split_heuristics)

  # heuristics = raster.point_split_heuristics if compute_split_heuristics else None
  # radii = compute_radius(gaussians2d) if compute_radii else None

  # return Rendering(image=raster.image, 
  #                 image_weight=raster.image_weight, 
  #                 point_depth=point_depth,
                    
  #                 split_heuristics=heuristics,
  #                 points_in_view=indexes,
  #                 gaussians_2d = gaussians2d,
  #                 radii=radii)

