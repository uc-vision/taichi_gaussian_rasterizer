

from dataclasses import dataclass
from functools import cached_property
from beartype import beartype
from beartype.typing import Optional, Tuple
import torch

from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.misc.radius import compute_radius
from taichi_splatting.rasterizer import rasterize, RasterConfig
from taichi_splatting.spherical_harmonics import  evaluate_sh_at

from taichi_splatting.perspective import (CameraParams)

from taichi_splatting.perspective.projection import project_to_image
from taichi_splatting.torch_lib.projection import ndc_depth


@dataclass(frozen=True, kw_only=True)
class Rendering:
  """ Collection of outputs from the renderer, 

  depth and depth var are optional, as they are only computed if render_depth=True
  split_heuristics is computed in the backward pass if compute_split_heuristics=True

  radii is computed in the backward pass if compute_radii=True
  """
  image: torch.Tensor        # (H, W, C) - rendered image, C channels of features
  image_weight: torch.Tensor # (H, W, 1) - weight of each pixel (total alpha)

  # Information relevant to points rendered
  points_in_view: torch.Tensor  # (N, 1) - indexes of points in view 
  point_depth: torch.Tensor  # (N, 1) - depth of each point

  split_heuristics: Optional[torch.Tensor] = None  # (N, 2) - split and prune heuristic
  radii : Optional[torch.Tensor] = None  # (N, 1) - radius of each point

  camera : CameraParams

  depth: Optional[torch.Tensor] = None      # (H, W)    - depth map 
  depth_var: Optional[torch.Tensor] = None  # (H, W) - depth variance map

  @cached_property
  def ndc_depth(self) -> torch.Tensor:
    return ndc_depth(self.depth, self.camera.near_plane, self.camera.far_plane)


  @property
  def split_score(self) -> torch.Tensor:
    return self.split_heuristics[:, 0]  
  
  @property
  def prune_cost(self) -> torch.Tensor:
    return self.split_heuristics[:, 1]

  @cached_property
  def visible_mask(self) -> torch.Tensor:
    """ If a point in the view is visible """
    return self.prune_cost > 0

  @cached_property
  def visible_indices(self) -> torch.Tensor:
    """ Indexes of visible points """
    return self.points_in_view[self.visible_mask]


  @property
  def image_size(self) -> Tuple[int, int]:
    return self.camera.image_size
  
  @property
  def num_points(self) -> int:
    return self.points_in_view.shape[0]

@beartype
def render_gaussians(
  gaussians: Gaussians3D,
  camera_params: CameraParams, 
  config:RasterConfig = RasterConfig(),      
  use_sh:bool = False,      
  render_depth:bool = False, 
  compute_radii:bool = False,
  use_depth16:bool = False,
) -> Rendering:
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
  
  Returns:
    images : Rendering - rendered images, with optional depth and depth variance and point weights
    
  """


  gaussians2d, depths, indexes = project_to_image(gaussians, camera_params, config)

  if use_sh:
    features = evaluate_sh_at(gaussians.feature, gaussians.position.detach(), indexes, camera_params.camera_position)
  else:
    features = gaussians.feature
    assert len(features.shape) == 2, f"Features must be (N, C) if use_sh=False, got {features.shape}"


  return render_projected(indexes, gaussians2d, features, depths, camera_params, config, 
                   render_depth=render_depth, use_depth16=use_depth16,  compute_radii=compute_radii)


@torch.compile
def compute_depth_variance(depth_depthsq, weight, eps=1e-6):
    weight_eps = weight + eps

    depth = depth_depthsq[..., 0] / weight_eps
    depth_var = depth_depthsq[..., 1] / weight_eps

    return depth, depth_var - depth**2


def render_projected(indexes:torch.Tensor, gaussians2d:torch.Tensor, 
      features:torch.Tensor, depths:torch.Tensor, 
      camera_params: CameraParams, config:RasterConfig,      

      render_depth:bool = False,  use_depth16:bool = False, use_ndc_depth:bool = False, compute_radii:bool = False):

  ndc_depths = ndc_depth(depths, camera_params.near_plane, camera_params.far_plane)

  if render_depth:
    depths = ndc_depth if use_ndc_depth else depths
    features = torch.cat([depths, depths**2, features], dim=1)

  raster = rasterize(gaussians2d, ndc_depths, features.contiguous(),
    image_size=camera_params.image_size, 
    config=config, 
    use_depth16=use_depth16)

  img_depth, img_depth_var = None, None
  feature_image = raster.image

  if render_depth:
    img_depth, img_depth_var = compute_depth_variance(feature_image[..., :2], raster.image_weight)
    feature_image = feature_image[..., 2:]

  heuristics = raster.point_split_heuristics if config.compute_split_heuristics else None
  radii = compute_radius(gaussians2d, config.gaussian_scale) if compute_radii else None

  return Rendering(image=feature_image, 
                  image_weight=raster.image_weight, 
                  depth=img_depth, 
                  depth_var=img_depth_var, 

                  camera=camera_params,
                    
                  split_heuristics=heuristics,
                  points_in_view=indexes,

                  point_depth=depths,
                  radii=radii)


def viewspace_gradient(gaussians2d: torch.Tensor):
  assert gaussians2d.shape[1] == 6, f"Expected 2D gaussians, got {gaussians2d.shape}"
  assert gaussians2d.grad is not None, "Expected gradients on gaussians2d, run backward first with gaussians2d.retain_grad()"

  xy_grad = gaussians2d.grad[:, :2]
  return torch.norm(xy_grad, dim=1)



