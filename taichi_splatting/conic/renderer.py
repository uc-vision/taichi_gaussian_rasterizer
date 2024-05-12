

from dataclasses import dataclass
from typing import Optional, Tuple
from beartype import beartype
import torch

from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.conic.bounds import compute_radius
from taichi_splatting.spherical_harmonics import  evaluate_sh_at

from taichi_splatting.conic.rasterizer import rasterize, RasterConfig
from taichi_splatting.conic.perspective import (project_to_conic, CameraParams)

from taichi_splatting.culling import (frustum_culling)


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
  render_depth:bool = False, 

  compute_split_heuristics:bool = False,
  compute_radii:bool = False
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
    compute_split_heuristics: bool - whether to compute the visibility for each point in the image
  
  Returns:
    images : Rendering - rendered images, with optional depth and depth variance and point weights
    
  """


  indexes = gaussians_in_view(gaussians.position, camera_params, config.tile_size, config.margin_tiles)
  # view_gaussians = gaussians[indexes] 

  if use_sh:
    features = evaluate_sh_at(gaussians.feature, gaussians.position.detach(), indexes, camera_params.camera_position)
  else:
    features = gaussians.feature[indexes]
    assert len(features.shape) == 2, f"Features must be (N, C) if use_sh=False, got {features.shape}"

  gaussians2d, depth = project_to_conic(gaussians, indexes, camera_params)
  return render_projected(indexes, gaussians2d, features, depth, camera_params, config, 
                   render_depth=render_depth, 
                   compute_split_heuristics=compute_split_heuristics, compute_radii=compute_radii)


def render_projected(indexes:torch.Tensor, gaussians2d:torch.Tensor, 
      features:torch.Tensor, point_depth:torch.Tensor, 
      camera_params: CameraParams, config:RasterConfig,      

      render_depth:bool = False, 
      compute_split_heuristics:bool = False, compute_radii:bool = False):


  far, near = camera_params.far_plane, camera_params.near_plane

  if render_depth:
    ndc_depth =  (0.5 + far + near - (2.0 * near * far) / point_depth) / (2.0 * (far - near))
    
    ndc_depth = ndc_depth.unsqueeze(-1)
    features = torch.cat([ndc_depth, ndc_depth.pow(2), features], dim=1)

  raster = rasterize(gaussians2d, point_depth, depth_range=(camera_params.near_plane, camera_params.far_plane), features=features.contiguous(),
    image_size=camera_params.image_size, config=config, compute_split_heuristics=compute_split_heuristics)

  depth, depth_var = None, None
  feature_image = raster.image

  if render_depth:
    depth_feat = raster.image[..., :2] / (raster.image_weight.unsqueeze(-1) + 1e-6)

    depth, depth_sq = depth_feat.unbind(dim=-1)
    depth_var = (depth_sq  - depth.pow(2)) 

    feature_image = feature_image[..., 2:]

  heuristics = raster.point_split_heuristics if compute_split_heuristics else None
  radii = compute_radius(gaussians2d) if compute_radii else None

  return Rendering(image=feature_image, 
                  image_weight=raster.image_weight, 
                  point_depth=point_depth,

                  depth=depth, 
                  depth_var=depth_var, 
                    
                  split_heuristics=heuristics,
                  points_in_view=indexes,
                  gaussians_2d = gaussians2d,
                  radii=radii)


def viewspace_gradient(gaussians2d: torch.Tensor):
  assert gaussians2d.shape[1] == 6, f"Expected 2D gaussians, got {gaussians2d.shape}"
  assert gaussians2d.grad is not None, "Expected gradients on gaussians2d, run backward first with gaussians2d.retain_grad()"

  xy_grad = gaussians2d.grad[:, :2]
  return torch.norm(xy_grad, dim=1)



def gaussians_in_view(
  positions: torch.Tensor, 
  camera_params: CameraParams,
  tile_size: int = 16,
  margin_tiles: int = 3
):
  point_mask = frustum_culling(positions,
    camera_params=camera_params, margin_pixels=margin_tiles * tile_size
  )

  return torch.nonzero(point_mask, as_tuple=True)[0]
