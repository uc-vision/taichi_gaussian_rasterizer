import argparse
import numpy as np
import torch

from taichi_splatting.camera_params import expand_proj
from taichi_splatting.conic.bounds import compute_obb
from taichi_splatting.conic.renderer import render_gaussians
from taichi_splatting.data_types import RasterConfig

from taichi_splatting.surfel import gaussian3d_to_surfel, surfel_bounds, rasterize_surfels


from taichi_splatting.testing.examples import test_grid
import taichi as ti

from .display import display_image, draw_corners, draw_obb, numpy_image




def find_threshold(alpha, beta):
  x = np.linspace(0, 4, 10000)
  y = np.exp(-(0.5 * x**2) ** beta)
  return x[np.argmax(y < alpha)]



def main():
  torch.set_printoptions(precision=4, sci_mode=False)
  ti.init(arch=ti.cuda)

  device = torch.device('cuda:0')
  torch.set_default_device(device)


  parser = argparse.ArgumentParser()
  parser.add_argument("--n", type=int, default=5)
  parser.add_argument("--beta", type=float, default=10.0)
  parser.add_argument("--alpha", type=float, default=0.01)
  parser.add_argument("--surfel", action="store_true")

  args = parser.parse_args()

  config = RasterConfig(beta=args.beta, alpha_threshold=args.alpha)
  gaussians, camera_params = test_grid(5, device)

  r=find_threshold(config.alpha_threshold, config.beta)

  if not args.surfel:

    render = render_gaussians(gaussians, camera_params, config=config, compute_radii=True)
    image = numpy_image(render.image)
    obb = compute_obb(render.gaussians_2d, r)
    
    draw_obb(image, obb)
  else:

    surfel = gaussian3d_to_surfel(gaussians, torch.arange(0, gaussians.batch_size[0], device=device), camera_params.T_camera_world)
    bounds, depth = surfel_bounds(surfel, expand_proj(camera_params.T_image_camera), gaussian_scale=r)

    render = rasterize_surfels(surfel, gaussians.feature, 
                      camera_params.image_size, expand_proj(camera_params.T_image_camera), config)
    
    image = numpy_image(render.image)
    draw_corners(image, bounds.reshape(-1, 4, 2))

  display_image("output_image", image)
  


if __name__=="__main__":
   main()