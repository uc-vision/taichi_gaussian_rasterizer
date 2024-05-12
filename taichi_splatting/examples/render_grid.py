import argparse
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from taichi_splatting import conic, surfel

from taichi_splatting.camera_params import expand_proj
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.conic.bounds import compute_obb


from taichi_splatting.testing.examples import test_grid
import taichi as ti

from .display import display_image, draw_corners, draw_obb, numpy_image


def find_threshold(alpha, beta):
  x = np.linspace(0, 4, 10000)
  y = np.exp(-(0.5 * x**2) ** beta)
  return x[np.argmax(y < alpha)]

def render_tilemap(tile_ranges, config):
    density = (tile_ranges[:, :, 1] - tile_ranges[:, :, 0]).to(torch.float32)
    density = F.interpolate(density.unsqueeze(0).unsqueeze(0), scale_factor=config.tile_size, mode='nearest').squeeze(0).squeeze(0)
    density = numpy_image(density / density.max())

    density = cv2.cvtColor(density, cv2.COLOR_GRAY2BGR)
    return density


def main():
  torch.set_printoptions(precision=4, sci_mode=False)
  ti.init(arch=ti.cuda)

  device = torch.device('cuda:0')
  torch.set_default_device(device)


  parser = argparse.ArgumentParser()
  parser.add_argument("--n", type=int, default=5)
  parser.add_argument("--scale", type=float, default=1.)
  parser.add_argument("--beta", type=float, default=10.0)
  parser.add_argument("--alpha", type=float, default=0.01)
  parser.add_argument("--surfel", action="store_true")



  args = parser.parse_args()

  config = RasterConfig(beta=args.beta, alpha_threshold=args.alpha)
  gaussians, camera_params = test_grid(args.n, device, scale=args.scale)

  r=find_threshold(config.alpha_threshold, config.beta)

  if not args.surfel:

    render = conic.render_gaussians(gaussians, camera_params, config=config)
    _, tile_ranges = conic.map_to_tiles(render.gaussians_2d, render.point_depth, camera_params.depth_range, camera_params.image_size, config)
    density = render_tilemap(tile_ranges, config)

    obb = compute_obb(render.gaussians_2d, r)    
    draw_obb(density, obb, color=(0, 255, 0))
    display_image("tile_density", density)  

    image = numpy_image(render.image)
    draw_obb(image, obb)

  else:

    surfels = surfel.gaussian3d_to_surfel(gaussians, torch.arange(0, gaussians.batch_size[0], device=device), camera_params.T_camera_world)
    quads, depth = surfel.surfel_bounds(surfels, expand_proj(camera_params.T_image_camera), gaussian_scale=r)


    _, tile_ranges = surfel.map_to_tiles(quads, depth, camera_params.depth_range, 
                                         camera_params.image_size, config)
    

    density = render_tilemap(tile_ranges, config)
    draw_corners(density, quads.reshape(-1, 4, 2))

    display_image("tile_density", density)


    render = surfel.rasterize_surfels(surfels, gaussians.feature, 
                      camera_params.image_size, expand_proj(camera_params.T_image_camera), config)
    
    image = numpy_image(render.image)
    draw_corners(image, quads.reshape(-1, 4, 2))

  display_image("output_image", image)
  


if __name__=="__main__":
   main()