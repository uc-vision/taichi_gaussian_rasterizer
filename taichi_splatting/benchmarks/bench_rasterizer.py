import argparse
from dataclasses import replace
from functools import partial

import torch
import taichi as ti
from taichi_splatting.benchmarks.util import benchmarked

from taichi_splatting.rasterizer.function import  rasterize_with_tiles, RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d
from taichi_splatting.taichi_queue import TaichiQueue, taichi_queue
from taichi_splatting.tests.random_data import random_2d_gaussians
from taichi_splatting.mapper.tile_mapper import map_to_tiles


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)

  parser.add_argument('--num_channels', type=int, default=3)
  parser.add_argument('--scale_factor', type=int, default=4)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=1000)
  parser.add_argument('--antialias', action='store_true')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--skip_forward', action='store_true')
  parser.add_argument('--saturate_threshold', type=float, default=0.9999)
  parser.add_argument('--alpha_threshold', type=float, default=1/255)
  parser.add_argument('--pixel_stride', type=str, default='2,2')

  args = parser.parse_args(args)
  args.image_size = tuple(map(int, args.image_size.split(',')))
  args.pixel_stride = tuple(map(int, args.pixel_stride.split(',')))
  return args


def bench_rasterizer(args):
  with taichi_queue(arch=ti.cuda, 
                    log_level=ti.INFO if not args.debug else ti.DEBUG, 
                    debug=args.debug):
    torch.manual_seed(args.seed)

    depth_range = (0.1, 100.)
    gaussians = random_2d_gaussians(args.n, args.image_size, num_channels=args.num_channels,
            scale_factor=args.scale_factor, alpha_range=(0.75, 1.0), depth_range=depth_range).to(args.device)
    config = RasterConfig(tile_size=args.tile_size, antialias=args.antialias, pixel_stride=args.pixel_stride, 
      saturate_threshold=args.saturate_threshold, alpha_threshold=args.alpha_threshold)
    
    gaussians2d = project_gaussians2d(gaussians)

    overlap_to_point, tile_ranges = map_to_tiles(gaussians2d, 
        depth=gaussians.z_depth, 
        image_size=args.image_size, 
        config=config)
    
    points_per_tile = (tile_ranges[:, :, 1] - tile_ranges[:, :, 0])
    overlap_ratio = points_per_tile.sum() / args.n
    print(overlap_to_point.shape) 

    print(f'scale_factor={args.scale_factor}, n={args.n}, tile_size={args.tile_size} point_overlap={overlap_ratio:.2f} tile_points={points_per_tile.float().mean():.2f}')
    print('----------------------------------------------------------')    

    forward = partial(rasterize_with_tiles, gaussians2d=gaussians2d, features=gaussians.feature, 
      tile_overlap_ranges=tile_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
      image_size=args.image_size, config=config)
    
    if not args.skip_forward:
      benchmarked('forward', forward, profile=args.profile, iters=args.iters * 4)  

      forward_vis = partial(rasterize_with_tiles, gaussians2d=gaussians2d, features=gaussians.feature, 
        tile_overlap_ranges=tile_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
        image_size=args.image_size, config=replace(config, compute_visibility=True))
      
      benchmarked('forward_vis', forward_vis, profile=args.profile, iters=args.iters * 4)  

    gaussians.feature.requires_grad_(True)
    
    def backward():
      raster = forward()
      raster.image.sum().backward()

    benchmarked('backward (features)', backward, profile=args.profile, iters=args.iters)  

    gaussians.feature.requires_grad_(False)
    gaussians2d.requires_grad_(True)

    benchmarked('backward (gaussians)', backward, profile=args.profile, iters=args.iters)  

    gaussians.feature.requires_grad_(True)
    gaussians2d.requires_grad_(True)

    benchmarked('backward (all)', backward, profile=args.profile, iters=args.iters)  

    
    def compute_point_heuristic():
      raster = rasterize_with_tiles(gaussians2d=gaussians2d, features=gaussians.feature, 
        tile_overlap_ranges=tile_ranges.view(-1, 2), overlap_to_point=overlap_to_point,
        image_size=args.image_size, config=replace(config, compute_point_heuristic=True))
      
      raster.image.sum().backward()

    benchmarked('backward (compute_point_heuristic)', compute_point_heuristic, profile=args.profile, iters=args.iters)  


def main():
  args = parse_args()
  bench_rasterizer(args)

if __name__ == '__main__':
  main()