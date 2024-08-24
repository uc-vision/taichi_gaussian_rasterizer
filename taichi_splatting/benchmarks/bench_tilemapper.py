import argparse

import torch
import taichi as ti
from taichi_splatting.benchmarks.util import benchmarked

from taichi_splatting.rasterizer.function import  RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d
from taichi_splatting.tests.random_data import random_2d_gaussians
from taichi_splatting.mapper import tile_mapper


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--scale_factor', type=float, default=2)
  parser.add_argument('--tile_size', type=int, default=16)

  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=1000)
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--depth16', action='store_true')

  args = parser.parse_args(args)
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args


def bench_rasterizer(args):

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
        device_memory_GB=0.1, debug=args.debug)
  
  torch.manual_seed(args.seed)

  depth_range = (0.1, 100.)
  gaussians = random_2d_gaussians(args.n, args.image_size, 
          scale_factor=args.scale_factor, alpha_range=(0.5, 1.0), depth_range=depth_range).to(args.device)
  config = RasterConfig(tile_size=args.tile_size)
  
  gaussians2d = project_gaussians2d(gaussians)

  for k, module in dict(tile_mapper=tile_mapper).items():

    def map_to_tiles():

      return module.map_to_tiles(gaussians2d, 
        depth=gaussians.z_depth, 
        image_size=args.image_size, 
        config=config,
        use_depth16=args.depth16)

    _, tile_ranges = map_to_tiles()

    points_per_tile = (tile_ranges[:, :, 1] - tile_ranges[:, :, 0])
    overlap_ratio = points_per_tile.sum() / args.n 

    print(f'{k}: scale_factor={args.scale_factor}, n={args.n}, tile_size={args.tile_size} point_overlap={overlap_ratio:.2f} tile_points={points_per_tile.float().mean():.2f}')
    benchmarked(k, map_to_tiles, profile=args.profile, iters=args.iters)  

    print('----------------------------------------------------------')    



def main():
  args = parse_args()
  bench_rasterizer(args)

if __name__ == '__main__':
  main()