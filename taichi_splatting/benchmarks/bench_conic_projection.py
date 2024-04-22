import argparse
from functools import partial

import torch
from taichi_splatting.benchmarks.util import benchmarked
from taichi_splatting.perspective import projection
# from taichi_splatting import projection


import taichi as ti
from taichi_splatting.tests.random_data import random_3d_gaussians, random_camera


def parse_args(args=None):
  parser = argparse.ArgumentParser()
  parser.add_argument('--profile', action='store_true')

  parser.add_argument('--image_size', type=str, default='1024,768')
  parser.add_argument('--device', type=str, default='cuda:0')
  parser.add_argument('--n', type=int, default=1000000)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--iters', type=int, default=500)
  

  args = parser.parse_args(args)
  args.image_size = tuple(map(int, args.image_size.split(',')))
  return args


def bench_projection(args):

  ti.init(arch=ti.cuda, offline_cache=True, log_level=ti.DEBUG, 
        device_memory_GB=0.1)
  
     
  torch.manual_seed(args.seed)

  with torch.no_grad():
    camera_params = random_camera()
    gaussians = random_3d_gaussians(args.n, camera_params)
    indexes = torch.arange(args.n, device=args.device)

    gaussians, camera_params = gaussians.to(args.device), camera_params.to(args.device)

    project_forward = partial(projection.project_to_image, gaussians, indexes, camera_params.T_image_camera, camera_params.T_camera_world)
    benchmarked('forward', project_forward, profile=args.profile, iters=args.iters)  


  gaussians.requires_grad_(True)

  def project_backward():
    points, depth = projection.project_to_image(gaussians, indexes, camera_params.T_image_camera, camera_params.T_camera_world)
    loss = points.sum() + depth.sum()
    loss.backward()

  
  benchmarked('backward (gaussians)', project_backward, profile=args.profile, iters=args.iters)  


  gaussians.requires_grad_(False)
  camera_params.T_camera_world.requires_grad_(True)
  benchmarked('backward (extrinsics)', project_backward, profile=args.profile, iters=args.iters)  

  camera_params.T_camera_world.requires_grad_(False)
  camera_params.T_image_camera.requires_grad_(True)
  benchmarked('backward (intrinsics)', project_backward, profile=args.profile, iters=args.iters)  

  gaussians.requires_grad_(True)
  camera_params.T_camera_world.requires_grad_(True)
  camera_params.T_image_camera.requires_grad_(True)
  benchmarked('backward (everything)', project_backward, profile=args.profile, iters=args.iters)  


def main():
  args = parse_args()
  bench_projection(args)

if __name__ == '__main__':
  main()
