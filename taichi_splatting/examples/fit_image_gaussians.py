from functools import partial
import math
from pathlib import Path
from beartype import beartype
import cv2
import argparse
import numpy as np
import taichi as ti

import torch
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d, uniform_split_gaussians2d

from taichi_splatting.optim.sparse_adam import SparseAdam
from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)
  parser.add_argument('--pixel_tile', type=str, help='Pixel tile for backward pass default "2,2"')

  parser.add_argument('--n', type=int, default=1000)
  parser.add_argument('--target', type=int, default=None)
  parser.add_argument('--max_epoch', type=int, default=200)
  parser.add_argument('--prune_rate', type=float, default=0.1, help='Rate of pruning proportional to number of points')
  parser.add_argument('--opacity_reg', type=float, default=0.001)
  parser.add_argument('--scale_reg', type=float, default=0.00001)

  parser.add_argument('--no_antialias', action='store_true')

  parser.add_argument('--noise_scale', type=float, default=0.0)

  parser.add_argument('--write_frames', type=Path, default=None)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--epoch_size', type=int, default=10, help='Number of iterations per measurement/profiling')
  
  args = parser.parse_args()

  if args.pixel_tile:
    args.pixel_tile = tuple(map(int, args.pixel_tile.split(',')))

  return args


def log_lerp(t, a, b):
  return math.exp(math.log(b) * t + math.log(a) * (1 - t))

def lerp(t, a, b):
  return b * t + a * (1 - t)


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  

def train_epoch(opt, gaussians, ref_image, 
        config:RasterConfig,        
        epoch_size=100, 
        grad_alpha=0.9, 
        opacity_reg=0.0,
        scale_reg=0.0,
        noise_threshold=0.05,
        noise_lr=0.0, 
        k = 100):
    
    h, w = ref_image.shape[:2]

    split_heuristics = torch.zeros((gaussians.batch_size[0], 2), device=gaussians.position.device)

    for i in range(epoch_size):
      opt.zero_grad()

      gaussians2d = project_gaussians2d(gaussians)  
      opacity = torch.sigmoid(gaussians.alpha_logit).unsqueeze(-1)


      raster = rasterize(gaussians2d=gaussians2d, 
        depth=gaussians.z_depth.clamp(0, 1),
        features=gaussians.feature, 
        image_size=(w, h), 
        config=config)


      scale = torch.exp(gaussians.log_scaling)
      
      loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
              + opacity_reg * opacity.mean()
              + scale_reg * scale.pow(2).mean())

      loss.backward()

      check_finite(gaussians, 'gaussians', warn=True)


      visible = torch.nonzero(raster.point_split_heuristics[:, 0]).squeeze(1)
      # opt.step()
      opt.step(visible_indexes = visible)

      with torch.no_grad():
        gaussians.log_scaling.clamp_max(5)

        split_heuristics =  raster.point_split_heuristics if i == 0 \
            else (1 - grad_alpha) * split_heuristics + grad_alpha * raster.point_split_heuristics
        
        opacity = torch.sigmoid(gaussians.alpha_logit)
        op_factor = torch.sigmoid(k * (noise_threshold - opacity)).unsqueeze(1)

        # noise = sample_gaussians(gaussians) * op_factor * noise_lr
        noise = torch.randn_like(gaussians.position) * op_factor * noise_lr
        gaussians.position += noise

      prune_cost, densify_score = split_heuristics.unbind(dim=1)
    return raster.image, prune_cost, densify_score 


def main():
  device = torch.device('cuda:0')

  cmd_args = parse_args()
  
  ref_image = cv2.imread(cmd_args.image_file)
  assert ref_image is not None, f'Could not read {cmd_args.image_file}'

  h, w = ref_image.shape[:2]

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
          debug=cmd_args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    # cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('err', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)

    # cv2.resizeWindow('gradient', w, h)
    # cv2.resizeWindow('err', w, h)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)
  lr_range = (0.5, 0.1)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(1.0, 1.0), scale_factor=0.1).to(torch.device('cuda:0'))
  
  parameter_groups = dict(
    position=dict(lr=lr_range[0]),
    log_scaling=dict(lr=0.025),
    rotation=dict(lr=0.005),
    # alpha_logit=dict(lr=0.05),
    feature=dict(lr=0.01)
  )

  create_optimizer = partial(SparseAdam, betas=(0.7, 0.999))
  # create_optimizer = partial(optim.Adam, foreach=True, betas=(0.7, 0.999), amsgrad=True, weight_decay=0.0)
  # create_optimizer = partial(AdamWScheduleFree, betas=(0.7, 0.999), weight_decay=0.0, warmup_steps=1000)



  params = ParameterClass(gaussians.to_tensordict(), 
        parameter_groups, optimizer=create_optimizer)
  
  keys = set(params.keys())
  trainable = set(params.optimized_keys())

  print(f'attributes - trainable: {trainable} other: {keys - trainable}')

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
  
  config = RasterConfig(compute_split_heuristics=True,
                        tile_size=cmd_args.tile_size, 
                        gaussian_scale=3.0, 
                        antialias=not cmd_args.no_antialias,
                        pixel_stride=cmd_args.pixel_tile or (2, 2))

  @beartype
  def take_n(t:torch.Tensor, n:int, descending=False):
    """ Return mask of n largest or smallest values in a tensor."""
    idx = torch.argsort(t, descending=descending)[:n]

    # convert to mask
    mask = torch.zeros_like(t, dtype=torch.bool)
    mask[idx] = True

    return mask
    
  def split_prune(n, target, n_prune, densify_score, prune_cost):
      prune_mask = take_n(prune_cost, n_prune, descending=False)

      target_split = ((target - n) + n_prune) 
      split_mask = take_n(densify_score, target_split, descending=True)

      both = (split_mask & prune_mask)
      return split_mask ^ both, prune_mask ^ both




  def timed_epoch(*args, **kwargs):
    start = time.time()
    image, grad, vis = train_epoch(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    return image, grad, vis, end - start


  train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch

  
  for epoch in range(cmd_args.max_epoch):
    epoch_size = cmd_args.epoch_size
    t = (epoch + 1) / (cmd_args.max_epoch - 1)


    params.set_learning_rate(position = log_lerp(t, *lr_range))


    image, densify_score, prune_cost, epoch_time = train(params.optimizer, params, ref_image, 
                                        epoch_size=epoch_size, config=config, 
                                        opacity_reg=cmd_args.opacity_reg,
                                        scale_reg=cmd_args.scale_reg,
                                        noise_lr=cmd_args.noise_scale * (1 - t)**2)
    

    with torch.no_grad():

      if cmd_args.show:
        # gaussians2d = project_gaussians2d(params)
        # depths = encode_depth32(params.z_depth)
        # raster =  rasterize(gaussians2d, depths, densify_score.contiguous().unsqueeze(-1), 
        #                     image_size=(w, h), config=config, compute_split_heuristics=True)

        err = torch.abs(ref_image - image)
        
        display_image('rendered', image)
        # display_image('err',  0.25 * err / err.mean(dim=(0, 1), keepdim=True))

    
      if cmd_args.write_frames:
        filename = cmd_args.write_frames / f'{epoch:04d}.png'
        filename.parent.mkdir(exist_ok=True, parents=True)
        print(f'Writing {filename}')
        cv2.imwrite(str(filename), 
                    (image.detach().clamp(0, 1) * 255).cpu().numpy())

      cpsnr = psnr(ref_image, image)
      print(f'{epoch + 1}: {epoch_size / epoch_time:.1f} iters/sec CPSNR {cpsnr:.2f}')

      if cmd_args.target and epoch < cmd_args.max_epoch - 1:
        gaussians = Gaussians2D(**params.tensors, batch_size=params.batch_size)

        t_points = min(math.pow(t * 2, 0.5), 1.0)

        n = gaussians.batch_size[0]

        split_mask, prune_mask = split_prune(n = n, 
                    target = math.ceil(cmd_args.n * (1 - t_points) + t_points * cmd_args.target),
                    n_prune=int(cmd_args.prune_rate * n * (1 - t)**2),
                    densify_score=densify_score, prune_cost=prune_cost)


        splits = uniform_split_gaussians2d(gaussians[split_mask], noise=0.0)


        params = params[~(split_mask | prune_mask)]
        params = params.append_tensors(splits.to_tensordict())

        print(f" split {split_mask.sum()}, pruned {prune_mask.sum()} {params.batch_size} points")

        


def with_benchmark(f):
  def g(*args , **kwargs):
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
      with record_function("model_inference"):
        result = f(*args, **kwargs)
        torch.cuda.synchronize()

      prof_table = prof.key_averages().table(sort_by="self_cuda_time_total", 
                                          row_limit=25, max_name_column_width=100)
      print(prof_table)
      return result
  return g

  

if __name__ == '__main__':
  main()