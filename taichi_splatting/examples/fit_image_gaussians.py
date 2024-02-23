import math
from pathlib import Path
import cv2
import argparse
import numpy as np
import taichi as ti

import torch
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.mapper.tile_mapper import map_to_tiles
from taichi_splatting.misc.encode_depth import encode_depth32
from taichi_splatting.misc.renderer2d import project_gaussians2d, split_gaussians2d, uniform_split_gaussians2d

from taichi_splatting.rasterizer.function import rasterize, rasterize_with_tiles

from taichi_splatting.misc.parameter_class import ParameterClass
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_ops.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--tile_size', type=int, default=16)

  parser.add_argument('--n', type=int, default=1000)
  parser.add_argument('--target', type=int, default=None)
  parser.add_argument('--max_epoch', type=int, default=200)
  parser.add_argument('--split_rate', type=float, default=0.1, help='Rate of points to split each epoch (proportional to number of points)')

  parser.add_argument('--write_frames', type=Path, default=None)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--epoch', type=int, default=20, help='Number of iterations per measurement/profiling')
  
  return parser.parse_args()


def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  


def train_epoch(opt, gaussians, ref_image, epoch_size=100, config:RasterConfig = RasterConfig(), grad_alpha=0.9):
    h, w = ref_image.shape[:2]

    contrib = torch.zeros((gaussians.batch_size[0], 2), device=gaussians.position.device)

    for i in range(epoch_size):
      opt.zero_grad()

      gaussians2d = project_gaussians2d(gaussians)  
      depths = encode_depth32(gaussians.depth)


      raster = rasterize(gaussians2d=gaussians2d, 
        encoded_depths=depths,
        features=gaussians.feature, 
        image_size=(w, h), 
        config=config,
        compute_split_heuristics=True)

      loss = torch.nn.functional.l1_loss(raster.image, ref_image) #+ (1e-4 * gaussians.log_scaling).pow(2).sum()
      loss.backward()


      check_finite(gaussians)
      opt.step()

      with torch.no_grad():
        gaussians.log_scaling.clamp_(min=-1, max=4)

        contrib =  raster.point_split_heuristics if i == 0 \
            else (1 - grad_alpha) * contrib + grad_alpha * raster.point_split_heuristics


      visibility, gradient = contrib.unbind(dim=1)
    return raster.image, visibility, gradient 


def main():
  device = torch.device('cuda:0')

  cmd_args = parse_args()
  
  ref_image = cv2.imread(cmd_args.image_file)
  h, w = ref_image.shape[:2]

  ti.init(arch=ti.cuda, log_level=ti.INFO, 
          debug=cmd_args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    cv2.namedWindow('gradient', cv2.WINDOW_NORMAL)
    cv2.namedWindow('err', cv2.WINDOW_NORMAL)
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('gradient', w, h)
    cv2.resizeWindow('err', w, h)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), scale_factor=1.0, beta_range=(1.0, 1.0)).to(torch.device('cuda:0'))
  learning_rates = dict(
    position=0.1,
    log_scaling=0.05,
    rotation=0.005,
    alpha_logit=0.1,
    feature=0.01,
    beta = 0.1
    #beta=0.001
  )
  
  params = ParameterClass.create(gaussians.to_tensordict(), learning_rates, base_lr=1.0)
  keys = set(params.keys())
  trainable = set(params.optimized_keys())

  print(f'attributes - trainable: {trainable} other: {keys - trainable}')

  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
  config = RasterConfig(tile_size=cmd_args.tile_size, gaussian_scale=3.0)

  def timed_epoch(*args, **kwargs):
    start = time.time()
    image, grad, vis = train_epoch(*args, **kwargs)
    torch.cuda.synchronize()
    end = time.time()

    return image, grad, vis, end - start


  train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch
  split_rate = cmd_args.split_rate

  for epoch in range(cmd_args.max_epoch):
    epoch_size = cmd_args.epoch

    image, gradient, visibility, epoch_time = train(params.optimizer, params, ref_image, 
                                        epoch_size=epoch_size, config=config)
    

    with torch.no_grad():

      if cmd_args.show:
        gaussians2d = project_gaussians2d(params)
        depths = encode_depth32(params.depth)
        raster =  rasterize(gaussians2d, depths, gradient.contiguous().unsqueeze(-1), image_size=(w, h), config=config, compute_split_heuristics=True)

        err = torch.abs(ref_image - image)
        
        display_image('gradient', (0.5 * raster.image / raster.image.mean() ))
        display_image('rendered', image)
        display_image('err',  0.25 * err / err.mean(dim=(0, 1), keepdim=True))

    
      if cmd_args.write_frames:
        filename = cmd_args.write_frames / f'{epoch:04d}.png'
        filename.parent.mkdir(exist_ok=True, parents=True)
        print(f'Writing {filename}')
        cv2.imwrite(str(filename), 
                    (image.detach().clamp(0, 1) * 255).cpu().numpy())

      cpsnr = psnr(ref_image, image)
      print(f'{epoch}: {epoch_size / epoch_time:.1f} iters/sec CPSNR {cpsnr:.2f}')

      if cmd_args.target:
        gaussians = Gaussians2D(**params.tensors, batch_size=params.batch_size)

        split_ratio = np.clip(cmd_args.target / gaussians.batch_size[0], 
                              a_min=0.25, a_max=4)
        
        if min(split_ratio, 1/split_ratio) > 0.98:
          factor = math.pow(split_rate, 1/(cmd_args.max_epoch - epoch))

          split_rate *= factor


        grad_thresh = torch.quantile(gradient, 1 - (split_rate * split_ratio))
        vis_thresh = torch.quantile(visibility, split_rate * 1/split_ratio )
    
        prune_mask = (visibility <= vis_thresh)  & (gradient <  grad_thresh)
        split_mask = torch.zeros_like(prune_mask, dtype=torch.bool)

        split_mask = (gradient > grad_thresh) & (visibility > vis_thresh)
        splits = split_gaussians2d(gaussians[split_mask], scaling=0.8)

        params = params[~(split_mask | prune_mask)]
        params = params.append_tensors(splits.to_tensordict())
        # # # params = params[~prune_mask]

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