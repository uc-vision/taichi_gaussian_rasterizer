from functools import partial
import math
from typing import Dict, List
import cv2
import argparse
import numpy as np
import torch.nn as nn
import taichi as ti

import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.examples.mlp import mlp
from taichi_splatting.misc.renderer2d import project_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite

torch.set_float32_matmul_precision('high')

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)

  parser.add_argument('--n', type=int, default=20)
  parser.add_argument('--iters', type=int, default=20)

  parser.add_argument('--epoch', type=int, default=20, help='base epoch size (increases with t)')

  parser.add_argument('--opacity_reg', type=float, default=0.0000)
  parser.add_argument('--scale_reg', type=float, default=10.0)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  
  args = parser.parse_args()
  return args

def initialize_gaussians(batch_size, image_size, n_gaussians, device):
    """Initialize random Gaussians for a batch."""
    gaussians = []
    for _ in range(batch_size):
        g = random_2d_gaussians(n_gaussians, image_size, alpha_range=(0.5, 1.0), scale_factor=1.0)
        gaussians.append(g.to(device))
    return gaussians

def log_lerp(t, a, b):
  return math.exp(math.log(b) * t + math.log(a) * (1 - t))

def lerp(t, a, b):
  return b * t + a * (1 - t)


def display_image(name, image):
    image = (image.clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(1)
    

def psnr(a, b):
  return 10 * torch.log10(1 / torch.nn.functional.mse_loss(a, b))  


def cat_values(dicts, dim=1):
  assert all(d.batch_size == dicts[0].batch_size for d in dicts)
  
  dicts = [d.to_tensordict() for d in dicts]
  assert all(d.keys() == dicts[0].keys() for d in dicts)

  keys, cls = dicts[0].keys(), type(dicts[0])
  concatenated = {k: torch.cat([d[k] for d in dicts], dim=dim) for k in keys}
  return cls.from_dict(concatenated, batch_size=dicts[0].batch_size)

def element_sizes(t):
  """ Get non batch sizes from a tensorclass"""
  return {k:v.shape[1:] for k, v in t.items()}


def split_tensorclass(t, flat_tensor:torch.Tensor):
  sizes = element_sizes(t)
  splits = [np.prod(s) for s in sizes.values()]

  tensors = torch.split(flat_tensor, splits, dim=1)
  
  return t.__class__.from_dict(
      {k: v.view(t.batch_size + s) 
       for k, v, s in zip(sizes.keys(), tensors, sizes.values())},
      batch_size=t.batch_size
  )

def flatten_tensorclass(t):
  flat_tensor = torch.cat([v.view(v.shape[0], -1) for v in t.values()], dim=1)
  return flat_tensor


def mean_dicts(dicts:List[Dict[str, float]]):
  return {k: sum(d[k] for d in dicts) / len(dicts) for k in dicts[0].keys()}

class Trainer:
  def __init__(self, 
               optimizer_mlp:torch.nn.Module, 
               mlp_opt:torch.optim.Optimizer, 
               ref_image: List[torch.Tensor],
               config:RasterConfig, 
               opacity_reg=0.0, 
               scale_reg=0.0):
    
    self.optimizer_mlp = optimizer_mlp
    self.mlp_opt = mlp_opt

    self.config = config
    self.opacity_reg = opacity_reg
    self.scale_reg = scale_reg

    self.ref_image: List[torch.Tensor] = ref_image
    self.running_scales = None


  def render(self, gaussians_batch):
    h, w = self.ref_images.shape[1:3]
    rendered_batch = []
    for gaussians in gaussians_batch:
        gaussians2d = project_gaussians2d(gaussians)
        raster = rasterize(
            gaussians2d=gaussians2d,
            depth=gaussians.z_depth.clamp(0, 1),
            features=gaussians.feature,
            image_size=(w, h),
            config=self.config
        )
        rendered_batch.append(raster.image)
    return torch.stack(rendered_batch)


  def render_step(self, gaussians_batch):
    """Compute loss for a batch."""
    rendered_images = self.render(gaussians_batch)  # Render all images in batch
    batch_loss = 0
    for rendered_image, ref_image in zip(rendered_images, self.ref_images):
        l1 = torch.nn.functional.l1_loss(rendered_image, ref_image)
        batch_loss += l1
    batch_loss /= len(gaussians_batch)  # Average loss over the batch
    return batch_loss


  def get_gradients(self, gaussians):
    gaussians = gaussians.clone()
    gaussians.requires_grad_(True)
    self.render_step(gaussians)
    grad =  gaussians.grad
    

    mean_abs_grad = grad.abs().mean(dim=0)
    if self.running_scales is None:
      self.running_scales = mean_abs_grad
    else:
      self.running_scales = lerp(0.999, self.running_scales, mean_abs_grad)


    return grad * 1e7

  def test(self, gaussians):
      
    """Run inference using the trained model."""
    with torch.enable_grad():
            
      grad = self.get_gradients(gaussians)
      check_finite(grad, "grad")

      inputs = flatten_tensorclass(grad)

    with torch.no_grad():
        step = self.optimizer_mlp(inputs)
        step = split_tensorclass(gaussians, step)
    raster = self.render(gaussians-step)
    psnr_value = psnr(self.ref_image, raster.image).item()
    print(f"Test PSNR: {psnr_value:.4f}")
    return raster.image
        
  # def test(self, gaussians,step_size=0.01,epoch_size=100):
  #   for i in range(epoch_size):
  #     grad = self.get_gradients(gaussians)
  #     check_finite(grad, "grad")
  #     # self.mlp_opt.zero_grad()

  #     inputs = flatten_tensorclass(gaussians)

  #     with torch.enable_grad():
  #       step = self.optimizer_mlp(inputs)
  #       step = split_tensorclass(gaussians, step)
  #     # self.mlp_opt.step()
  #     gaussians = gaussians - step * step_size
  #   return gaussians

  def train_epoch(self, gaussians_batch, step_size=0.01, epoch_size=100):
    metrics = []
    for i in range(epoch_size):
      self.mlp_opt.zero_grad()
      loss = 0
      step_batch= []
      metrics_batch = []
      for gaussians in gaussians_batch:
        grad = self.get_gradients(gaussians)
        check_finite(grad, "grad")
        inputs = flatten_tensorclass(grad)

        with torch.enable_grad():
          step = self.optimizer_mlp(inputs)
          step = split_tensorclass(gaussians, step)
          metrics_batch.append(self.render_step(gaussians - step))
        step_batch.append(step* step_size)
      metrics.append(mean_dicts(metrics_batch))

      self.mlp_opt.step()
      gaussians_batch = gaussians_batch - step_batch  

    return gaussians_batch, mean_dicts(metrics)




def main():
  torch.set_printoptions(precision=4, sci_mode=True)

  cmd_args = parse_args()
  device = torch.device('cuda:0')

  torch.set_grad_enabled(False)

  
  ref_image = cv2.imread(cmd_args.image_file)
  assert ref_image is not None, f'Could not read {cmd_args.image_file}'

  h, w = ref_image.shape[:2]


  TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=cmd_args.debug, device_memory_GB=0.1)


  if cmd_args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)
  torch.cuda.random.manual_seed(cmd_args.seed)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(torch.device('cuda:0')) 
  channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])


  # Create the MLP
  optimizer = mlp(inputs = channels, outputs=channels, 
              hidden_channels=[128, 256, 512,1024], 
              activation=nn.ReLU,
              norm=partial(nn.LayerNorm, elementwise_affine=False),
              # output_activation=nn.LazyLinear,
              output_scale=1e-12
              )
  optimizer.to(device=device)


  optimizer = torch.compile(optimizer)
  optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.0001)



  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255 
  config = RasterConfig()

  trainer = Trainer(optimizer, optimizer_opt, ref_image, config, 
                    opacity_reg=cmd_args.opacity_reg, scale_reg=cmd_args.scale_reg)
  
  epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]

  pbar = tqdm(total=cmd_args.iters)
  iteration = 0
  for  epoch_size in epochs:

    metrics = {}

    # Set warmup schedule for first iterations - log interpolate 
    step_size = log_lerp(min(iteration / 1000., 1.0), 0.1, 1.0)
    
    gaussians, train_metrics = trainer.train_epoch(gaussians, epoch_size=epoch_size, step_size=step_size)

    image = trainer.render(gaussians).image
    if cmd_args.show:
      display_image('rendered', image)

  
    metrics['CPSNR'] = psnr(ref_image, image).item()
    metrics['n'] = gaussians.batch_size[0]
    metrics.update(train_metrics)
    # for key, value in metrics.items():
    #   print(f"{key}: {value}")


    for k, v in metrics.items():
      if isinstance(v, float):
        metrics[k] = f'{v:.4f}'
      if isinstance(v, int):
        metrics[k] = f'{v:4d}'

    pbar.set_postfix(**metrics)

    iteration += epoch_size
    pbar.update(epoch_size)


      
if __name__ == "__main__":
  main()