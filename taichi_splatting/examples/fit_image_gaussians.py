from dataclasses import replace
from functools import partial
from itertools import product
import math
import os
from pathlib import Path
from beartype import beartype
import cv2
import argparse
import numpy as np
import taichi as ti

import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.misc.renderer2d import point_basis, project_gaussians2d, uniform_split_gaussians2d

from taichi_splatting.optim.sparse_adam import SparseAdam
from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.optim.parameter_class import ParameterClass
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from tensordict import TensorDict

from taichi_splatting.torch_lib.util import check_finite
from torch.profiler import profile, record_function, ProfilerActivity

import time
import torch.nn.functional as F
import torchvision as tv
import pdb

from torchviz import make_dot
import datetime

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)

  parser.add_argument('--n', type=int, default=20000)
  parser.add_argument('--iters', type=int, default=2000)

  parser.add_argument('--epoch', type=int, default=20, help='base epoch size (increases with t)')

  parser.add_argument('--opacity_reg', type=float, default=0.0001)
  parser.add_argument('--scale_reg', type=float, default=1.0)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  
  args = parser.parse_args()

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


def cat_values(d1, d2, dim=1):
  assert d1.batch_size == d2.batch_size
  d1 = d1.to_tensordict()
  d2 = d2.to_tensordict()

  return d1.__class__.from_dict({k:torch.cat([d1[k], d2[k]], dim=dim) 
                     for k in d1.keys()}, batch_size=d1.batch_size)

    

def train_epoch(
        optimizer_mlp:torch.nn.Module,
        mlp_opt:torch.optim.Optimizer,

        gaussians:Gaussians2D, 
        
        ref_image, 
        config:RasterConfig,        
        epoch_size=100, 
        
        n=20000,

        opacity_reg=0.0,
        scale_reg=0.0):

  h, w = ref_image.shape[:2]
  def render(gaussians1, i):
    gaussians1.requires_grad_(True)
    gaussians1.z_depth.grad = gaussians1.z_depth.new_zeros(gaussians1.z_depth.shape)

    gaussians2d = project_gaussians2d(gaussians1) 

    raster = rasterize(gaussians2d=gaussians2d, 
      depth=gaussians1.z_depth.clamp(0, 1),
      features=gaussians1.feature, 
      image_size=(w, h), 
      config=config)
    
    scale = torch.exp(gaussians1.log_scaling) / min(w, h)
    # loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
    #         + opacity_reg * gaussians1.opacity.mean()
    #         + scale_reg * scale.pow(2).mean())
    loss = (torch.nn.functional.l1_loss(raster.image, ref_image))
    
    print(f"loss_{i}: ", loss)
    
    # graph = make_dot(loss, params=dict(optimizer_mlp.named_parameters()))
    # graph.render(f"fit_image_computation_graph_{datetime.datetime.now()}", format="png", directory="/local/repo/taichi-splatting/taichi_splatting/examples")
    loss.backward()
    
    output_path_cv2 = f"/local/repo/taichi-splatting/taichi_splatting/examples/{i}.png"
    cv2.imwrite(output_path_cv2, raster.image.detach().cpu().numpy()*255)

    return raster

  with torch.set_grad_enabled(True):
    raster = render(gaussians, -1)
    for i in range(epoch_size):
      print("i: ", i)
    
      mlp_opt.zero_grad()
      check_finite(gaussians, 'gaussians', warn=True)

      detached_gaussian = gaussians.detach()
      detached_gaussian.requires_grad_(True)
      render(detached_gaussian, i)
      gradient = detached_gaussian.grad.detach()
      
      gaussians_grads = cat_values(gaussians.detach(), gradient)
      inputs = torch.cat(list(gaussians_grads.values()), dim=1)
      
      step = optimizer_mlp(inputs)
    
      # split step up to tensordict so that we can add it back to the gaussians
      gaussian_step = Gaussians2D(alpha_logit = step[:, :1],
                            feature = step[:, 1:4],
                            log_scaling = step[:, 4:6],
                            position = step[:, 6:8],
                            rotation = step[:, 8:10],
                            z_depth = step[:, 10:11],
                            batch_size=[n]
                            )
      
      gaussians.alpha_logit = gaussians.alpha_logit.detach() + gaussian_step.alpha_logit
      gaussians.feature = gaussians.feature.detach() + gaussian_step.feature
      gaussians.log_scaling = gaussians.log_scaling.detach() + gaussian_step.log_scaling
      gaussians.position = gaussians.position.detach() + gaussian_step.position
      gaussians.rotation = gaussians.rotation.detach() + gaussian_step.rotation
      gaussians.z_depth = gaussians.z_depth.detach() + gaussian_step.z_depth

      # gaussians.position = torch.stack([
      #     gaussians.position[..., 0].clamp(0, ref_image.shape[0]),
      #     gaussians.position[..., 1].clamp(0, ref_image.shape[1])
      # ], dim=-1)
      
      # render the scene again with the modified gaussians
      raster = render(gaussians, i)

      # step the mlp_opt to modify the model
      mlp_opt.step() 
   
  gaussians.alpha_logit = gaussians.alpha_logit.detach()
  gaussians.feature = gaussians.feature.detach()
  gaussians.log_scaling = gaussians.log_scaling.detach()
  gaussians.position = gaussians.position.detach()
  gaussians.rotation = gaussians.rotation.detach()
  gaussians.z_depth = gaussians.z_depth.detach()
  
  return raster.image






def main():
  torch.set_printoptions(precision=4, sci_mode=False)

  cmd_args = parse_args()
  device = torch.device('cuda:0')

  torch.set_grad_enabled(False)

  
  ref_image = cv2.imread(cmd_args.image_file)
  assert ref_image is not None, f'Could not read {cmd_args.image_file}'

  h, w = ref_image.shape[:2]


  TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=cmd_args.debug, device_memory_GB=0.1)

  print(f'Image size: {w}x{h}')

  if cmd_args.show:
    cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rendered', w, h)


  torch.manual_seed(cmd_args.seed)
  torch.cuda.random.manual_seed(cmd_args.seed)

  gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(torch.device('cuda:0')) 
  channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])


    # Create the MLP
  hidden_channels = [128,256,256,64, channels]  # Hidden layers  # Assuming output is 2D points after transformation

  point_optimizer_mlp = tv.ops.MLP(
        in_channels=channels * 2, 
        hidden_channels=hidden_channels,
        activation_layer=torch.nn.LeakyReLU  # Example activation
    ).to(device=gaussians.position.device)

  mlp_opt = torch.optim.Adam(point_optimizer_mlp.parameters(), lr=0.001)


  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255 
  config = RasterConfig()

  
  epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]

  pbar = tqdm(total=cmd_args.iters)
  iteration = 0
  for  epoch_size in epochs:

    metrics = {}
    image = train_epoch(point_optimizer_mlp, mlp_opt, gaussians, ref_image, 
                                      epoch_size=epoch_size, config=config, 
                                      n = cmd_args.n,
                                      opacity_reg=cmd_args.opacity_reg,
                                      scale_reg=cmd_args.scale_reg)


    if cmd_args.show:
      display_image('rendered', image)

  
    metrics['CPSNR'] = psnr(ref_image, image).item()
    metrics['n'] = gaussians.batch_size[0]


    for k, v in metrics.items():
      if isinstance(v, float):
        metrics[k] = f'{v:.2f}'
      if isinstance(v, int):
        metrics[k] = f'{v:4d}'

    pbar.set_postfix(**metrics)

    iteration += epoch_size
    pbar.update(epoch_size)


      
if __name__ == "__main__":
  main()