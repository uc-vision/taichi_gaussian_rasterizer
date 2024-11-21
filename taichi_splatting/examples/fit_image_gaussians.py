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
import torch.nn as nn
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
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('image_file', type=str)
  parser.add_argument('--seed', type=int, default=0)

  parser.add_argument('--n', type=int, default=20)
  parser.add_argument('--iters', type=int, default=20)

  parser.add_argument('--epoch', type=int, default=100, help='base epoch size (increases with t)')

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
        epoch_size=1000, 

        opacity_reg=0.0,
        scale_reg=0.0):
  scaling_factor = 1
    
  h, w = ref_image.shape[:2]


  with torch.enable_grad():
    def render(gaussians1):
      gaussians1.requires_grad_(True)
      gaussians1.z_depth.grad = gaussians1.z_depth.new_zeros(gaussians1.z_depth.shape)

      gaussians2d = project_gaussians2d(gaussians1) 

      raster = rasterize(gaussians2d=gaussians2d, 
        depth=gaussians1.z_depth.clamp(0, 1),
        features=gaussians1.feature, 
        
        image_size=(w, h), 
        config=config)

      scale = torch.exp(gaussians1.log_scaling) / min(w, h)
      loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
              + opacity_reg * gaussians1.opacity.mean()
              + scale_reg * scale.pow(2).mean())

      loss.backward(retain_graph=True)
      graph = make_dot(loss, params=dict(optimizer_mlp.named_parameters()))
      graph.render(f"fit_image_computation_graph_{time.time()}", format="png", directory="/csse/users/pwl25/ucvision/taichi-splatting/taichi_splatting")

      print(loss)
      
      
      return raster
    raster = render(gaussians)

    for i in range(epoch_size):
      mlp_opt.zero_grad()
      check_finite(gaussians, 'gaussians', warn=True)
      gaussians_grads = cat_values(gaussians, gaussians.grad)
    
      inputs = torch.cat(list(gaussians_grads.values()), dim=1)
      
      step = optimizer_mlp(inputs)
      print(step.requires_grad)
      # step *= scaling_factor 
      # step = torch.tensor(step, dtype=torch.float32)
      # step = step.float()  # Ensure `step` is of type float32 if it's not already

    # Create Gaussians2D object without detaching
      step_gaussians = Gaussians2D(
          alpha_logit=step[:, :1],  # Do not detach
          feature=step[:, 1:4],     # Do not detach
          log_scaling=step[:, 4:6], # Do not detach
          position=step[:, 6:8],    # Do not detach
          rotation=step[:, 8:10],   # Do not detach
          z_depth=step[:, 10:11],   # Do not detach
          batch_size=gaussians.batch_size
      )
      print(gaussians.grad)
      print(step.grad)
      # Add gaussians and step_gaussians, which both track gradients
      
      gaussians = torch.add( gaussians,  step_gaussians)

      # Ensure c has gradients now
      

# Retain gradients for gaussians
      # gaussians.retain_grad()

      # Now you can check gradients
      
      # with torch.no_grad():
      #   gaussians.alpha_logit += step[:, :1]
      #   gaussians.feature += step[:, 1:4]
      #   gaussians.log_scaling += step[:, 4:6]
      #   gaussians.position += step[:, 6:8]
      #   gaussians.rotation += step[:, 8:10]
      #   gaussians.z_depth += step[:, 10:11]
      # gaussians.alpha_logit.requires_grad_(True)
      # gaussians.feature.requires_grad_(True)
      # gaussians.log_scaling.requires_grad_(True)
      # gaussians.position.requires_grad_(True)
      # gaussians.rotation.requires_grad_(True)
      # gaussians.z_depth.requires_grad_(True)
      # print(gaussians.grad)

      # Also, ensure step has requires_grad set to True if you're using it in an optimization loop
      # step.requires_grad_(True)

      # gaussians.alpha_logit = gaussians.alpha_logit + step[:, :1]
      # gaussians.feature = gaussians.feature + step[:, 1:4]
      # gaussians.log_scaling = gaussians.log_scaling + step[:, 4:6]
      # gaussians.position = gaussians.position + step[:, 6:8]
      # gaussians.rotation = gaussians.rotation + step[:, 8:10]
      # gaussians.z_depth = gaussians.z_depth + step[:, 10:11]
    #   gaussian_fields = [
    # ('alpha_logit', 0, 1),
    # ('feature', 1, 4),
    # ('log_scaling', 4, 6),
    # ('position', 6, 8),
    # ('rotation', 8, 10),
    # ('z_depth', 10, 11)
    #   ]

      # Loop through each field and update it accordingly
      # for field, start, end in gaussian_fields:
      #     # Ensure the slice is still connected to the computation graph
      #   gaussians_field = getattr(gaussians, field)
      #   gaussians_field.data = gaussians_field.data + step[:, start:end].detach().clone()
      #   print(gaussians.grad)


      for param in gaussians.values():
        if param.grad is not None:
          param.grad.zero_()
      raster = render(gaussians1)
      

      mlp_loss = torch.nn.functional.mse_loss(raster.image, ref_image)
      mlp_loss.backward()
      mlp_opt.step()
    
      
      # split step up to tensordict so that we can add it back to the gaussians
      # zero the gaussians.grad

      # render the scene again with the modified gaussians

      # step the mlp_opt to modify the model

  
  return raster.image



def mlp(input_size, output_size, hidden_size, num_hidden_layers, activation=nn.ReLU):
  layers = [nn.Linear(input_size, hidden_size), activation()]
  for _ in range(num_hidden_layers):
    layers.append(nn.Linear(hidden_size, hidden_size))
    layers.append(activation())
  layers.append(nn.Linear(hidden_size, output_size))
  return nn.Sequential(*layers)


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
  hidden_channels = [512,256]  # Hidden layers  # Assuming output is 2D points after transformation

  point_optimizer_mlp = mlp(
        input_size=channels * 2, 
        hidden_size= 32,
        output_size= channels,
        num_hidden_layers = 6 
         # Example activation
    ).to(device=gaussians.position.device)
  mlp_opt = torch.optim.Adam(point_optimizer_mlp.parameters(), lr=0.01)
  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255 
  config = RasterConfig()

  
  epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]

  pbar = tqdm(total=cmd_args.iters)
  iteration = 0
  for  epoch_size in epochs:

    metrics = {}
    image = train_epoch(point_optimizer_mlp, mlp_opt, gaussians, ref_image, 
                                      epoch_size=epoch_size, config=config, 
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