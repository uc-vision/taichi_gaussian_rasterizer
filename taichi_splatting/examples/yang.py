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

        opacity_reg=0.0,
        scale_reg=0.0):
    
  h, w = ref_image.shape[:2]
  def render(gaussians1):
    gaussians1.requires_grad_(True)
    gaussians1.z_depth.grad = gaussians1.z_depth.new_zeros(gaussians1.z_depth.shape).clone()

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

    print("loss: ", loss)
    

    loss.backward()


    
    # graph = make_dot(loss, params=dict(optimizer_mlp.named_parameters()))
    # graph.render(f"fit_image_computation_graph_{time.time()}", format="png", directory="/local/repo/taichi-splatting/taichi_splatting/examples")
    
    # for name, param in optimizer_mlp.named_parameters():
    #   print(f"{name}: ", param.grad)
  
    stored_gradients['alpha_logit'] = gaussians1.alpha_logit.grad.detach().clone()
    stored_gradients['feature'] = gaussians1.feature.grad.detach().clone()
    stored_gradients['log_scaling'] = gaussians1.log_scaling.grad.detach().clone()
    stored_gradients['position'] = gaussians1.position.grad.detach().clone()
    stored_gradients['rotation'] = gaussians1.rotation.grad.detach().clone()
    stored_gradients['z_depth'] = gaussians1.z_depth.grad.detach().clone()
    for key, _ in gaussians1.items():
      field = getattr(gaussians1, key)
      field.detach()

    return raster



  with torch.set_grad_enabled(True):
    stored_gradients = {}
    raster = render(gaussians)


    for i in range(epoch_size):
      print("i: ", i)
      mlp_opt.zero_grad()
      check_finite(gaussians, 'gaussians', warn=True)
      gradient = Gaussians2D(alpha_logit = stored_gradients['alpha_logit'],
                             feature = stored_gradients['feature'],
                             log_scaling = stored_gradients['log_scaling'],
                             position = stored_gradients['position'],
                             rotation = stored_gradients['rotation'],
                             z_depth = stored_gradients['z_depth'],
                             batch_size=[20000])
      
      gaussians_grads = cat_values(gaussians, gradient)
      inputs = torch.cat(list(gaussians_grads.values()), dim=1)
      step = optimizer_mlp(inputs)
      # step.retain_grad()
      grads = gradient

      # gaussian_step = Gaussians2D(alpha_logit = step[:, :1],
      #                             feature = step[:, 1:4],
      #                             log_scaling = step[:, 4:6],
      #                             position = step[:, 6:8],
      #                             rotation = step[:, 8:10],
      #                             z_depth = step[:, 10:11],
      #                             batch_size=[20000]
      #                             )
     
      # with torch.no_grad():
      # gaussians = gaussians + gaussian_step
        # gaussians += gaussian_step
      
      # gaussians.alpha_logit += gaussian_step.alpha_logit
      # gaussians.feature += gaussian_step.feature
      # gaussians.log_scaling += gaussian_step.log_scaling
      # gaussians.position += gaussian_step.position
      # gaussians.rotation += gaussian_step.rotation
      # gaussians.z_depth += gaussian_step.z_depth
      gaussians = Gaussians2D(alpha_logit = gaussians.alpha_logit.detach() + step[:, :1],
                              feature = gaussians.feature.detach() + step[:, 1:4],
                              log_scaling = gaussians.log_scaling.detach() + step[:, 4:6],
                              position = gaussians.position.detach() + step[:, 6:8],
                              rotation = gaussians.rotation.detach() + step[:, 8:10],
                              z_depth = gaussians.z_depth.detach() + step[:, 10:11],
                              batch_size=[20000])



      # gaussians.alpha_logit.requires_grad_(True)
      # gaussians.feature.requires_grad_(True)
      # gaussians.log_scaling.requires_grad_(True)
      # gaussians.position.requires_grad_(True)
      # gaussians.rotation.requires_grad_(True)





      # for key, grad in grads.items():
      #   field = getattr(gaussians, key)
      #   field.requires_grad_(True)
      #   field.retain_grad()
      #   field.grad = grad.clone()
      
      for key, grad in grads.items():
        field = getattr(gaussians, key)
        field.requires_grad_(True)
        field.retain_grad()
        field.grad = grad.clone()
        # field.grad.zero_()
      
      


   
      # split step up to tensordict so that we can add it back to the gaussians
      # zero the gaussians.grad
      # for param in optimizer_mlp.parameters():
      #   if param.grad is not None:
      #       param.grad.zero_()

      # for key, field in gaussians.items():
      #   if field.grad is not None:
      #       field.grad.zero_()

      # # render the scene again with the modified gaussians
      # gaussians.alpha_logit.retain_grad()
      # gaussians.feature.retain_grad()
      # gaussians.log_scaling.retain_grad()
      # gaussians.position.retain_grad()
      # gaussians.rotation.retain_grad()
      # gaussians.z_depth.retain_grad()
      
      raster = render(gaussians)

      # for key, _ in gaussians.items():
      #   field = getattr(gaussians, key)
      #   field.detach()
      # gaussians = updated_gaussians
      # # # # step the mlp_opt to modify the model

      mlp_opt.step()



    

    # transformed_points = point_optimizer_mlp(points_to_optimize).to(device=params.position.device)
    # points_to_optimize -= transformed_points
    #here will add render and the another optimiser    
  return raster.image






def main():
  torch.autograd.set_detect_anomaly(True)
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
        # in_channels=channels,
        hidden_channels=hidden_channels,
        activation_layer=torch.nn.ReLU  # Example activation
        # activation_layer=torch.nn.Sigmoid
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