from dataclasses import dataclass
from functools import cached_property, partial
import math
from pathlib import Path
from beartype import beartype
import cv2
import argparse
import taichi as ti

from tensordict import tensorclass
import torch
from torch import nn


from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.examples.tiny_nn import tiny_nn
from taichi_splatting.misc.renderer2d import point_basis, project_gaussians2d, uniform_split_gaussians2d

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

  parser.add_argument('--feature_size', type=int, default=16)

  parser.add_argument('--n', type=int, default=1000)
  parser.add_argument('--target', type=int, default=None)
  parser.add_argument('--max_epoch', type=int, default=100)
  parser.add_argument('--prune_rate', type=float, default=0.05, help='Rate of pruning proportional to number of points')
  parser.add_argument('--opacity_reg', type=float, default=0.001)
  parser.add_argument('--scale_reg', type=float, default=0.00001)

  parser.add_argument('--no_antialias', action='store_true')

  parser.add_argument('--noise_scale', type=float, default=0.0)

  parser.add_argument('--write_frames', type=Path, default=None)

  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--profile', action='store_true')
  parser.add_argument('--epoch_size', type=int, default=20, help='Number of iterations per measurement/profiling')
  
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


@tensorclass
class FeatureGaussians2D():
  position     : torch.Tensor # 2  - xy
  z_depth        : torch.Tensor # 1  - for sorting
  feature      : torch.Tensor # N  - (any rgb, label etc)


@dataclass
class Scene:
  
  points : ParameterClass
  model: nn.Module

  model_opt: torch.optim.Optimizer


  @property
  def point_count(self):
    return self.points.batch_size[0]
  
  
  def gaussians(self) -> Gaussians2D:
    out:torch.Tensor = self.model(self.points.feature).to(dtype=torch.float32) # Nx8
    # 2 (log_scaling) + 2 (rotation) + 1 (alpha_logit) + 3 (rgb)
    return Gaussians2D(
      position = self.points.position,
      z_depth = self.points.z_depth,
      log_scaling = out[:, 0:2],
      rotation = out[:, 2:4],
      alpha_logit = out[:, 4],
      feature = out[:, 5:8].sigmoid()
    )
  
  def fit_to(self, target:Gaussians2D, iters:int=100):

    for _ in range(iters):
      gaussians = self.gaussians()
      
      loss = torch.scalar_tensor(0.0, device=self.device)
      for k in ['log_scaling', 'rotation', 'alpha_logit', 'feature']:
        loss += torch.nn.functional.l1_loss(
          getattr(gaussians, k), 
          getattr(target, k))
      
      loss.backward()
      self.step(visible_indexes=torch.arange(self.point_count))
  
  @property
  def device(self):
    return self.points.position.device
  
  def step(self, visible_indexes:torch.Tensor):
    self.points.step(visible_indexes=visible_indexes)
    self.model_opt.step()

    self.points.zero_grad()
    self.model_opt.zero_grad()


def train_epoch(scene:Scene, 
        ref_image, 
        config:RasterConfig,        
        epoch_size=100, 
        grad_alpha=0.9, 
        opacity_reg=0.0,
        scale_reg=0.0):
    
    h, w = ref_image.shape[:2]

    split_heuristics = torch.zeros((scene.point_count, 2), device=scene.device)

    for i in range(epoch_size):
      gaussians:Gaussians2D = scene.gaussians()
      gaussians2d = project_gaussians2d(gaussians)  

      raster = rasterize(gaussians2d=gaussians2d, 
        depth=gaussians.z_depth.clamp(0, 1),
        features=gaussians.feature, 
        image_size=(w, h), 
        config=config)

      loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
        # + opacity_reg * gaussians.alpha.mean()
        + scale_reg * gaussians.scaling.pow(2).mean())
      # loss = (torch.nn.functional.l1_loss(raster.image, ref_image) 
      #         + opacity_reg * gaussians.alpha.mean()
      #         + scale_reg * gaussians.scaling.pow(2).mean())

      loss.backward()

      check_finite(gaussians, 'gaussians', warn=True)


      visible = torch.nonzero(raster.point_split_heuristics[:, 0]).squeeze(1)
      # opt.step()

      basis = point_basis(gaussians)
      scene.points.update_group('position', basis=basis)

      scene.step(visible_indexes = visible)

      with torch.no_grad():

        split_heuristics =  raster.point_split_heuristics if i == 0 \
            else (1 - grad_alpha) * split_heuristics + grad_alpha * raster.point_split_heuristics
        

      prune_cost, densify_score = split_heuristics.unbind(dim=1)
    return raster.image, prune_cost, densify_score 

def make_scene(n:int, w:int, h:int, feature_size:int, device:torch.device, position_lr:float=0.1):
  gaussians = random_2d_gaussians(n, (w, h), alpha_range=(0.5, 1.0), scale_factor=0.1)
  feature_gaussians = FeatureGaussians2D(gaussians.position, 
                                 feature=torch.randn(gaussians.batch_size[0], feature_size, dtype=torch.float32),
                                 z_depth=gaussians.z_depth, 
                                 batch_size=gaussians.batch_size).to(device)
  
  model = tiny_nn(hidden=64, layers=2, num_features=feature_size, output_features=8)
  # 2 (log_scaling) + 2 (rotation) + 1 (alpha_logit) + 3 (rgb)

  model.to(device)
  model_opt = torch.optim.Adam(model.parameters(), lr=0.01)

  parameter_groups = dict(
    position=dict(lr=position_lr, type='adam'),
    feature=dict(lr=0.1),
  ) 

  create_optimizer = partial(SparseAdam, betas=(0.5, 0.999))
  params = ParameterClass(feature_gaussians.to_tensordict(), 
        parameter_groups, optimizer=create_optimizer)

  scene = Scene(params, model, model_opt)
  # scene.fit_to(gaussians.to(device=device))

  return scene

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
  lr_range = (0.1, 0.1)


  
  scene = make_scene(cmd_args.n, w, h, cmd_args.feature_size, device, position_lr=lr_range[0])



  ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
  
  config = RasterConfig(compute_split_heuristics=True,
                        tile_size=cmd_args.tile_size, 
                        gaussian_scale=3.0, 
                        antialias=not cmd_args.no_antialias,
                        pixel_stride=cmd_args.pixel_tile or (2, 2))



  train = with_benchmark(timed_epoch) if cmd_args.profile else timed_epoch

  
  for epoch in range(cmd_args.max_epoch):
    epoch_size = cmd_args.epoch_size
    t = (epoch + 1) / (cmd_args.max_epoch - 1)


    image, densify_score, prune_cost, epoch_time = train(scene, ref_image, config=config, 
                                        epoch_size=epoch_size, 
                                        opacity_reg=cmd_args.opacity_reg,
                                        scale_reg=cmd_args.scale_reg)
    

    with torch.no_grad():

      if cmd_args.show:      
        display_image('rendered', image)

    
      if cmd_args.write_frames:
        filename = cmd_args.write_frames / f'{epoch:04d}.png'
        filename.parent.mkdir(exist_ok=True, parents=True)
        print(f'Writing {filename}')
        cv2.imwrite(str(filename), 
                    (image.detach().clamp(0, 1) * 255).cpu().numpy())

      cpsnr = psnr(ref_image, image)
      print(f'{epoch + 1}: {epoch_size / epoch_time:.1f} iters/sec CPSNR {cpsnr:.2f}')

      # if cmd_args.target and epoch < cmd_args.max_epoch - 1:
      #   gaussians = Gaussians2D(**params.tensors, batch_size=params.batch_size)

      #   t_points = min(math.pow(t * 2, 0.5), 1.0)

      #   n = gaussians.batch_size[0]

      #   split_mask, prune_mask = split_prune(n = n, 
      #               target = math.ceil(cmd_args.n * (1 - t_points) + t_points * cmd_args.target),
      #               n_prune=int(cmd_args.prune_rate * n * (1 - t)**2),
      #               densify_score=densify_score, prune_cost=prune_cost)


      #   splits = uniform_split_gaussians2d(gaussians[split_mask])


      #   params = params[~(split_mask | prune_mask)]
      #   params = params.append_tensors(splits.to_tensordict())

      #   print(f" split {split_mask.sum()}, pruned {prune_mask.sum()} {params.batch_size} points")

        


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