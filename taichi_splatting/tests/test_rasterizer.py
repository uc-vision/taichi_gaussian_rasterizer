

import argparse

from tqdm import tqdm
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.misc.renderer2d import  project_gaussians2d
from taichi_splatting.rasterizer.function import rasterize_with_tiles
from taichi_splatting.tests.random_data import random_2d_gaussians
from taichi_splatting.taichi_queue import  taichi_queue

import numpy as np

import torch
import taichi as ti

import cv2

def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    if image.shape[-1] == 2:
      image = np.concatenate([image, np.zeros_like(image[..., :1])], axis=-1)

    cv2.imshow(name, image)
    cv2.waitKey(1)


def make_inputs(config, seed, device=torch.device('cuda:0')):
    torch.random.manual_seed(seed)

    n = torch.randint(1, 50, (1,)).item()
    channels = torch.randint(1, 4, (1,)).item()

    image_size = (8, 8)

    gaussians = random_2d_gaussians(n, image_size, num_channels=channels, scale_factor=1.0, alpha_range=(0.2, 0.8)).to(device=device)  
    gaussians2d = project_gaussians2d(gaussians)
    
    overlap_to_point = torch.arange(0, n, device=device, dtype=torch.int32)
    tile_ranges = torch.tensor([[0, n]], device=device, dtype=torch.int32)

    gaussians2d, colors = [x.to(dtype=torch.float64) for x in [gaussians2d, gaussians.feature]]

    def rasterize(mean, axis, sigma, alpha, colors):
      packed = torch.cat([mean, axis, sigma, alpha], dim=-1)

      render = rasterize_with_tiles(packed, colors, overlap_to_point=overlap_to_point, tile_overlap_ranges=tile_ranges.view(-1, 2), 
                     image_size=image_size, config=config)
      
      return render.image
    
    return (gaussians2d[:, 0:2].requires_grad_(True), 
            gaussians2d[:, 2:4].requires_grad_(True), 
            gaussians2d[:, 4:6].requires_grad_(True), 

            gaussians2d[:, 6:7].requires_grad_(True), 
            colors.requires_grad_(True)), rasterize


def run_rasterizer_gradcheck(name, config, show = False, iters = 100, device=torch.device('cuda:0'), debug=False):
  with taichi_queue(arch=ti.cuda, log_level=ti.INFO if not debug else ti.DEBUG, debug=debug):
    torch.random.manual_seed(0)
    seeds = torch.randint(0, 1000, (iters, ), device=device)

    if show:
      cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    pbar = tqdm(total=len(seeds), desc=name)
    for seed in seeds:
        inputs, render = make_inputs(config, seed)
        torch.autograd.gradcheck(render, inputs, eps=1e-6, check_grad_dtypes=True, check_undefined_grad=True)

        if show:
          image = render(*inputs)
          display_image("image", image)
        
        pbar.update(1)
        pbar.set_postfix(seed=seed.item())



def test_antialias(show=False, debug=False):
  config = RasterConfig(tile_size=8, pixel_stride=(1, 1), antialias=True, use_alpha_blending=True)
  run_rasterizer_gradcheck("antialias", config, show, debug=debug)

def test_no_antialias(show=False, debug=False):
  config = RasterConfig(tile_size=8, pixel_stride=(1, 1), antialias=False, use_alpha_blending=True)
  run_rasterizer_gradcheck("no_antialias", config, show, debug=debug)

# def test_no_blending(show=False, debug=False):
#   config = RasterConfig(tile_size=8, pixel_stride=(1, 1), antialias=False, use_alpha_blending=False)
#   run_rasterizer_gradcheck("no_blending", config, show, debug=debug)

def main(show=False, debug=False):
  torch.set_printoptions(precision=10, sci_mode=False)

  test_antialias(show, debug)
  test_no_antialias(show, debug)
  # test_no_blending(show, debug)- fix grad


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--show", action="store_true")
  parser.add_argument("--debug", action="store_true")
  args = parser.parse_args()

  main(args.show, args.debug)