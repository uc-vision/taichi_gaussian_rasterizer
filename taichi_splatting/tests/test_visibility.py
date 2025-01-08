
from tqdm import tqdm
from taichi_splatting.tests.test_projection import compare
from taichi_splatting.data_types import RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d
from taichi_splatting.rasterizer.function import rasterize
from taichi_splatting.tests.random_data import random_2d_gaussians
from taichi_splatting.taichi_queue import taichi_queue


import torch
import taichi as ti
import cv2
import argparse
import numpy as np

def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    while cv2.waitKey(1) == -1:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--max_n', type=int, default=10000)
    return parser.parse_args()




def test_visibility(debug=False, max_n=10000):
  with taichi_queue(arch=ti.gpu, log_level=ti.INFO if not debug else ti.DEBUG, debug=debug):

    torch.cuda.manual_seed(0)
    torch.manual_seed(0)

    image_size = (320, 200)

    for i in tqdm(range(100)):

      n = np.random.randint(1, max_n)
      gaussians = random_2d_gaussians(n, image_size, scale_factor=0.2, alpha_range=(0.2, 1.0)).cuda()
    
      gaussians = gaussians.to(dtype=torch.float64)
      gaussians.feature.requires_grad_(True)

      config = RasterConfig(compute_visibility=True, compute_point_heuristic=True)

      gaussians2d = project_gaussians2d(gaussians)
      raster = rasterize(gaussians2d=gaussians2d, 
        depth = torch.clamp(gaussians.z_depth, 0, 1).to(torch.float32),
        features=gaussians.feature, 
        image_size=image_size, 
        config=config)

      # gives image gradient == 1.0 (and thus feature gradient == visibility for all channels)
      err = raster.image.sum()
      err.backward()

      visibility_grad = gaussians.feature.grad[:, 0]
      assert compare("visibility", visibility_grad, raster.visibility)


if __name__ == "__main__":
    args = parse_args()
    test_visibility(args.debug, args.max_n)

