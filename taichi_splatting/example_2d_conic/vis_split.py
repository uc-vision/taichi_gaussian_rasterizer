

from functools import partial
from taichi_splatting.misc.renderer2d import render_gaussians, split_gaussians2d, uniform_split_gaussians2d
from taichi_splatting.tests.random_data import random_2d_gaussians
import torch
import taichi as ti

import cv2

def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(0)

def main():
    ti.init(ti.gpu)

    torch.cuda.manual_seed(0)
    while True: 
      gaussians = random_2d_gaussians(5, (640, 480), scale_factor=0.2, alpha_range=(1.0, 1.0)).cuda()

      rendering = render_gaussians(gaussians, (640, 480))
      display_image('image', rendering.image)
      splits = uniform_split_gaussians2d(gaussians, 2)

      rendering = render_gaussians(splits, (640, 480))
      display_image('image', rendering.image)

if __name__ == "__main__":
    main()