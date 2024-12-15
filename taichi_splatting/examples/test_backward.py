

from functools import partial
from taichi_splatting.misc.renderer2d import render_gaussians, split_gaussians2d, uniform_split_gaussians2d
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
import torch
import taichi as ti

import cv2

def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    while cv2.waitKey(1) == -1:
        pass

def main():
    TaichiQueue.init(ti.gpu)

    torch.manual_seed(0)
    gaussians = random_2d_gaussians(5, (640, 480), scale_factor=10.0, alpha_range=(0.2, 0.2)).cuda()

    gaussians.requires_grad_(True)
    rendering = render_gaussians(gaussians, (640, 480))
    rendering.image.sum().backward()


      # splits = uniform_split_gaussians2d(gaussians, 2, random_axis=True)

      # rendering = render_gaussians(splits, (640, 480))
      # display_image('image', rendering.image)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
