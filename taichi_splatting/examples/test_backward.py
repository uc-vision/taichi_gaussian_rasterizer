

import argparse
from functools import partial
from taichi_splatting.data_types import RasterConfig
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
 
def main(args):
    TaichiQueue.init(ti.gpu)

    config = RasterConfig(
        tile_size=args.tile_size,
        pixel_stride=args.pixel_stride,
    )

    torch.manual_seed(0)
    gaussians = random_2d_gaussians(args.n, (640, 480), scale_factor=10.0, alpha_range=(0.2, 0.3)).cuda()

    gaussians.requires_grad_(True)
    rendering = render_gaussians(gaussians, (640, 480), config)
    rendering.image.sum().backward()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--tile_size", type=int, default=16)
    parser.add_argument("--pixel_stride", type=str, default="1,1")
    args = parser.parse_args()

    args.pixel_stride = tuple([int(x) for x in args.pixel_stride.split(",")])
    

    try:
        main(args)
    except KeyboardInterrupt:
        pass
