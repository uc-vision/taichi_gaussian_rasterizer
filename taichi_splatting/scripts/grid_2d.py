import math
import torch
import torch.nn.functional as F
import cv2

from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.conic.renderer2d import render_gaussians

import taichi as ti



def display_image(name, image):
    image = (image.detach().clamp(0, 1) * 255).to(torch.uint8)
    image = image.cpu().numpy()

    cv2.imshow(name, image)
    cv2.waitKey(0)


def grid_2d(i, j):
  x, y = torch.meshgrid(torch.arange(i), torch.arange(j), indexing='ij')
  return torch.stack([x, y], dim=-1)





def main():
  ti.init(arch=ti.cuda)

  image_size = (1000, 1000)
  device = torch.device('cuda:0')

  
  points = 500 + (grid_2d(3, 3).view(-1, 2).to(torch.float32) - 1) * 200  
  n = points.shape[0]

  r = torch.tensor([1.0, 0.0])
  s = (torch.tensor([100.0, 100.0]) / math.sqrt(2)).log()

  # position     : torch.Tensor # 2  - xy
  # depth        : torch.Tensor # 1  - for sorting
  # log_scaling   : torch.Tensor # 2  - scale = exp(log_scalining) 
  # rotation      : torch.Tensor # 2  - unit length imaginary number
  # alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  # feature      : torch.Tensor # N  - (any rgb, label etc)


  gaussians = Gaussians2D(
    position = points,
    log_scaling = s.view(1, 2).expand(n, -1),
    rotation = r.view(1, 2).expand(n, -1),
    alpha_logit = torch.full((n, ), fill_value=10.0),
    feature = torch.rand(n, 3),
    depth = torch.zeros(n, 1),  # dummy depth
    batch_size = (n,)
  ).to(device=device)



  image = render_gaussians(gaussians, image_size,
                           raster_config=RasterConfig(beta=200.0)).image
  display_image("image", image)
  


if __name__=="__main__":
   main()