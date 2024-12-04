import taichi as ti

from typing import Callable, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from .renderer2d import Gaussians2D
from .mlp import mlp, mlp_body
from taichi_splatting.rasterizer.function import rasterize

def group_norm(num_channels:int):
  return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=False)

class UNet2D(nn.Module):
  def __init__(self, f:int, activation:Callable=nn.ReLU, norm:Callable=group_norm) -> None:
    super().__init__()

    # Downsampling: f -> 2f -> 4f -> 8f -> 16f
    def make_down_layer(i: int) -> nn.Sequential:
        in_channels = f * (2**i)
        out_channels = 2 * in_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=1, stride=2),
            norm(out_channels),
            activation(),
        )

    self.down_layers = nn.ModuleList([
        make_down_layer(i) for i in range(4)
    ])

    # Up path: 16f -> 8f -> 4f -> 2f -> f
    def make_up_layer(i: int) -> nn.Sequential:
        out_channels = f * (2**i)
        in_channels = 2 * out_channels
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            norm(out_channels),
            activation(),
        )

    self.up_layers = nn.ModuleList(reversed([
        make_up_layer(i) for i in range(4)
    ]))

    self.final_layer = nn.Sequential(
      nn.Conv2d(f, f, kernel_size=3, padding=1),
      norm(f),
      activation(),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    intermediates: List[torch.Tensor] = []  
    # Down path: f -> 2f -> 4f -> 8f -> 16f
    for layer in self.down_layers:
        intermediates.append(x)
        x = layer(x)

    # Up path with skip connections: 16f -> 8f -> 4f -> 2f -> f
    intermediates = list(reversed(intermediates))

    for i, layer in enumerate(self.up_layers):
        x = layer(x) + intermediates[i]
        
    return self.final_layer(x)


class GaussianMixer(nn.Module):
  def __init__(self, inputs:int, outputs:int, 
               n_render:int=16,
               n_base:int=64):
    super().__init__()
    
    self.init_mlp = mlp_body(inputs, hidden_channels=[n_base] * 2, 
                        activation=nn.ReLU, norm=nn.LayerNorm)
    
    self.down_project = nn.Linear(n_base, n_render)
    
    self.up_project = nn.Linear(n_render, n_base)
    self.unet = UNet2D(f=n_render, activation=nn.ReLU)

    self.final_mlp = mlp(n_base, outputs=outputs, hidden_channels=[n_base] * 2, 
                         activation=nn.ReLU, norm=nn.LayerNorm, output_scale=1e-12)


  @torch.compiler.disable
  def render(self, features:torch.Tensor, gaussians:Gaussians2D, image_size: Tuple[int, int], raster_config: RasterConfig = RasterConfig()):
    h, w = image_size
    
    gaussians2d = project_gaussians2d(gaussians) 
    raster = rasterize(gaussians2d=gaussians2d, 
      depth=gaussians.z_depth.clamp(0, 1),
      features=features, 
      image_size=(w, h), 
      config=raster_config)
    return raster.image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last) # B, H, W, n_render -> 1, n_render, H, W

  def sample_positions(self, image:torch.Tensor,  # 1, n_render, H, W
                       positions:torch.Tensor,    # B, 2
                       ) -> torch.Tensor:         # B, n_render
    h, w = image.shape[-2:]
    B = positions.shape[0]

    # normalize positions to be in the range [w, h] -> [-1, 1] for F.grid_sample
    positions = ((positions / positions.new_tensor([w, h])) * 2.0 - 1.0).view(1, 1, B, 2)
    samples = F.grid_sample(image, positions, align_corners=False) # B, n_render, 1, 1

    return samples.view(B, -1) # B, n_render, 1, 1 -> B, n_render
  


  def forward(self, x: torch.Tensor, gaussians: Gaussians2D, image_size: Tuple[int, int], raster_config: RasterConfig) -> torch.Tensor:
    feature = self.init_mlp(x)      # B,inputs -> B, n_base
    x = self.down_project(feature)  # B, n_base -> B, n_render

    image = self.render(x.to(torch.float32), gaussians, image_size, raster_config) # B, n_render -> 1, n_render, H, W
    image = self.unet(image)   # B, n_render, H, W -> B, n_render, H, W

    # sample at gaussian centres from the unet output
    x = self.sample_positions(image, gaussians.position) 
    x = self.up_project(x)              # B, n_render -> B, n_base

    # shortcut from output of init_mlp
    x = self.final_mlp(x + feature)     # B, n_base -> B, outputs
    return x


if __name__ == '__main__':

  torch.set_float32_matmul_precision('high')
  TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=False, device_memory_GB=0.1)

  device = torch.device('cuda', 0)

  # unet = UNet2D(f=16).to(device)
  # print(unet)

  # x = torch.randn(1, 16, 32, 32).to(device)
  # y = unet(x)
  # print(y.shape)


  n_inputs = 10
  mixer = GaussianMixer(inputs=n_inputs, outputs=n_inputs, n_render=16, n_base=128).to(device)
  # mixer = torch.compile(mixer, options={"max_autotune": True})
  mixer = torch.compile(mixer)

  image_size = (320, 240)
  gaussians = random_2d_gaussians(100000, image_size, alpha_range=(0.5, 1.0), scale_factor=1.0).to(device) 

  # config = RasterConfig(pixel_stride=(1, 1), tile_size=8) # can handle 64 dimension rendering
  config = RasterConfig()

  x = torch.randn(gaussians.batch_size[0], 10).to(device)
  with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    y = mixer(x, gaussians, image_size, config)
    y.sum().backward()  

    for _ in tqdm.tqdm(range(1000)):
      y = mixer(x, gaussians, image_size, config)
      y.sum().backward()
        
