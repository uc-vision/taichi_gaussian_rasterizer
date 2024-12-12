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
from renderer2d import Gaussians2D
from mlp import mlp, mlp_body
from taichi_splatting.rasterizer.function import rasterize
def normalize_raster(raster_image: torch.Tensor, raster_alpha: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalizes the raster image using raster.alpha with an epsilon for numerical stability.
    
    Args:
        raster_image (torch.Tensor): The raster image tensor (e.g., shape [B, C, H, W]).
        raster_alpha (torch.Tensor): The alpha tensor for normalization (e.g., shape [B, 1, H, W]).
        eps (float): A small epsilon value to prevent division by zero.
        
    Returns:
        torch.Tensor: The normalized raster image.
    """
    m = nn.Sigmoid()
    normalized_image = raster_image / (m(raster_alpha).unsqueeze(-1) + eps)
    return normalized_image

def group_norm(num_channels:int):
  return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=False)



import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
  def __init__(self, in_channels, inter_channels):
    super(AttentionGate, self).__init__()
    self.theta_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
    self.phi_g = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0)
    self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, g):
    theta_x = self.theta_x(x)
    phi_g = self.phi_g(g)
    f = self.relu(theta_x + phi_g)
    # print(f"f: {f.shape}")
    psi = self.sigmoid(self.psi(f))
    return x * psi

class UNet4(nn.Module):
  def __init__(self, dropout_rate=0.5, l2_lambda=0.01):
    super(UNet4, self).__init__()
    def conv_block(in_channels, out_channels, dropout_rate):
      return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.Dropout(dropout_rate)
      )

    def upconv_block(in_channels, out_channels):
      return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
    self.layer1 = conv_block(19, 64, dropout_rate)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.layer2 = conv_block(64, 128, dropout_rate)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.layer3 = conv_block(128, 256, dropout_rate)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Bottleneck
    self.layer4 = conv_block(256, 512, dropout_rate)

    # Expanding Path
    self.upconv3 = upconv_block(512, 256)
    self.attention3 = AttentionGate(256, 128)
    self.layer5 = conv_block(512, 256, dropout_rate)

    self.upconv2 = upconv_block(256, 128)
    self.attention2 = AttentionGate(128, 64)
    self.layer6 = conv_block(256, 128, dropout_rate)

    self.upconv1 = upconv_block(128, 64)
    self.attention1 = AttentionGate(64, 32)
    self.layer7 = conv_block(128, 19, dropout_rate)

    # Output Layer
    self.output_layer = nn.Conv2d(19, 19, kernel_size=1)


  

    # Contracting Path
      
  def forward(self, x):
    # Add positional distance information
    # print(f"x:{x.shape}")
    # Contracting Path
    x1 = self.layer1(x)
    p1 = self.pool1(x1)
    # print(f"x1:{x1.shape}")
    x2 = self.layer2(p1)
    p2 = self.pool2(x2)
    # print(f"x2:{x2.shape}")
    x3 = self.layer3(p2)
    p3 = self.pool3(x3)
    # print(f"x3:{x3.shape}")
    # Bottleneck
    x4 = self.layer4(p3)
    # print(f"x4:{x4.shape}")
    # Expanding Path
    up3 = self.upconv3(x4)
    # print(f"up3:{up3.shape}")
    
    att3 = self.attention3(x3, up3)
    # print(f"att3{att3}")
    concat3 = torch.cat([up3, att3], dim=1)
    x5 = self.layer5(concat3)
    # print(f"x5:{x5.shape}")
    up2 = self.upconv2(x5)
    att2 = self.attention2(x2, up2)
    concat2 = torch.cat([up2, att2], dim=1)
    x6 = self.layer6(concat2)
    

    up1 = self.upconv1(x6)
    
    att1 = self.attention1(x1, up1)
    concat1 = torch.cat([up1, att1], dim=1)
    x7 = self.layer7(concat1)
    # print(x.shape)

    # Output Layer
    output = self.output_layer(x7)

    return output



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
        make_down_layer(i) for i in range(2)
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
        make_up_layer(i) for i in range(2)
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
class DenoisingLayer(nn.Module):
    
    def __init__(self,channels):
        super(DenoisingLayer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=4),  # Add MaxPooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=8, stride=4)   # Add MaxPooling
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


        
class GaussianMixer(nn.Module):
  def __init__(self, inputs:int, outputs:int, 
               n_render:int=16,
               n_base:int=64):
    super().__init__()
    
    self.init_mlp = mlp_body(inputs, hidden_channels=[n_base] * 2, 
                        activation=nn.ReLU, norm=nn.LayerNorm)
    
    self.down_project = nn.Linear(n_base, n_render)
    
    self.up_project = nn.Linear(n_render+3, n_base)
    self.unet_4 = UNet4()
    self.unet = UNet2D(f=n_render+3, activation=nn.ReLU)
    self.denoising_layer = DenoisingLayer(n_render+3)
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
    # print(raster.image.shape)
    # print(raster.image_weight.shape)
    raster_image = normalize_raster(raster_image=raster.image,raster_alpha=raster.image_weight, eps=1e-12)
    return raster_image.unsqueeze(0).permute(0, 3, 1, 2).to(memory_format=torch.channels_last) # B, H, W, n_render -> 1, n_render, H, W

  def sample_positions(self, image:torch.Tensor,  # 1, n_render, H, W
                       positions:torch.Tensor,    # B, 2
                       ) -> torch.Tensor:         # B, n_render
    h, w = image.shape[-2:]
    B = positions.shape[0]

    # normalize positions to be in the range [w, h] -> [-1, 1] for F.grid_sample
    positions = ((positions / positions.new_tensor([w, h])) * 2.0 - 1.0).view(1, 1, B, 2)
    samples = F.grid_sample(image, positions, align_corners=False) # B, n_render, 1, 1

    return samples.view(B, -1) # B, n_render, 1, 1 -> B, n_render
  


  def forward(self, x: torch.Tensor, gaussians: Gaussians2D, image_size: Tuple[int, int], raster_config: RasterConfig,ref_image:torch.Tensor) -> torch.Tensor:
    
    
    feature = self.init_mlp(x)      # B,inputs -> B, n_base
    x = self.down_project(feature)  # B, n_base -> B, n_render

    image = self.render(x.to(torch.float32), gaussians, image_size, raster_config) # B, n_render -> 1, n_render, H, W
    precon_image = ref_image.unsqueeze(0).permute(0,3,1,2)
    
    con_image = torch.cat((precon_image,image),dim=1)
    # image = self.unet_4(con_image)
    image = self.unet(con_image)   # B, n_render, H, W -> B, n_render, H, W
    # image = self.denoising_layer(image)
    # sample at gaussian centres from the unet output
    x = self.sample_positions(image, gaussians.position) 
    x = self.up_project(x)              # B, n_render -> B, n_base

    # shortcut from output of init_mlp
    x = self.final_mlp( x+ feature) 
    # print(x.shape)    # B, n_base -> B, outputs
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
  mixer = GaussianMixer(inputs=n_inputs, outputs=n_inputs, n_render=13, n_base=128).to(device)
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
        