import taichi as ti

from typing import Callable, List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from taichi_splatting.data_types import RasterConfig
from taichi_splatting.misc.renderer2d import project_gaussians2d
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from renderer2d import Gaussians2D
from mlp import mlp, mlp_body
from taichi_splatting.rasterizer.function import rasterize

import os
import matplotlib.pyplot as plt
from functools import partial
import math
import cv2
import argparse
import numpy as np

import torch
from tqdm import tqdm
from fit_image_gaussians import parse_args, partial, log_lerp, psnr, display_image, flatten_tensorclass, split_tensorclass, mean_dicts, lerp
from fused_ssim import fused_ssim
from taichi_splatting.torch_lib.util import check_finite



def normalize_raster(raster_image: torch.Tensor,
                     raster_alpha: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:
    """
    Normalizes the raster image using raster.image_weight
     with an epsilon for numerical stability.
    
    Args:
        raster_image (torch.Tensor): The raster image tensor (e.g., shape [B, C, H, W]).
        raster_alpha (torch.Tensor): The image_weight tensor for normalization (e.g., shape [B, 1, H, W]).
        eps (float): A small epsilon value to prevent division by zero.
        
    Returns:
        torch.Tensor: The normalized raster image.
    """
    m = nn.Sigmoid()
    normalized_image = raster_image / (m(raster_alpha).unsqueeze(-1) + eps)
    return normalized_image


def group_norm(num_channels: int):
    return nn.GroupNorm(num_groups=1, num_channels=num_channels, affine=False)


import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):

    def __init__(self, in_channels, inter_channels):
        super(AttentionGate, self).__init__()
        self.theta_x = nn.Conv2d(in_channels,
                                 inter_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.phi_g = nn.Conv2d(in_channels,
                               inter_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)
        self.psi = nn.Conv2d(inter_channels,
                             1,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g)
        f = self.relu(theta_x + phi_g)
        # print(f"f: {f.shape}")
        psi = self.sigmoid(self.psi(f))
        return x * psi


class UNet3(nn.Module):
    "This is the 3 layer Unet with pooling and attention gate"

    def __init__(self, dropout_rate=0.5, l2_lambda=0.01):
        super(UNet4, self).__init__()

        def conv_block(in_channels, out_channels, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels), nn.ReLU(),
                nn.Dropout(dropout_rate))

        def upconv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=2,
                                   stride=2))

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

    def forward(self, x):

        x1 = self.layer1(x)
        p1 = self.pool1(x1)

        x2 = self.layer2(p1)
        p2 = self.pool2(x2)

        x3 = self.layer3(p2)
        p3 = self.pool3(x3)

        x4 = self.layer4(p3)

        up3 = self.upconv3(x4)

        att3 = self.attention3(x3, up3)

        concat3 = torch.cat([up3, att3], dim=1)
        x5 = self.layer5(concat3)

        up2 = self.upconv2(x5)
        att2 = self.attention2(x2, up2)
        concat2 = torch.cat([up2, att2], dim=1)
        x6 = self.layer6(concat2)

        up1 = self.upconv1(x6)

        att1 = self.attention1(x1, up1)
        concat1 = torch.cat([up1, att1], dim=1)
        x7 = self.layer7(concat1)

        output = self.output_layer(x7)

        return output


class UNet2D(nn.Module):
    "Unet with the down and up layer"

    def __init__(self,
                 f: int,
                 activation: Callable = nn.ReLU,
                 norm: Callable = group_norm) -> None:
        super().__init__()

        # Downsampling: f -> 2f -> 4f -> 8f -> 16f
        def make_down_layer(i: int) -> nn.Sequential:
            in_channels = f * (2**i)
            out_channels = 2 * in_channels
            return nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=3,
                          padding=1,
                          stride=2),
                norm(out_channels),
                activation(),
            )

        self.down_layers = nn.ModuleList(
            [make_down_layer(i) for i in range(2)])

        # Up path: 16f -> 8f -> 4f -> 2f -> f
        def make_up_layer(i: int) -> nn.Sequential:
            out_channels = f * (2**i)
            in_channels = 2 * out_channels
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels,
                                   out_channels,
                                   kernel_size=2,
                                   stride=2),
                norm(out_channels),
                activation(),
            )

        self.up_layers = nn.ModuleList(
            reversed([make_up_layer(i) for i in range(2)]))

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

    def __init__(self,
                 inputs: int,
                 outputs: int,
                 n_render: int = 16,
                 n_base: int = 64,
                 method:str = "mlp"):
        super().__init__()
        self.method = method
        self.down_project = nn.Linear(n_base, n_render)
        if self.method == "mlp_unet":
            n_render =  n_render + 3  # Increase input size to include rendering features
        self.init_mlp = mlp_body(inputs,
                                 hidden_channels=[n_base] * 2,
                                 activation=nn.ReLU,
                                 norm=nn.LayerNorm)
        self.up_project = nn.Linear(n_render, n_base)
        self.unet = UNet2D(f=n_render, activation=nn.ReLU)
        if self.method == "mlp":
          input_size = inputs
        else:
          input_size = n_base
        self.final_mlp = mlp(input_size,
                            outputs=outputs,
                            hidden_channels=[n_base] * 2,
                            activation=nn.ReLU,
                            norm=nn.LayerNorm,
                            output_scale=1e-12)


    @torch.compiler.disable
    def render(self,
               features: torch.Tensor,
               gaussians: Gaussians2D,
               image_size: Tuple[int, int],
               raster_config: RasterConfig = RasterConfig()):
        h, w = image_size

        gaussians2d = project_gaussians2d(gaussians)
        raster = rasterize(gaussians2d=gaussians2d,
                           depth=gaussians.z_depth.clamp(0, 1),
                           features=features,
                           image_size=(w, h),
                           config=raster_config)
        raster_image = normalize_raster(raster_image=raster.image,
                                        raster_alpha=raster.image_weight,
                                        eps=1e-12)
        return raster_image.unsqueeze(0).permute(0, 3, 1, 2).to(
            memory_format=torch.channels_last
        )  # B, H, W, n_render -> 1, n_render, H, W

    def sample_positions(
            self,
            image: torch.Tensor,  # 1, n_render, H, W
            positions: torch.Tensor,  # B, 2
    ) -> torch.Tensor:  # B, n_render
        h, w = image.shape[-2:]
        B = positions.shape[0]
        # normalize positions to be in the range [w, h] -> [-1, 1] for F.grid_sample
        positions = ((positions / positions.new_tensor([w, h])) * 2.0 -
                     1.0).view(1, 1, B, 2)
        samples = F.grid_sample(image, positions,
                                align_corners=False)  # B, n_render, 1, 1
        return samples.view(B, -1)  # B, n_render, 1, 1 -> B, n_render

    def forward(self, x: torch.Tensor, gaussians: Gaussians2D,
                image_size: Tuple[int, int], raster_config: RasterConfig,
                ref_image: torch.Tensor) -> torch.Tensor:
        if self.method == "mlp_unet":

          feature = self.init_mlp(x)      # B,inputs -> B, n_base
          x = self.down_project(feature)  # B, n_base -> B, n_render

          image = self.render(x.to(torch.float32), gaussians, image_size, raster_config) # B, n_render -> 1, n_render, H, W

          precon_image = ref_image.unsqueeze(0).permute(0,3,1,2)
          con_image = torch.cat((precon_image,image),dim=1)

          image = self.unet(con_image)   # B, n_render, H, W -> B, n_render, H, W

          # sample at gaussian centres from the unet output
          x = self.sample_positions(image, gaussians.position)
          x = self.up_project(x)              # B, n_render -> B, n_base
          # shortcut from output of init_mlp
          x = self.final_mlp(x+ feature)  # B, n_base -> B, outputs
        else:
          x = self.final_mlp(x)
        return x


class Trainer:

    def __init__(self,
                 optimizer_mlp: torch.nn.Module,
                 mlp_opt: torch.optim.Optimizer,
                 ref_image: torch.Tensor,
                 config: RasterConfig,
                 opacity_reg=0.0,
                 scale_reg=0.0):

        self.optimizer_mlp = optimizer_mlp
        self.mlp_opt = mlp_opt

        self.config = config
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg

        self.ref_image: torch.Tensor = ref_image
        self.running_scales = None

    def render(self, gaussians):
        h, w = self.ref_image.shape[:2]

        gaussians2d = project_gaussians2d(gaussians)
        raster = rasterize(gaussians2d=gaussians2d,
                           depth=gaussians.z_depth.clamp(0, 1),
                           features=gaussians.feature,
                           image_size=(w, h),
                           config=self.config)
        return raster

    def render_step(self, gaussians):
        with torch.enable_grad():
            raster = self.render(gaussians)

            h, w = self.ref_image.shape[:2]
            scale = torch.exp(gaussians.log_scaling) / min(w, h)
            opacity_reg = self.opacity_reg * gaussians.opacity.mean()
            scale_reg = self.scale_reg * scale.pow(2).mean()
            depth_reg = 0.0 * gaussians.z_depth.sum()

            l1 = torch.nn.functional.l1_loss(raster.image, self.ref_image)
            mse = torch.nn.functional.mse_loss(raster.image, self.ref_image)
            # # print(raster.image.shape)
            # # print(self.ref_image.shape)
            ssim = 1 - fused_ssim(raster.image.unsqueeze(0),
                                  self.ref_image.unsqueeze(0),
                                  train=True)
            loss = mse + ssim * 0.1 + l1 * 0.1 + opacity_reg + scale_reg + depth_reg
            loss.backward()

            return dict(loss=loss.item(),
                        opacity_reg=opacity_reg.item(),
                        scale_reg=scale_reg.item())

    def get_gradients(self, gaussians):
        gaussians = gaussians.clone()
        gaussians.requires_grad_(True)
        metrics = self.render_step(gaussians)
        grad = gaussians.grad

        mean_abs_grad = grad.abs().mean(dim=0)
        if self.running_scales is None:
            self.running_scales = mean_abs_grad
        else:
            self.running_scales = lerp(0.999, self.running_scales,
                                       mean_abs_grad)

        return grad * 1e7, metrics

    def test(self, gaussians, step_size=0.01, epoch_size=100):
        """Run inference using the trained model."""
        metrics = []
        for i in range(epoch_size):
            with torch.enable_grad():
                # Compute gradients
                grad, metric = self.get_gradients(gaussians)
                check_finite(grad, "grad")

                # Flatten gradients and predict updates using the MLP
                inputs = flatten_tensorclass(grad)

                with torch.no_grad():
                    step = self.optimizer_mlp(inputs, gaussians,
                                              self.ref_image.shape[:2],
                                              self.config, self.ref_image)
                    step = split_tensorclass(gaussians, step)
                    metrics.append(metric)
                # Update Gaussians with the step
                gaussians = gaussians - step * step_size
        return gaussians, mean_dicts(metrics)

    def train_epoch(self, gaussians, step_size=0.01, epoch_size=100):
        metrics = []
        for i in range(epoch_size):

            grad, _ = self.get_gradients(gaussians)
            check_finite(grad, "grad")
            self.mlp_opt.zero_grad()

            inputs = flatten_tensorclass(grad)

            with torch.enable_grad():
                step = self.optimizer_mlp(inputs, gaussians,
                                          self.ref_image.shape[:2],
                                          self.config,
                                          self.ref_image)  # h ,w channel
                step = split_tensorclass(gaussians, step)

                metrics.append(self.render_step(gaussians - step))

            self.mlp_opt.step()
            gaussians = gaussians - step * step_size

        return gaussians, mean_dicts(metrics)


def main():

    torch.set_printoptions(precision=4, sci_mode=True)

    cmd_args = parse_args()
    device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    ref_image = cv2.imread(
        '/csse/users/pwl25/pear/images/DSC_1366_12kv2r16k_7.jpg')
    assert ref_image is not None, f'Could not read {cmd_args.image_file}'

    h, w = ref_image.shape[:2]

    TaichiQueue.init(arch=ti.cuda,
                     log_level=ti.INFO,
                     debug=cmd_args.debug,
                     device_memory_GB=0.1)

    if cmd_args.show:
        cv2.namedWindow('rendered', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('rendered', w, h)

    torch.manual_seed(cmd_args.seed)
    torch.cuda.random.manual_seed(cmd_args.seed)

    gaussians = random_2d_gaussians(cmd_args.n, (w, h),
                                    alpha_range=(0.5, 1.0),
                                    scale_factor=1.0).to(
                                        torch.device('cuda:0'))
    n_inputs = sum(
        [np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])

    # Create the MLP
    optimizer = GaussianMixer(inputs=n_inputs,
                              outputs=n_inputs,
                              n_render=16,
                              n_base=128,
                              method = cmd_args.method).to(device)
    optimizer.to(device=device)

    dataset_folder = cmd_args.image_file  # Using the argument as a folder path
    image_files = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    optimizer = torch.compile(optimizer)
    optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.00001)

    config = RasterConfig()

    trainer = Trainer(optimizer_mlp=optimizer,
                      mlp_opt=optimizer_opt,
                      ref_image=ref_image,
                      config=config,
                      opacity_reg=cmd_args.opacity_reg,
                      scale_reg=cmd_args.scale_reg)
    epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]

    iteration = 0
    test_interval = cmd_args.test

    for num, img_path in enumerate(image_files, start=1):
        ref_image = cv2.imread(img_path)
        ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32,
                                                   device=device) / 255
        assert ref_image is not None, f'Could not read {img_path}'

        h, w = ref_image.shape[:2]
        torch.manual_seed(cmd_args.seed)
        torch.cuda.random.manual_seed(cmd_args.seed)
        trainer.ref_image = ref_image
        gaussians = random_2d_gaussians(cmd_args.n, (w, h),
                                        alpha_range=(0.5, 1.0),
                                        scale_factor=1.0).to(
                                            torch.device('cuda:0'))
        pbar = tqdm(total=cmd_args.iters, desc="Initializing")

        for epoch_size in epochs:

            metrics = {}

            # Set warmup schedule for first iterations - log interpolate
            step_size = log_lerp(min(iteration / 100., 1.0), 0.1, 1.0)

            if cmd_args and num % test_interval == 0:
                pbar.set_description(f"Testing Progress")
                gaussians, train_metrics = trainer.test(gaussians,
                                                        epoch_size=epoch_size,
                                                        step_size=step_size)

            else:
                pbar.set_description(f"Training Progress")
                gaussians, train_metrics = trainer.train_epoch(
                    gaussians, epoch_size=epoch_size, step_size=step_size)

            image = trainer.render(gaussians).image
            if cmd_args.show:
                display_image('rendered', image)

            metrics['CPSNR'] = psnr(ref_image, image).item()
            metrics['n'] = gaussians.batch_size[0]
            metrics.update(train_metrics)

            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics[k] = f'{v:.4f}'
                if isinstance(v, int):
                    metrics[k] = f'{v:4d}'

            pbar.set_postfix(**metrics)

            iteration += epoch_size
            pbar.update(epoch_size)


if __name__ == "__main__":
    main()
