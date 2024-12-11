# for epoch in tqdm.tqdm(range(epochs)):
#       with torch.enable_grad():

#         optimizer_opt.zero_grad()

#         y = mixer(x, gaussians, image_size, config)
#         c = x-y

#         rendered_image = mixer.render(c, gaussians, image_size).squeeze(0)

#         loss = criterion(rendered_image.permute(1,2, 0), ref_image)

#         loss.backward()
#         if epoch % 10 == 0:  # Print every 10 epochs
#           print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}")

#       optimizer_opt.step()
#       x = c.detach()
#       # y.sum().backward()
import os
import matplotlib.pyplot as plt
from functools import partial
import math
from typing import Dict, List
import cv2
import argparse
import numpy as np
import torch.nn as nn
import taichi as ti

import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.examples.mlp import mlp
from taichi_splatting.misc.renderer2d import project_gaussians2d
from fit_image_gaussians import parse_args, partial, log_lerp, psnr, display_image, flatten_tensorclass, split_tensorclass, mean_dicts, lerp
from gaussian_mixer import GaussianMixer
from taichi_splatting.rasterizer.function import rasterize
from fused_ssim import fused_ssim
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite
import os
import matplotlib.pyplot as plt


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
            ssim= 1 - fused_ssim(raster.image.unsqueeze(0), self.ref_image.unsqueeze(0),train=True)
            loss =  mse + ssim *0.1+ l1*0.1  + opacity_reg + scale_reg + depth_reg
            loss.backward()

            return dict(loss=loss.item(),
                        opacity_reg=opacity_reg.item(),
                        scale_reg=scale_reg.item())

    def get_gradients(self, gaussians):
        gaussians = gaussians.clone()
        gaussians.requires_grad_(True)
        self.render_step(gaussians)
        grad = gaussians.grad

        mean_abs_grad = grad.abs().mean(dim=0)
        if self.running_scales is None:
            self.running_scales = mean_abs_grad
        else:
            self.running_scales = lerp(0.999, self.running_scales,
                                       mean_abs_grad)

        return grad * 1e7

    def test(self, gaussians):
        """Run inference using the trained model."""
        with torch.no_grad():

            grad = self.get_gradients(gaussians)
            check_finite(grad, "grad")

            inputs = flatten_tensorclass(grad)

            with torch.enable_grad():
                step = self.optimizer_mlp(inputs)
                step = split_tensorclass(gaussians, step)
        raster = self.render(gaussians - step)
        psnr_value = psnr(self.ref_image, raster.image).item()
        print(f"Test PSNR: {psnr_value:.4f}")
        return raster.image

    def train_epoch(self, gaussians, step_size=0.01, epoch_size=100):
        metrics = []
        for i in range(epoch_size):

            grad = self.get_gradients(gaussians)
            check_finite(grad, "grad")
            self.mlp_opt.zero_grad()

            inputs = flatten_tensorclass(grad)
            # input1 = flatten_tensorclass(gaussians)
            # inputs = torch.cat((inputs,input1),dim=-1)

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

    # print(f'Image size: {w}x{h}')

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
                              n_render=128,
                              n_base=256).to(device)
    optimizer.to(device=device)

    # print(optimizer)

    dataset_folder = cmd_args.image_file  # Using the argument as a folder path
    image_files = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    optimizer = torch.compile(optimizer)
    optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.0001)#0.00001 for MSE, #

    config = RasterConfig()

    trainer = Trainer(optimizer_mlp=optimizer,
                      mlp_opt=optimizer_opt,
                      ref_image=ref_image,
                      config=config,
                      opacity_reg=cmd_args.opacity_reg,
                      scale_reg=cmd_args.scale_reg)
    epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]

    pbar = tqdm(total=cmd_args.iters)
    iteration = 0

    for img_path in tqdm(image_files, desc="Training on dataset"):
        ref_image = cv2.imread(img_path)
        ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32,
                                                   device=device) / 255
        assert ref_image is not None, f'Could not read {img_path}'

        h, w = ref_image.shape[:2]
        print(f'Training on image: {img_path}, size: {w}x{h}')

        torch.manual_seed(cmd_args.seed)
        torch.cuda.random.manual_seed(cmd_args.seed)
        trainer.ref_image = ref_image
        gaussians = random_2d_gaussians(cmd_args.n, (w, h),
                                        alpha_range=(0.5, 1.0),
                                        scale_factor=1.0).to(
                                            torch.device('cuda:0'))

        for epoch_size in epochs:

            metrics = {}

            # Set warmup schedule for first iterations - log interpolate
            step_size = log_lerp(min(iteration / 400., 1.0), 0.1, 1.0)

            gaussians, train_metrics = trainer.train_epoch(
                gaussians, epoch_size=epoch_size, step_size=step_size)

            image = trainer.render(gaussians).image
            if cmd_args.show:
                display_image('rendered', image)

            metrics['CPSNR'] = psnr(ref_image, image).item()
            metrics['n'] = gaussians.batch_size[0]
            metrics.update(train_metrics)
            # for key, value in metrics.items():
            #   print(f"{key}: {value}")

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
