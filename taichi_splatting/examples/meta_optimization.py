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
from gaussian_mixer import GaussianMixer
from taichi_splatting.torch_lib.util import check_finite
class Trainer:

    def __init__(self,
                 optimizer_mlp: torch.nn.Module,
                 mlp_opt: torch.optim.Optimizer,
                 adam_optimizer: torch.optim.Optimizer,
                 ref_image: torch.Tensor,
                 config: RasterConfig,
                 opacity_reg=0.0,
                 scale_reg=0.0):
        self.adam_optimizer = adam_optimizer
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
            self.adam_optimizer.zero_grad()
            grad, metric = self.get_gradients(gaussians)
            check_finite(grad, "grad")

            # Store initial Gaussian state
            gaussians_clone = gaussians.clone().detach().requires_grad_(True)
            
            with torch.enable_grad():
                self.render_step(gaussians=gaussians_clone)
            self.adam_optimizer.step()
            gau_grad,_ = self.get_gradients(gaussians=gaussians_clone)
            adam_step = gaussians_clone - gau_grad

            # Train MLP to mimic Adam step
            self.mlp_opt.zero_grad()
            grad, metric = self.get_gradients(gaussians)
            check_finite(grad, "grad")
            inputs = flatten_tensorclass(grad)

            with torch.enable_grad():
                step = self.optimizer_mlp(inputs, gaussians_clone, self.ref_image.shape[:2], self.config, self.ref_image)
                step = split_tensorclass(gaussians, step)
                
                h, w = self.ref_image.shape[:2]
                scale = torch.exp(gaussians.log_scaling) / min(w, h)
                opacity_reg = self.opacity_reg * gaussians.opacity.mean()
                scale_reg = self.scale_reg * scale.pow(2).mean()
                depth_reg = 0.0 * gaussians.z_depth.sum()
                # Compare rendered images between Adam step and MLP step
                r1 = self.render(adam_step)
                r2 = self.render(gaussians - step)
                meta_loss = torch.nn.functional.l1_loss(r2.image, r1.image)
                meta_loss = meta_loss + opacity_reg + scale_reg + depth_reg
                meta_loss.backward()

            self.mlp_opt.step()
            
            # Update Gaussians with the learned MLP step
            gaussians = gaussians - step * step_size
            
            # Track metrics
            metrics.append(metric)
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
    learning_rate = 0.000000001
    adam_optimizer = adam_optimizer = torch.optim.Adam(
        [
            {'params': gaussians.alpha_logit, 'lr': learning_rate},
            {'params': gaussians.feature, 'lr': learning_rate},
            {'params': gaussians.log_scaling, 'lr': learning_rate},
            {'params': gaussians.position, 'lr': learning_rate},
            {'params': gaussians.rotation, 'lr': learning_rate},
            {'params': gaussians.z_depth, 'lr': learning_rate},
        ],
        betas=(0.9, 0.999),  # Default beta values for Adam
        eps=1e-8  # Small epsilon to prevent division by zero
    )
    config = RasterConfig()

    trainer = Trainer(optimizer_mlp=optimizer,
                        adam_optimizer=adam_optimizer,
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
