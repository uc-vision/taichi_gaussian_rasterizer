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
from fit_image_gaussians import parse_args, Trainer, partial, log_lerp, psnr, display_image

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite
import os
import matplotlib.pyplot as plt


def save_checkpoint(optimizer,
                    optimizer_opt,
                    metrics_history,
                    epoch_size,
                    filename="checkpoint.pth"):
    checkpoint = {
        'epoch_size': epoch_size,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_opt_state_dict': optimizer_opt.state_dict(),
        'metrics': metrics_history,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def main():
    torch.set_printoptions(precision=4, sci_mode=True)

    cmd_args = parse_args()
    device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    # Get all image file paths from the dataset folder
    dataset_folder = cmd_args.image_file  # Using the argument as a folder path
    image_files = [
        os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    assert image_files, f'No valid image files found in {dataset_folder}'

    TaichiQueue.init(arch=ti.cuda,
                     log_level=ti.INFO,
                     debug=cmd_args.debug,
                     device_memory_GB=0.1)

    metrics_history = []  # To store metrics for plotting
    ref_image = cv2.imread(
        '/csse/users/pwl25/pear/images/DSC_1366_12kv2r16k_7.jpg')
    # assert ref_image is not None, f'Could not read {'/csse/users/pwl25/pear/images/DSC_1366_12kv2r16k_7.jpg'}'

    h, w = ref_image.shape[:2]
    gaussians = random_2d_gaussians(cmd_args.n, (w, h),
                                    alpha_range=(0.5, 1.0),
                                    scale_factor=1.0).to(
                                        torch.device('cuda:0'))
    channels = sum(
        [np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])

    # Create the MLP
    optimizer = mlp(inputs=channels,
                    outputs=channels,
                    hidden_channels=[128, 128, 128],
                    activation=nn.ReLU,
                    norm=partial(nn.LayerNorm, elementwise_affine=False),
                    output_scale=1e-12)
    optimizer.to(device=device)
    optimizer = torch.compile(optimizer)
    optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.0001)

    # ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255
    config = RasterConfig()
    for img_path in tqdm(image_files, desc="Training on dataset"):
        ref_image = cv2.imread(img_path)
        assert ref_image is not None, f'Could not read {img_path}'

        h, w = ref_image.shape[:2]
        print(f'Training on image: {img_path}, size: {w}x{h}')

        torch.manual_seed(cmd_args.seed)
        torch.cuda.random.manual_seed(cmd_args.seed)

        gaussians = random_2d_gaussians(cmd_args.n, (w, h),
                                        alpha_range=(0.5, 1.0),
                                        scale_factor=1.0).to(
                                            torch.device('cuda:0'))
        # channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])

        # # Create the MLP
        # optimizer = mlp(inputs=channels, outputs=channels,
        #                 hidden_channels=[128, 128, 128],
        #                 activation=nn.ReLU,
        #                 norm=partial(nn.LayerNorm, elementwise_affine=False),
        #                 output_scale=1e-12
        #                 )
        # optimizer.to(device=device)
        # optimizer = torch.compile(optimizer)
        # optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.0001)

        ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32,
                                                   device=device) / 255
        # config = RasterConfig()

        trainer = Trainer(optimizer,
                          optimizer_opt,
                          ref_image,
                          config,
                          opacity_reg=cmd_args.opacity_reg,
                          scale_reg=cmd_args.scale_reg)

        epochs = [
            cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)
        ]
        pbar = tqdm(total=cmd_args.iters)
        iteration = 0
        for epoch_size in epochs:
            metrics = {}
            step_size = log_lerp(min(iteration / 1000., 1.0), 0.1, 1.0)
            gaussians, train_metrics = trainer.train_epoch(
                gaussians, epoch_size=epoch_size, step_size=step_size)
            iteration += epoch_size
            image = trainer.render(gaussians).image
            if cmd_args.show:
                display_image('rendered', image)

            # Record metrics for plotting
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
        print(epoch_size)
        save_checkpoint(optimizer,
                        optimizer_opt,
                        metrics_history,
                        epoch_size,
                        filename="checkpoint.pth")
    # Plot training results
    # iterations, cpsnr_values = zip(*metrics_history)
    # plt.figure(figsize=(10, 6))
    # plt.plot(iterations, cpsnr_values, label="CPSNR")
    # plt.xlabel("Iteration")
    # plt.ylabel("CPSNR")
    # plt.title("Training Progress on Dataset")
    # plt.legend()
    # plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
