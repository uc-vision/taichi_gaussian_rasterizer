
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
from fit_image_gaussians import parse_args,Trainer,partial,log_lerp,psnr,display_image

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite
import os
import matplotlib.pyplot as plt

def save_checkpoint(optimizer,optimizer_opt, metrics_history,epoch_size, filename="checkpoint.pth"):
    checkpoint = {
        'epoch_size': epoch_size,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_opt_state_dict': optimizer_opt.state_dict(),
        'metrics': metrics_history,
        
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")





def batch_images(image_files, batch_size):
    """Split image file paths into batches."""
    for i in range(0, len(image_files), batch_size):
        yield image_files[i:i + batch_size]
def load_batch_images(image_batch, device):
    """Load and preprocess a batch of images."""
    images = []
    for img_path in image_batch:
        img = cv2.imread(img_path)
        assert img is not None, f"Could not read {img_path}"
        img = torch.from_numpy(img).to(dtype=torch.float32, device=device) / 255
        images.append(img)
    
    # Stack images into a single tensor (batch_size, H, W, C)
    return torch.stack(images)


def main():
    torch.set_printoptions(precision=4, sci_mode=True)

    cmd_args = parse_args()
    device = torch.device("cuda:0")

    torch.set_grad_enabled(False)

    # Get all image file paths from the dataset folder
    dataset_folder = cmd_args.image_file  # Using the argument as a folder path
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    assert image_files, f"No valid image files found in {dataset_folder}"

    TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO, debug=cmd_args.debug, device_memory_GB=0.1)

    batch_size = 4  # Set batch size
    metrics_history = []  # To store metrics for plotting

    # Initialize MLP and optimizer
    ref_image_sample = cv2.imread(image_files[0])
    h, w = ref_image_sample.shape[:2]
    gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(device)
    channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])

    optimizer = mlp(inputs=channels, outputs=channels,
                    hidden_channels=[128, 128, 128],
                    activation=nn.ReLU,
                    norm=partial(nn.LayerNorm, elementwise_affine=False),
                    output_scale=1e-12
                    )
    optimizer.to(device=device)
    optimizer = torch.compile(optimizer)
    optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.0001)

    config = RasterConfig()

    for image_batch in batch_images(image_files, batch_size):
        # Load batch images
        ref_images = load_batch_images(image_batch, device)
        h, w = ref_images.shape[1:3]
        print(f"Training on batch of {len(image_batch)} images, size: {w}x{h}")

        torch.manual_seed(cmd_args.seed)
        torch.cuda.random.manual_seed(cmd_args.seed)

        gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(device)

        trainer = Trainer(optimizer, optimizer_opt, ref_images, config,
                          opacity_reg=cmd_args.opacity_reg, scale_reg=cmd_args.scale_reg)

        epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]
        pbar = tqdm(total=cmd_args.iters)
        iteration = 0

        for epoch_size in epochs:
            metrics = {}
            step_size = log_lerp(min(iteration / 1000.0, 1.0), 0.1, 1.0)
            gaussians, train_metrics = trainer.train_epoch(gaussians, epoch_size=epoch_size, step_size=step_size)
            iteration += epoch_size

            # Render and calculate metrics for the batch
            rendered_images = trainer.render(gaussians).image  # Render the batch
            batch_psnr = [psnr(ref, rendered).item() for ref, rendered in zip(ref_images, rendered_images)]

            # Record average PSNR for the batch
            metrics["CPSNR"] = np.mean(batch_psnr)
            metrics["n"] = gaussians.batch_size[0]
            metrics.update(train_metrics)

            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics[k] = f"{v:.4f}"
                if isinstance(v, int):
                    metrics[k] = f"{v:4d}"

            pbar.set_postfix(**metrics)
            pbar.update(epoch_size)

        save_checkpoint(optimizer, optimizer_opt, metrics_history, iteration, filename="checkpoint_batch.pth")

    print("Training complete.")
main()