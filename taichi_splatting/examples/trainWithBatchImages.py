
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
from taichi_splatting.examples.mlp import mlp,TransformerMLP
from taichi_splatting.misc.renderer2d import project_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians
from fit_image_gaussians import parse_args, log_lerp,lerp,mean_dicts,psnr,display_image
from taichi_splatting.torch_lib.util import check_finite
import os
import matplotlib.pyplot as plt
import torch.nn.utils as nn_utils
def psnr_batch_efficient(batch_a, batch_b):
    """
    Calculate PSNR for a batch of images more efficiently.

    Args:
    - batch_a: Tensor of shape [batch_size, channels, height, width] representing predicted images.
    - batch_b: Tensor of shape [batch_size, channels, height, width] representing reference images.

    Returns:
    - Average PSNR for the batch.
    """
    # Calculate MSE for each image in the batch
    mse = torch.nn.functional.mse_loss(batch_a, batch_b, reduction='none')
    mse_per_image = mse.view(mse.size(0), -1).mean(dim=1)

    # Compute PSNR for each image in the batch
    psnr_per_image = 10 * torch.log10(1 / mse_per_image)

    # Return the average PSNR across the batch
    return psnr_per_image.mean()
def save_checkpoint(optimizer,optimizer_opt, metrics_history,epoch_size, filename="checkpoint.pth"):
    checkpoint = {
        'epoch_size': epoch_size,
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_opt_state_dict': optimizer_opt.state_dict(),
        'metrics': metrics_history,
        
    }
    torch.save(checkpoint, filename)
    # print(f"Checkpoint saved to {filename}")
def element_sizes(t):
  """ Get non batch sizes from a tensorclass"""
 
  return {k:v.shape[2:] for k, v in t.items()}

def split_tensorclass(t, flat_tensor:torch.Tensor):
  
    step =[]
    
    # print(f"flat tensors: {flat_tensor.shape}")
    sizes = element_sizes(t)
    # print(f"element size :{element_sizes}")

    splits = [np.prod(s) for s in sizes.values()]
    # print(f"split : {splits}")
    tensors = torch.split(flat_tensor, splits, dim=2)
    # print(f"tensor {tensors[0].shape}")
    # print(t.batch_size)
    return t.__class__.from_dict(
    {
        k: v.view(t.batch_size + s)  # Reshape tensor `v`
        for k, v, s in zip(sizes.keys(), tensors, sizes.values())  # Iterate over field names, tensors, and their sizes
    },
    batch_size=t.batch_size)  # Set batch size for the new tensorclass)


    



def batch_images(image_files, batch_size):
    """Split image file paths into batches."""
    for i in range(0, len(image_files), batch_size):
        # print(i)
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
    return torch.stack(images,dim=0)

def initialize_gaussians(batch_size, image_size, n_gaussians, device):
    """
    Initialize random Gaussians for a batch, ensuring each batch entry has unique parameters.
    """
    gaussians = []
    for _ in range(batch_size):
        # Randomly generate Gaussians for each batch entry
        single_gaussian = random_2d_gaussians(n_gaussians, image_size, alpha_range=(0.5, 1.0), scale_factor=1.0)
        gaussians.append(single_gaussian)

    # Stack into a single tensor for batched processing
    batched_gaussians = torch.stack(gaussians).to(device)  # Shape: [batch_size, n_gaussians, feature_size]
    return batched_gaussians



def flatten_tensorclass(t):
#   print(f"flatten Tensor : {t}")
  flat_tensor = torch.cat([v.view(v.shape[0],v.shape[1], -1) for v in t.values()], dim=2)
  return flat_tensor

class Trainer:
    def __init__(self, 
                optimizer_mlp:torch.nn.Module, 
                mlp_opt:torch.optim.Optimizer, 
                ref_image:torch.Tensor,
                config:RasterConfig, 
                opacity_reg=0.0, 
                scale_reg=0.0):
        
        self.optimizer_mlp = optimizer_mlp
        self.mlp_opt = mlp_opt
        self.config = config
        self.opacity_reg = opacity_reg
        self.scale_reg = scale_reg
        self.ref_image:torch.Tensor = ref_image
        self.running_scales = None

    def setRefImage(self,refimage):
        self.ref_image = refimage

    def render(self, gaussians_batch):
        """Render a batch of images."""
        rendered_batch = []
        # print(f"render : {gaussians_batch}")
        h, w = self.ref_image.shape[1:3]
        for gaussians in gaussians_batch:
            gaussians2d = project_gaussians2d(gaussians)
            raster = rasterize(
                gaussians2d=gaussians2d,
                depth=gaussians.z_depth.clamp(0, 1),
                features=gaussians.feature,
                image_size=(w, h),
                config=self.config,
            )
            rendered_batch.append(raster.image)
        
        return torch.stack(rendered_batch)  # Shape: [batch_size, H, W, C]



    def render_step(self, gaussians_batch):
        """Compute the loss for a batch of images."""
        with torch.enable_grad():
            # print(f"render_step : {gaussians_batch}")
            rendered_images = self.render(gaussians_batch)
            display_image('rendered', rendered_images[0])
            batch_loss = 0
            for rendered_image, ref_image in zip(rendered_images, self.ref_image):
                # display_image('rendered', rendered_image)
                h, w = self.ref_image[0].shape[:2]
                scale = torch.exp(gaussians_batch.log_scaling) / min(w, h)
                # print(f"scale: {scale}")
                opacity_reg = self.opacity_reg * gaussians_batch.opacity.mean()
                scale_reg = self.scale_reg * scale.pow(2).mean()
                depth_reg = 0.0 * gaussians_batch.z_depth.sum()
                l1 = torch.nn.functional.l1_loss(rendered_image, ref_image)
                loss  = l1 + opacity_reg*2 + scale_reg*2 + depth_reg
                
                batch_loss = batch_loss+ loss
            # batch_loss  = batch_loss
            # print(len(gaussians_batch))
            batch_loss.backward()
        
            # Average loss over the batch
        return dict(loss=batch_loss.item(), opacity_reg=opacity_reg.item(), scale_reg=scale_reg.item())


    def get_gradients(self,  gaussian_):
        # print(f"get_gradient : {gaussian_}")
        gaussian_batch =  gaussian_.clone()
        gaussian_batch.requires_grad_(True)
        metrics = self.render_step( gaussian_batch)
        # print(f"gaussian_batch grad: {gaussian_batch.grad}")
        grad = gaussian_batch.grad
        

        mean_abs_grad = grad.abs().mean(dim=0)

        if self.running_scales is None:
            self.running_scales = mean_abs_grad
        else:
            self.running_scales = lerp(0.999, self.running_scales, mean_abs_grad)


        return grad * 1e5, metrics

    def train_epoch(self, gaussians_batch, step_size=10, epoch_size=100):
        metrics = []
        for _ in range(epoch_size):
            self.mlp_opt.zero_grad()
            # print(f"train_epoch : {gaussians_batch}")
            # Compute the batch loss
            grads,_ = self.get_gradients(gaussians_batch)
            check_finite(grads, "grad")
            inputs = flatten_tensorclass(grads)
            # print(f"inputs:{inputs.shape}")
            
            with torch.enable_grad():
                step = self.optimizer_mlp(inputs)
                finite_mask = torch.isfinite(step)
                step = torch.where(finite_mask, step, torch.zeros_like(step))
                step = split_tensorclass(gaussians_batch, step)
                
                # print(f"step2: {gaussians_batch - step}")
                metrics.append(self.render_step(gaussians_batch - step))
            # Update Gaussians
            gaussians_batch = gaussians_batch - step * step_size

            nn_utils.clip_grad_norm_(self.optimizer_mlp.parameters(), 1.0)
            
            self.mlp_opt.step()

            
        return gaussians_batch, mean_dicts(metrics)
    def test(self, gaussians,step_size=0.01,epoch_size = 100):
      
        """Run inference using the trained model."""
        metrics = []
        for i in range(epoch_size):
            with torch.enable_grad():
                    # Compute gradients
                grad,metric = self.get_gradients(gaussians)
                check_finite(grad, "grad")

                # Flatten gradients and predict updates using the MLP
                inputs = flatten_tensorclass(grad)
                # input2 = flatten_tensorclass(gaussians)
                # inputs = torch.cat([inputs, input2], dim=1)

                with torch.no_grad():
                    step = self.optimizer_mlp(inputs)
                    step = split_tensorclass(gaussians, step)
                    metrics.append(metric)
                # Update Gaussians with the step
                gaussians = gaussians - step * step_size
        return gaussians, mean_dicts(metrics)
   

def main():
    cmd_args = parse_args()
    device = torch.device("cuda:0")

    torch.set_grad_enabled(False)

    # Load all image paths
    dataset_folder = cmd_args.image_file  # Using the argument as a folder path
    image_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    assert image_files, f'No valid image files found in {dataset_folder}'

    batch_size = cmd_args.batch_size
    batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    

    # Initialize Gaussians and Trainer
    ref_image = cv2.imread(image_files[0])
    # assert ref_image is not None, f'Could not read {'/csse/users/pwl25/pear/images/DSC_1366_12kv2r16k_7.jpg'}'

    h, w = ref_image.shape[:2]  # Replace with your image dimensions
    TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=cmd_args.debug, device_memory_GB=0.1)
    n_gaussians = cmd_args.n
    sample_gaussian = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(torch.device('cuda:0'))
    
    # Create MLP and optimizer
    channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in sample_gaussian.items()])
    # print(f"channels: {channels}")
    

    # Create the MLP
    optimizer = mlp(inputs=channels, outputs=channels,
                    hidden_channels=[128, 256, 512,1024,128],
                    activation=nn.ReLU,
                    norm=partial(nn.LayerNorm, elementwise_affine=False),
                    output_scale=1e-12
                    )
    # optimizer = TransformerMLP(input_dim=channels, output_dim=channels,
    #                 hidden_channels=[128],
    #                 num_heads=1, num_layers=2
    #                 )
    optimizer.to(device=device)
    optimizer = torch.compile(optimizer)
    optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.001)# mlp = 0.001  transformer = 0.01
    epochs = [cmd_args.epoch for _ in range(cmd_args.iters // cmd_args.epoch)]
    config = RasterConfig()
    trainer = Trainer(optimizer, optimizer_opt, None, config,opacity_reg=cmd_args.opacity_reg, scale_reg=cmd_args.scale_reg)
    pbar_batch = tqdm(total=len(batches),desc="Overall Progress")
    batch_i = 0
    for batch_i, batch_files in enumerate(batches, start=1):  
        trainer.setRefImage(load_batch_images(batch_files, device))
        pbar = tqdm(total=cmd_args.iters,desc=f"Batch {batch_i}/{len(batches)}")
        iteration = 0
        gaussians_batch = initialize_gaussians(batch_size, (w, h), n_gaussians, device)
        # print(gaussians_batch.shape)
        for  epoch_size in epochs:
            metrics = {}
            step_size = log_lerp(min(iteration / 100000., 1.0), 0.1, 1.0)
            gaussians_batch, train_metrics = trainer.train_epoch(gaussians_batch, step_size, epoch_size)
            iteration += epoch_size
            image = trainer.render(gaussians_batch)
            # print(image.shape)
            # if cmd_args.show:
            #     display_image('rendered', image)
            metrics['CPSNR'] = psnr_batch_efficient(trainer.ref_image, image).item()
            metrics['n'] = gaussians_batch.batch_size[0]
            metrics.update(train_metrics)
            for k, v in metrics.items():
                if isinstance(v, float):
                    metrics[k] = f'{v:.4f}'
                if isinstance(v, int):
                    metrics[k] = f'{v:4d}'

            pbar.set_postfix(**metrics)
            iteration += epoch_size
            pbar.update(epoch_size)
            save_checkpoint(optimizer,optimizer_opt, metrics,batch_i, filename="play.pth")
        
            iteration += epoch_size
        pbar.close()

    # Update the overall progress bar
        pbar_batch.update(1)

        # print(f"Metrics: {metrics}")
if __name__ == "__main__":
  main()
