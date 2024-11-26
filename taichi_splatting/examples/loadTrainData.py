import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import torch.nn as nn
import cv2
from taichi_splatting.examples.mlp import mlp
from taichi_splatting.rasterizer.function import rasterize
from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from fit_image_gaussians import parse_args, Trainer, psnr,partial,log_lerp,psnr,display_image
from taichi_splatting.tests.random_data import random_2d_gaussians
from functools import partial
import math
from typing import Dict, List
import cv2
import argparse
import numpy as np
import torch.nn as nn
import taichi as ti
from torchviz import make_dot
import torch
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians2D, RasterConfig
from taichi_splatting.examples.mlp import mlp
from taichi_splatting.misc.renderer2d import project_gaussians2d

from taichi_splatting.rasterizer.function import rasterize

from taichi_splatting.taichi_queue import TaichiQueue
from taichi_splatting.tests.random_data import random_2d_gaussians

from taichi_splatting.torch_lib.util import check_finite


def main():
    torch.set_printoptions(precision=4, sci_mode=True)

    cmd_args = parse_args()
    device = torch.device('cuda:0')

    torch.set_grad_enabled(False)

    # Path to the saved model and optimizer
    PATH = 'checkpoint2.pth'  # Replace with your checkpoint file
    test_image_path = '/csse/users/pwl25/pear/images/DSC_1356_12kv2r5k_11.jpg'  # Replace with your test image path
    ref_image = cv2.imread(test_image_path)
    assert ref_image is not None, f"Could not read image {test_image_path}"

    # Preprocess the image (resize, normalize, etc.)
    h, w = ref_image.shape[:2]
    TaichiQueue.init(arch=ti.cuda, log_level=ti.INFO,  
          debug=cmd_args.debug, device_memory_GB=0.1)
    torch.manual_seed(cmd_args.seed)
    torch.cuda.random.manual_seed(cmd_args.seed)
    # Load the model and optimizer
    gaussians = random_2d_gaussians(cmd_args.n, (w, h), alpha_range=(0.5, 1.0), scale_factor=1.0).to(torch.device('cuda:0')) 
    channels = sum([np.prod(v.shape[1:], dtype=int) for k, v in gaussians.items()])

    optimizer = mlp(inputs = channels, outputs=channels, 
              hidden_channels=[128, 128, 128], 
              activation=nn.ReLU,
              norm=partial(nn.LayerNorm, elementwise_affine=False),
              # output_activation=nn.Tanh,
              output_scale=1e-12
              )
    optimizer.to(device=device)
    optimizer = torch.compile(optimizer)
    optimizer_opt = torch.optim.Adam(optimizer.parameters(), lr=0.0001)
     
    

    # Load weights and optimizer states
    checkpoint = torch.load(PATH, weights_only=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    optimizer_opt.load_state_dict(checkpoint['optimizer_opt_state_dict'])
    epoch_size = checkpoint['epoch_size']
    
    metrics = checkpoint['metrics']


    # Set the model to evaluation mode

    # Load an image for testing
    
    ref_image = torch.from_numpy(ref_image).to(dtype=torch.float32, device=device) / 255 
    config = RasterConfig()

    trainer = Trainer(optimizer, optimizer_opt, ref_image, config, 
                        opacity_reg=cmd_args.opacity_reg, scale_reg=cmd_args.scale_reg)
    # rendered_image = trainer.test(gaussians)

    # # Display or save the rendered image
    # rendered_image_np = (rendered_image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    # output_path = "rendered_image.png"
    # cv2.imwrite(output_path, rendered_image_np)
    # print(f"Rendered image saved to {output_path}")

    # # Optionally display the image
    # display_image('Rendered Image', rendered_image)
    # loss = torch.nn.functional.l1_loss(rendered_image, ref_image)
    # graph = make_dot(loss, params=dict(optimizer.named_parameters()))
    # graph.render("fit_image_computation_graph", format="png")
    
    pbar = tqdm(total=cmd_args.iters)
    metrics = {}
    gaussians,train_metrics = trainer.test(gaussians)
        
    image = trainer.render(gaussians).image
    if cmd_args.show:
        display_image('rendered', image)
    output_path = "rendered_image.png"
    rendered_image_np = (image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    cv2.imwrite(output_path, rendered_image_np)
    metrics['CPSNR'] = psnr(ref_image, image).item()
    metrics['n'] = gaussians.batch_size[0]
    # metrics.update(train_metrics)
    pbar.set_postfix(**metrics)
        # Record metrics for plotting
    # metrics['CPSNR'] = psnr(ref_image, image).item()
    # metrics['n'] = gaussians.batch_size[0]
        # metrics.update(train_metrics)
    
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")


    # for k, v in metrics.items():
    #     if isinstance(v, float):
    #         metrics[k] = f'{v:.4f}'
    #     if isinstance(v, int):
    #         metrics[k] = f'{v:4d}'

    # pbar.set_postfix(**metrics)
if __name__ == "__main__":
  main()
