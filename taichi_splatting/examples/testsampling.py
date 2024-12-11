import torch
import numpy as np
import taichi as ti
from taichi_splatting.data_types import RasterConfig
from renderer2d import Gaussians2D
from gaussian_mixer import GaussianMixer
from taichi_splatting.tests.random_data import random_2d_gaussians
from taichi_splatting.taichi_queue import TaichiQueue
def test_render_and_sample():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    TaichiQueue.init(arch=ti.cuda,
                     log_level=ti.INFO,
                     debug=None,
                     device_memory_GB=0.1)
    # Initialize Gaussian Mixer
    n_inputs = 16  # 16D feature vector
    mixer = GaussianMixer(inputs=n_inputs, outputs=n_inputs, n_render=16, n_base=128).to(device)

    # Define Gaussian
    image_size = (128, 128)  # Small image for testing
     # Random 16D feature vector

    # Create a single Gaussian
    gaussians = random_2d_gaussians(3000, image_size,
                                    alpha_range=(0.5, 1.0),
                                    scale_factor=1.0).to(
                                        torch.device('cuda:0'))
    features = torch.randn(gaussians.batch_size[0], 10).to(device)
    # Render the Gaussian
    config = RasterConfig()
    rendered_image = mixer.render(features.to(torch.float32), gaussians, image_size, config)  # (n_render, H, W)

    # Sample back at the center of the Gaussian
    sampled_features = mixer.sample_positions(rendered_image, gaussians.position)  # (n_render,)

    # Validate that the sampled features are close to the input features
    print(f"Original features:\n{features}")
    print(f"Sampled features:\n{sampled_features}")
    print(f"Difference:\n{features - sampled_features}")
    is_close = torch.allclose(features, sampled_features, atol=1e-2)
    assert is_close, f"Sampled features {sampled_features} are not close to input features {features}"

    print("Test passed: Rendered and sampled features are close.")
    
if __name__ == "__main__":
    test_render_and_sample()
