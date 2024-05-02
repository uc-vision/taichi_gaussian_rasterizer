from typing import Callable, Tuple
from tqdm import tqdm
from taichi_splatting.data_types import Gaussians3D
from taichi_splatting.camera_params import CameraParams

from taichi_splatting.tests.util import eval_with_grad
from taichi_splatting.torch_lib.util import check_finite
import taichi_splatting.conic.perspective.projection as ti_proj
from taichi_splatting.testing import random_camera, random_3d_gaussians

import taichi_splatting.conic.perspective.torch_projection  as torch_proj


import torch
import taichi as ti

from torch.autograd.gradcheck import GradcheckError


ti.init(arch=ti.cpu, offline_cache=True, log_level=ti.INFO, debug=True)


def random_inputs(device='cpu', max_points=1000, dtype=torch.float32) -> Callable[[], Tuple[Gaussians3D, CameraParams]]:
  def f(seed:int = 0) -> Tuple[Gaussians3D, CameraParams]:
    torch.manual_seed(seed)
    camera = random_camera()
    n = torch.randint(size=(1,), low=1, high=max_points).item()

    gaussians = random_3d_gaussians(n=n, camera_params=camera)
    check_finite(gaussians, 'gaussians')

    return (x.to(device=device, dtype=dtype).requires_grad_(True)
             for x in [gaussians, camera])
  return f


def compare(name, x, y, **kwargs):
  if not torch.allclose(x, y, **kwargs):
    print(f"x={x}")
    print(f"y={y}")

    atol = (x - y).abs().max().item()

    raise AssertionError(f"{name} mismatch with atol={atol}")

def compare_outputs(out1, out2):
  points1, depth1 = out1
  points2, depth2 = out2

  compare("uv",  points1[:, 0:2], points2[:, 0:2])
  compare("conic",  points1[:, 2:5], points2[:, 2:5])
  compare("alpha",  points1[:, 5], points2[:, 5])
  compare("depth",  depth1, depth2)

def compare_grads(grad1, grad2):
  
  position1, log_scaling1, rotation1, alpha_logit1, _, image_camera1, camera_world1 = grad1
  position2, log_scaling2, rotation2, alpha_logit2, _, image_camera2, camera_world2 = grad2

  compare("position grad", position1, position2)
  compare("log_scaling grad", log_scaling1, log_scaling2)
  compare("rotation grad", rotation1, rotation2)
  compare("alpha_logit grad", alpha_logit1, alpha_logit2)

  compare("image_camera grad", image_camera1, image_camera2)
  compare("camera_world grad", camera_world1, camera_world2)

def test_projection(iters = 100, dtype=torch.float64):
  gen_inputs = random_inputs(max_points=1000, dtype=dtype)

  for i in tqdm(range(iters)):
    gaussians, camera = gen_inputs(i)
    inputs = (*gaussians.shape_tensors(), 
      torch.arange(gaussians.batch_size[0], device=gaussians.device), 
      camera.T_image_camera, camera.T_camera_world)

    out1, grad1 = eval_with_grad(ti_proj.apply, *inputs)
    out2, grad2 = eval_with_grad(torch_proj.apply, *inputs)

    try:
      compare_outputs(out1, out2)
      compare_grads(grad1, grad2)

    except AssertionError as e:
      print(f"seed={i}")
      print(f"camera={camera}")
      raise e

def test_projection_grad(iters = 100):
  gen_inputs = random_inputs(max_points=10, dtype=torch.float64)

  for i in tqdm(range(iters), desc="projection_gradcheck"):
      gaussians, camera = gen_inputs(i)
      indexes =  torch.arange(gaussians.batch_size[0], device=gaussians.device)

      inputs = [*gaussians.shape_tensors(), indexes, 
                camera.T_image_camera, camera.T_camera_world]
      
      try:
        torch.autograd.gradcheck(ti_proj.apply, inputs)
      except GradcheckError as e:
        print(f"seed={i}")
        print(f"camera={camera}")
        raise e

if __name__ == '__main__':
  torch.set_printoptions(precision=8, sci_mode=False)

  test_projection()
  test_projection_grad()
