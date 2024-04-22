import torch
import taichi as ti
from tqdm import tqdm
from taichi_splatting.tests.util import compare_with_grad

from taichi_splatting.torch_lib import spherical_harmonics as torch_sh
from taichi_splatting import spherical_harmonics as taichi_sh

import warnings
warnings.filterwarnings('ignore') 


ti.init(debug=True)

def random_inputs(max_dim=3, max_deg=3, max_n=100, device='cpu', dtype=torch.float32):
  def f(seed):
    torch.random.manual_seed(seed)
    dimension = torch.randint(1, max_dim + 1, (1,)).item()
    degree = torch.randint(1, max_deg + 1, (1, )).item()
    n = torch.randint(1, max_n + 2, (1,) ).item()

    params = torch.rand(n, dimension, (degree + 1)**2, device=device, dtype=dtype)

    points = torch.randn(n, 3, device=device, dtype=dtype)
    camera_pos = torch.randn(3, device=device, dtype=dtype)

    indexes = torch.randint(0, n, (n  // 2, ), device=device)

    return (params.requires_grad_(True), points.requires_grad_(True), 
            indexes, camera_pos.requires_grad_(True))
  return f

def test_sh(iters = 100, device='cpu'):
  make_inputs = random_inputs(max_n=100, device=device, dtype=torch.float32)
  compare_with_grad("spherical_harmonics", 
    input_names=["params", "dirs", "camera_pos"], 
    output_names="out",
    f1=taichi_sh.evaluate_sh_at,
    f2=torch_sh.evaluate_sh_at,
    gen_inputs=make_inputs,
    iters=iters)
  



def gradcheck(func, args, **kwargs):
  args = [x.detach().requires_grad_(x.requires_grad) for x in args]
  torch.autograd.gradcheck(func, args, **kwargs)

def test_sh_gradcheck(iters = 100, device='cpu'):
  make_inputs = random_inputs(max_dim=2, max_n=10, device=device, dtype=torch.float64)

  seeds = torch.randint(40, 10000, (iters, ), device=device)
  for seed in tqdm(seeds, desc="spherical_harmonics_gradcheck"):
      inputs = make_inputs(seed)
      gradcheck(taichi_sh.evaluate_sh_at, inputs)

if __name__ == '__main__':
  test_sh()
  test_sh_gradcheck()