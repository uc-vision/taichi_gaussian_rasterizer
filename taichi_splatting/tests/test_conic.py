import torch

import taichi as ti
from tqdm import tqdm

from taichi_splatting.conic.perspective.torch_projection import cov_to_conic
from taichi_splatting.taichi_lib.f64 import (
  conic_pdf_with_grad, conic_pdf, vec2, vec3)

from taichi_splatting.tests.util import compare_with_grad

import warnings
warnings.filterwarnings('ignore') 


ti.init(debug=True)

def torch_conic_pdf(xy:torch.Tensor, uv:torch.Tensor, uv_conic:torch.Tensor) -> torch.Tensor:
    # detach gradient as the taichi code doesn't generate gradients for xy
    dx, dy = (xy.detach() - uv).T
    a, b, c = uv_conic.T
    return torch.exp(-0.5 * (dx**2 * a + dy**2 * c) - dx * dy * b)



@ti.kernel
def kernel_conic_pdf_grad(
   xy : ti.types.ndarray(vec2, ndim=1),
   uv : ti.types.ndarray(vec2, ndim=1),
   uv_conic : ti.types.ndarray(vec3, ndim=1),

   dp_duv : ti.types.ndarray(vec2, ndim=1),
   dp_dconic : ti.types.ndarray(vec3, ndim=1),
   beta: ti.template()):

   for i in range(uv.shape[0]):
      _, grad_uv, grad_conic = conic_pdf_with_grad(xy[i], uv[i], uv_conic[i], beta)
      dp_duv[i] = grad_uv
      dp_dconic[i] = grad_conic

@ti.kernel
def kernel_conic_pdf(
   xy : ti.types.ndarray(vec2, ndim=1),
   uv : ti.types.ndarray(vec2, ndim=1),
   uv_conic : ti.types.ndarray(vec3, ndim=1),

   out_p : ti.types.ndarray(ti.f64, ndim=1),
   beta: ti.template()):

   for i in range(uv.shape[0]):
      p = conic_pdf(xy[i], uv[i], uv_conic[i], beta)
      out_p[i] = p
           

class ConicPdf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xy, uv, uv_conic, beta=1.0):
        p = torch.zeros_like(xy[:, 0])
        ctx.beta = beta
        ctx.save_for_backward(xy, uv, uv_conic, p)
        kernel_conic_pdf(xy, uv, uv_conic, p, beta)
        return p

    @staticmethod
    def backward(ctx, grad_p):
        xy, uv, uv_conic, _ = ctx.saved_tensors
        grad_uv = torch.zeros_like(uv)
        grad_conic = torch.zeros_like(uv_conic)
        kernel_conic_pdf_grad(xy, uv, uv_conic, grad_uv, grad_conic, ctx.beta)

        grad_p = grad_p.unsqueeze(1)
        return None, grad_uv * grad_p, grad_conic * grad_p


def random_inputs(n, device='cpu', dtype=torch.float64):
    def f(seed):
      torch.random.manual_seed(seed)

      dx = torch.randn(n, 2, device=device, dtype=dtype)
      uv = torch.rand(n, 2, device=device, dtype=dtype) * 100

      m = 10 * torch.randn(n, 2, 2, device=device, dtype=dtype)
      cov = m @ m.transpose(-1, -2)

      conic = cov_to_conic(cov)

      # No gradient on xy as conic_pdf_with_grad doesn't provide it
      return (uv + dx), uv.requires_grad_(True), conic.exp().requires_grad_(True)
    return f

def test_conic():
  compare_with_grad("conic_pdf", ["xy", "uv", "uv_conic"], "p", 
        ConicPdf.apply, torch_conic_pdf, random_inputs(10), iters=100)

def test_conic_gradcheck(iters = 100, device='cpu'):
  make_inputs = random_inputs(10)

  seeds = torch.randint(00, 10000, (iters, ), device=device)
  for seed in tqdm(seeds, desc="conic_gradcheck"):
      inputs = make_inputs(seed)
      torch.autograd.gradcheck(ConicPdf.apply, inputs)


if __name__ == '__main__':
  test_conic()
  test_conic_gradcheck()
  
