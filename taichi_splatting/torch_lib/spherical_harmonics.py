import math
from beartype import beartype
import torch

from . import rsh

rsh_cart_n = [rsh.rsh_cart_0, rsh.rsh_cart_1, rsh.rsh_cart_2, rsh.rsh_cart_3]
              
def check_sh_degree(sh_features):
  n_sh = sh_features.shape[2]
  n = int(math.sqrt(n_sh))

  assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
  return (n - 1)

@beartype
def evaluate_sh(params:torch.Tensor,   # N, K, D where D = (degree + 1)^2
                dirs:torch.Tensor,     # N, 3
                ) -> torch.Tensor:     # N, K

  degree = check_sh_degree(params)
  assert degree <= 3 and degree >= 0, f"SH degree must be between 0 and 3, got {degree}"

  rsh_cart = rsh_cart_n[degree]
  coeffs = rsh_cart(dirs) # N, D
 
  out = torch.einsum('nd,nkd->nk', coeffs, params)
  return torch.clamp(out + 0.5, 0., 1.)


@beartype
def evaluate_sh_at(params:torch.Tensor,  # M, K, D where D = (degree + 1)^2
                points:torch.Tensor,     # M, 3
                
                indexes:torch.Tensor,    # N
                camera_pos:torch.Tensor # 3
                ) -> torch.Tensor:    # N, K

  dirs = camera_pos.unsqueeze(0) - points[indexes]
  dirs = torch.nn.functional.normalize(dirs, dim=1)

 
  return evaluate_sh(params[indexes], dirs)

if __name__ == "__main__":
    dimension = 5
    degree = 3

    params = torch.rand(100, dimension, (degree + 1)**2).requires_grad_(True)
    dirs = torch.rand(100, 3)

    out = evaluate_sh(params, dirs)
    print(out.shape)


