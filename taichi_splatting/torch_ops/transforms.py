from beartype.typing import Tuple
import torch


def split_rt(
    transform: torch.Tensor,  # (batch_size, 4, 4)
) -> Tuple[torch.Tensor, torch.Tensor]:
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    return R.contiguous(), t.contiguous()

def join_rt(r, t):
  T = torch.eye(4, device=r.device, dtype=r.dtype)
  T[0:3, 0:3] = r
  T[0:3, 3] = t
  return T
  

def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=points.dtype, device=points.device)], axis=-1)

def transform44(transform, points):

  points = points.reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ points
  return transformed[..., 0].reshape(-1, 4)



def transform33(transform, points):

  points = points.reshape([-1, 3, 1])
  transformed = transform.reshape([1, 3, 3]) @ points
  return transformed[..., 0].reshape(-1, 3)


