from dataclasses import dataclass, replace
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import torch


@beartype
@dataclass
class CameraParams:
  T_image_camera: torch.Tensor # (3, 3) camera projection matrix
  T_camera_world  : torch.Tensor # (4, 4) camera view matrix

  near_plane: float
  far_plane: float
  image_size: Tuple[Integral, Integral]

  blur_cov:float = 0.3

  @property
  def device(self):
    return self.T_image_camera.device

  @property
  def T_image_world(self):
    T_image_camera = torch.eye(4, 
      device=self.T_image_camera.device, dtype=self.T_image_camera.dtype)
    T_image_camera[0:3, 0:3] = self.T_image_camera
    return T_image_camera @ self.T_camera_world
  
  def requires_grad_(self, requires_grad: bool):
    self.T_image_camera.requires_grad_(requires_grad)
    self.T_camera_world.requires_grad_(requires_grad)
    return self


  def __repr__(self):
    w, h = self.image_size
    fx, fy = self.T_image_camera[0, 0], self.T_image_camera[1, 1]
    cx, cy = self.T_image_camera[0, 2], self.T_image_camera[1, 2]

    pos_str = ", ".join([f"{x:.3f}" for x in self.camera_position])
    return f"CameraParams({w}x{h}, fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}, clipping={self.near_plane:.4f}-{self.far_plane:.4f}, position=({pos_str})"
  

  @property
  def camera_position(self):
    T_world_camera = torch.inverse(self.T_camera_world)
    return T_world_camera[0:3, 3]
  
  def scale_image(self, scale: float):
    image_size = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))
    scaling = torch.tensor([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], device=self.T_image_camera.device, dtype=self.T_image_camera.dtype)

    return replace(self, image_size=image_size, T_image_camera=scaling @ self.T_image_camera)


  def to(self, device=None, dtype=None):
    return CameraParams(
      T_image_camera=self.T_image_camera.to(device=device, dtype=dtype),
      T_camera_world=self.T_camera_world.to(device=device, dtype=dtype),
      near_plane=self.near_plane,
      far_plane=self.far_plane,
      image_size=self.image_size,
      blur_cov=self.blur_cov
    )

  def __post_init__(self):
    assert self.T_image_camera.shape == (3, 3), f"Expected shape (3, 3), got {self.T_image_camera.shape}"
    assert self.T_camera_world.shape == (4, 4), f"Expected shape (4, 4), got {self.T_camera_world.shape}"

    assert len(self.image_size) == 2
    assert self.near_plane > 0
    assert self.far_plane > self.near_plane