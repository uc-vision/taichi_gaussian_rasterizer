from dataclasses import dataclass, replace
from numbers import Integral
from beartype.typing import Tuple
from beartype import beartype
import torch


@beartype
@dataclass
class CameraParams:

  projection: torch.Tensor # (4) - [fx, fy, cx, cy]
  T_camera_world  : torch.Tensor # (4, 4) camera view matrix

  near_plane: float
  far_plane: float
  image_size: Tuple[Integral, Integral]


  @property
  def depth_range(self):
    return (self.near_plane, self.far_plane)


  @property
  def device(self):
    return self.projection.device
  
  @property
  def dtype(self):
    return self.projection.dtype
  
  @property
  def T_image_camera(self):
    fx, fy, cx, cy = self.projection
    m = [[fx, 0, cx], 
        [0, fy, cy], 
        [0, 0, 1]]
    return torch.tensor(m, device=self.device, dtype=self.dtype)

  @property
  def focal_length(self):
    return self.projection[0:2]
  
  @property
  def principal_point(self):
    return self.projection[2:4]

  @property
  def T_image_world(self):
    T_image_camera = torch.eye(4, 
      device=self.T_image_camera.device, dtype=self.dtype)
    T_image_camera[0:3, 0:3] = self.T_image_camera
    return T_image_camera @ self.T_camera_world
  
  def requires_grad_(self, requires_grad: bool):
    self.projection.requires_grad_(requires_grad)
    self.T_camera_world.requires_grad_(requires_grad)
    return self

  def detach(self):
    return replace(self, projection=self.projection.detach(), T_camera_world=self.T_camera_world.detach())

  def __repr__(self):
    w, h = self.image_size
    fx, fy, cx, cy = self.projection.detach().cpu().numpy()

    pos_str = ", ".join([f"{x:.3f}" for x in self.camera_position])
    return f"CameraParams({w}x{h}, fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}, clipping={self.near_plane:.4f}-{self.far_plane:.4f}, position=({pos_str})"
  

  @property
  def camera_position(self):
    T_world_camera = torch.inverse(self.T_camera_world)
    return T_world_camera[0:3, 3]
  
  def scale_image(self, scale: float):
    image_size = (int(self.image_size[0] * scale), int(self.image_size[1] * scale))

    return replace(self, image_size=image_size, projection=self.projection * scale)


  def to(self, device=None, dtype=None):
    return CameraParams(
      projection=self.projection.to(device=device, dtype=dtype),
      T_camera_world=self.T_camera_world.to(device=device, dtype=dtype),
      near_plane=self.near_plane,
      far_plane=self.far_plane,
      image_size=self.image_size
    )

  def __post_init__(self):
    assert self.projection.shape == (4, ), f"Expected shape (4,), got {self.projection.shape}"
    assert self.T_camera_world.shape == (4, 4), f"Expected shape (4, 4), got {self.T_camera_world.shape}"


    assert len(self.image_size) == 2
    assert self.near_plane > 0
    assert self.far_plane > self.near_plane