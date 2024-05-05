import cv2
import numpy as np
import torch

def numpy_image(image):
  return (image.detach().clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

def display_image(name, image):
    
    if isinstance(image, torch.Tensor):
      image = numpy_image(image)

    cv2.imshow(name, image)
    cv2.waitKey(0)


def draw_corners(image, corners, color=(0, 255, 0)):
    if isinstance(corners, torch.Tensor):
      corners = corners.cpu().numpy()

    corners = corners.astype(np.int32)

    for c in corners: 
      for j in range(4):
        a, b = c[j], c[(j + 1) % 4]
        cv2.line(image, tuple(a), tuple(b), color)

    return image


def draw_aabb(image:np.ndarray, uv:torch.Tensor, radii:torch.Tensor, color=(0, 255, 0)):
  for p, r in zip(uv, radii):
    p = p.cpu().numpy()
    r = int(r.item())

    corners = p.reshape(1, -1) + np.array([[-r, -r], [r, -r], [r, r], [-r, r]])
    draw_corners(image, corners, color)

  return image


def obb_corners(obb:torch.Tensor):

  uv = obb[:, 0:2]
  x,y  = obb[:, 2:4], obb[:, 4:6]
  corners = torch.stack([uv - x - y, uv + x - y, uv + x + y, uv - x + y], dim=1)

  return corners

def draw_obb(image:np.ndarray, obb:torch.Tensor, color=(0, 255, 0)):
  corners = obb_corners(obb)
  draw_corners(image, corners, color)

  return image