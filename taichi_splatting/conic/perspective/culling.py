from beartype import beartype
import taichi as ti
from taichi.math import mat4
import torch


from taichi_splatting.taichi_lib.f32 import project_perspective
from .params import CameraParams
 

@ti.kernel
def frustum_culling_kernel(
    positions: ti.types.ndarray(ti.math.vec3, ndim=1),  # (N, 11)

    T_image_world: ti.types.ndarray(mat4, ndim=1),  # (1, 4, 4)
    output_mask: ti.types.ndarray(ti.u1, ndim=1),  # (N), output
    
    near_plane: ti.f32,
    far_plane: ti.f32,

    image_size: ti.math.ivec2,
    margin_pixels: ti.i32,
):    
    # filter points in camera
    for point_id in range(positions.shape[0]):
        pixel, depth = project_perspective(
            position=positions[point_id],
            T_image_world=T_image_world[0],
        )

        output_mask[point_id] = (depth > near_plane and 
            depth < far_plane and 
            pixel.x >= -margin_pixels and pixel.x < image_size.x + margin_pixels and 
                pixel.y >= -margin_pixels and pixel.y < image_size.y + margin_pixels)

@beartype
def frustum_culling(positions: torch.Tensor, camera_params: CameraParams, margin_pixels: int = 48):
  mask = torch.empty(positions.shape[0], dtype=torch.bool, device=positions.device)

  frustum_culling_kernel(
    positions=positions.contiguous(),
    T_image_world=camera_params.T_image_world.unsqueeze(0),
    output_mask=mask,

    near_plane=camera_params.near_plane,
    far_plane=camera_params.far_plane,
    image_size=ti.math.ivec2(camera_params.image_size),
    
    margin_pixels=margin_pixels
  )

  return mask


