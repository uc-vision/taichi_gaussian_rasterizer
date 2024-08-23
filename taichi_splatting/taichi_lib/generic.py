from types import SimpleNamespace
import taichi as ti

from taichi_splatting.taichi_lib.conversions import struct_size

def make_library(dtype=ti.f32):
  """
  This function returns a namespace containing all the functions and data types
  that are used in the other modules. This is done to provide different precisions
  for the same code. Primarily for enabling gradient (gradcheck) testing using f64.
  """

  vec1 = ti.types.vector(1, dtype)
  vec2 = ti.types.vector(2, dtype)
  vec3 = ti.types.vector(3, dtype)
  vec4 = ti.types.vector(4, dtype)

  mat2 = ti.types.matrix(2, 2, dtype)
  mat3 = ti.types.matrix(3, 3, dtype)
  mat4 = ti.types.matrix(4, 4, dtype)

  mat3x4 = ti.types.matrix(3, 4, dtype)
  mat4x2 = ti.types.matrix(4, 2, dtype=dtype)

  #
  # Gaussian datatypes
  #


  @ti.dataclass
  class Gaussian2D:
      mean      : vec2
      axis      : vec2
      sigma     : vec2
      alpha     : dtype

  vec_g2d = ti.types.vector(struct_size(Gaussian2D), dtype=dtype)

  @ti.func
  def to_vec_g2d(mean:vec2, axis:vec2, sigma:vec2, alpha:dtype) -> vec_g2d:
    return vec_g2d(*mean, *axis, *sigma, alpha)
  

  @ti.func
  def from_vec_g2d(vec:vec_g2d) -> Gaussian2D:
    return Gaussian2D(vec[0:2], vec[2:4], vec[4:6], vec[6])

  @ti.func
  def unpack_vec_g2d(vec:vec_g2d) -> Gaussian2D:
    return vec[0:2], vec[2:4], vec[4:6], vec[6]


  # Taichi structs don't have static methods, but they can be added afterward

  Gaussian2D.vec = vec_g2d
  Gaussian2D.to_vec = to_vec_g2d
  Gaussian2D.from_vec = from_vec_g2d
  Gaussian2D.unpack = unpack_vec_g2d



  # Projection related functions
  #

  mat2x3f = ti.types.matrix(n=2, m=3, dtype=dtype)

  @ti.func
  def project_perspective_camera_image(
      position: vec3,
      T_camera_world: mat4,
      projective_transform: mat3,
  ):
      point_in_camera = (T_camera_world @ vec4(*position, 1)).xyz
      uv = (projective_transform @ point_in_camera) / point_in_camera.z
      return uv.xy, point_in_camera


  @ti.func
  def project_perspective(
      position: vec3,
      T_image_world: mat4,
  ):
      point_in_camera = (T_image_world @ vec4(*position, 1))
      return point_in_camera.xy / point_in_camera.z, point_in_camera.z



  def camera_origin(T_camera_world: mat4):
    r, t = split_rt(T_camera_world)
    t = -(r.transpose() @ t)
    return t



  @ti.func
  def project_with_jacobian(
      position: vec3,

      camera_T_world: mat3x4,
      projection: vec4,

      image_size: vec2, 
      clamp_margin: ti.template()
  ):
    
    f = projection[0:2]
    c = projection[2:4]

    in_camera = (camera_T_world @ vec4(*position, 1))

    z = in_camera.z
    uv = (f * in_camera.xy) / z + c

    t = ti.math.clamp(uv, -image_size * clamp_margin, (image_size  - 1) * (1 + clamp_margin))

    J = mat2x3f([
        [f.x/z, 0, -(t.x - c.x) / z],
        [0, f.y/z, -(t.y - c.y) / z],
    ])

    return uv, z, J



  @ti.func
  def gaussian_covariance_in_image(
      T_camera_world: mat3x4,
      cov_rotation: vec4,
      cov_scale: vec3,
      J: mat2x3f,
  ) -> mat2:
      """ Construct and rotate the covariance matrix in camera space
      """
      
      W = T_camera_world[:3, :3]
      RS = scaled_quat_to_mat(cov_rotation, cov_scale)

      # covariance matrix, 3x3, equation (6) in the paper
      # Sigma = R @ S @ S.transpose() @ R.transpose()
      # cov_uv = J @ W @ Sigma @ W.transpose() @ J.transpose()  # equation (5) in the paper
      
      m = J @ W @ RS
      return m @ m.transpose() 

  @ti.func
  def project_gaussian(
    camera_T_world: mat3x4, projection: vec4, image_size: vec2,
    position: vec3, rotation: vec4, scale: vec3,
    clamp_margin: ti.template()):
      
  
      uv, depth, J = project_with_jacobian(
          position, camera_T_world, projection, image_size, clamp_margin)

      uv_cov = upper(gaussian_covariance_in_image(
          camera_T_world, rotation, scale, J))

      return uv, depth, uv_cov

  # 
  # Miscellaneous math functions
  #
  @ti.func
  def sigmoid(x:dtype):
      return 1. / (1. + ti.exp(-x))

  @ti.func
  def inverse_sigmoid(x:dtype):
      return -ti.log(1. / x - 1.)

  #
  # Miscellaneous conversion functions
  #

  @ti.func
  def mat3_from_ndarray(ndarray:ti.template()):
    return mat3([ndarray[i, j] 
                            for i in ti.static(range(3)) for j in ti.static(range(3))])

  @ti.func
  def mat4_from_ndarray(ndarray:ti.template()):
    return mat4([ndarray[i, j] 
                            for i in ti.static(range(4)) for j in ti.static(range(4))])
  

  @ti.func
  def mat3x4_from_ndarray(ndarray:ti.template()):
    return mat3x4([ndarray[i, j] 
                            for i in ti.static(range(3)) for j in ti.static(range(4))])


  @ti.func
  def vec4_from_ndarray(ndarray:ti.template()):
    return vec4([ndarray[i] for i in ti.static(range(4))])
  


  @ti.func
  def isfin(x):
    return ~(ti.math.isinf(x) or ti.math.isnan(x))

  #
  # Ellipsoid related functions, covariance, conic, etc.
  #

  @ti.func
  def radii_from_cov(uv_cov: vec3) -> dtype:
      
      d = (uv_cov.x - uv_cov.z)
      max_eig_sq = (uv_cov.x + uv_cov.z +
          ti.sqrt(d * d + 4.0 * uv_cov.y * uv_cov.y)) / 2.0
      
      return ti.sqrt(max_eig_sq)

  @ti.func
  def eig(cov:vec3):
      tr = cov.x + cov.z
      det = cov.x * cov.z - cov.y * cov.y

      gap = tr**2 - 4 * det
      sqrt_gap = ti.sqrt(ti.max(gap, 0))

      lambda1 = (tr + sqrt_gap) * 0.5
      lambda2 = (tr - sqrt_gap) * 0.5

      v1 = vec2(cov.x - lambda2, cov.y).normalized() 
      v2 = vec2(-v1.y, v1.x)

      return ti.sqrt(vec2(lambda1, lambda2)), v1, v2



  @ti.func
  def ellipse_bounds(uv, v1, v2):
    extent  = ti.sqrt(v1**2 + v2**2)
    return (uv - extent), (uv + extent)
  
  
  
  @ti.func 
  def clamp_bounds(lower:vec2, upper:vec2, image_size:ti.math.ivec2):
    lower = ti.math.clamp(lower, 0, image_size - 1)
    upper = ti.math.clamp(upper, 0, image_size - 1)
    return lower, upper

  @ti.func
  def cov_axes(cov:vec3):
    sigma, v1, v2 = eig(cov)
    return v1 * sigma.x, v2 * sigma.y


  @ti.func
  def inverse_cov(cov: vec3):
    # inverse of upper triangular part of symmetric matrix
    inv_det = 1 / (cov.x * cov.z - cov.y * cov.y)
    return vec3(inv_det * cov.z, -inv_det * cov.y, inv_det * cov.x)


  @ti.func
  def upper(cov: mat2) -> vec3:
    return vec3(cov[0, 0], cov[0, 1], cov[1, 1])



  @ti.func
  def radii_from_conic(conic: vec3):
      return radii_from_cov(inverse_cov(conic))


  @ti.func
  def conic_pdf(xy: vec2, uv: vec2, uv_conic: vec3) -> dtype:
      dx, dy = xy - uv
      a, b, c = uv_conic

      inner = 0.5 * (dx**2 * a + dy**2 * c) + dx * dy * b
      p = ti.exp(-inner)
      return p


  @ti.func
  def conic_pdf_with_grad(xy: vec2, uv: vec2, uv_conic: vec3):
      d = xy - uv
      a, b, c = uv_conic

      dx2 = d.x**2
      dy2 = d.y**2
      dxdy = d.x * d.y
      
      inner = 0.5 * (dx2 * a + dy2 * c) + dxdy * b
      p = ti.exp(-inner)
      
      dp_duv = vec2(
          (b * d.y + a * d.x) * p,
          (b * d.x + c * d.y) * p
      )
      dp_dconic = vec3(-0.5 * dx2 * p, -dxdy * p, -0.5 * dy2 * p)

      return p, dp_duv, dp_dconic


# def gaussian_pdf_with_grad(u, m, v1, s1, s2):
#   """ Evaluate the intensity and its gradients w.r.t. the parameters
#       2D gaussian parameterised by mean, eigenvector and two scales """
#   d = u - m

#   tx = d.dot(v1)  / s1
#   ty = d.dot(perp(v1)) / s2

#   tx2, ty2 = tx**2, ty**2

#   p = exp(-0.5 * (tx2 + ty2))

#   ds1_dp = p * tx2 / s1
#   ds2_dp = p * ty2 / s2

#   dv1_dp = p * (tx/s1 * -d + ty/s2 * perp(d))
#   dm_dp = p * (tx/s1 * v1 + ty/s2 * perp(v1))
  
#   return p, dm_dp, ds1_dp, ds2_dp, dv1_dp

  @ti.func
  def perp(v:vec2) -> vec2:
    return vec2(-v.y, v.x)

  @ti.func
  def gaussian_pdf(xy: vec2, mean: vec2, axis: vec2, sigma: vec2) -> dtype:
    d = xy - mean

    tx = d.dot(axis) / sigma.x
    ty = d.dot(perp(axis)) / sigma.y

    return ti.exp(-0.5 * (tx**2 + ty**2))


  @ti.func
  def gaussian_pdf_with_grad(xy: vec2, mean: vec2, axis: vec2, sigma: vec2):
    d = xy - mean

    tx = d.dot(axis) / sigma.x
    ty = d.dot(perp(axis)) / sigma.y

    p = ti.exp(-0.5 * (tx**2 + ty**2))

    dsigma_dp = ti.vec2(tx**2, ty**2) * p / sigma
    t_sig = ti.vec2(tx / sigma.x, ty / sigma.y)

    daxis_dp = p * t_sig.x * -d + t_sig.y * perp(d)
    dm_dp = p * t_sig.x * axis + t_sig.y * perp(axis)

    return p, dm_dp, daxis_dp, dsigma_dp


  @ti.func
  def quat_to_mat(q:vec4) -> mat3:
    x, y, z, w = q
    x2, y2, z2 = x*x, y*y, z*z

    return mat3(
      1 - 2*y2 - 2*z2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
      2*x*y + 2*w*z, 1 - 2*x2 - 2*z2, 2*y*z - 2*w*x,
      2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x2 - 2*y2
    )

  @ti.func
  def scaled_quat_to_mat(q:vec4, s:vec3) -> mat3:
    x, y, z, w = q
    x2, y2, z2 = x*x, y*y, z*z

    return mat3(
      s.x * (1 - 2*y2 - 2*z2), s.y * (2*x*y - 2*w*z), s.z * (2*x*z + 2*w*y),
      s.x * (2*x*y + 2*w*z), s.y * (1 - 2*x2 - 2*z2), s.z * (2*y*z - 2*w*x),
      s.x * (2*x*z - 2*w*y), s.y * (2*y*z + 2*w*x), s.z * (1 - 2*x2 - 2*y2)
    )

  @ti.func
  def join_rt(r:mat3, t:vec3) -> mat4:
    return mat4(
        r[0, 0], r[0, 1], r[0, 2], t[0],
        r[1, 0], r[1, 1], r[1, 2], t[1],
        r[2, 0], r[2, 1], r[2, 2], t[2],
        0, 0, 0, 1
    )

  @ti.func
  def split_rt(rt:mat4) -> ti.template():
    return rt[:3, :3], rt[:3, 3]


  @ti.func
  def qt_to_mat(q:vec4, t:vec3) -> mat4:
    r = quat_to_mat(q)
    return mat4(
      r[0, 0], r[0, 1], r[0, 2], t[0],
      r[1, 0], r[1, 1], r[1, 2], t[1],
      r[2, 0], r[2, 1], r[2, 2], t[2],
      0, 0, 0, 1
    )
    

  @ti.func
  def scaling_matrix(scale:vec3) -> mat3:
    return mat3(
      scale.x, 0, 0,
      0, scale.y, 0,
      0, 0, scale.z
    )

  @ti.func
  def quat_mul(q1: vec4, q2: vec4) -> vec4:
      return vec4(
          q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,
          q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x,
          q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w,
          q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,
      )

  @ti.func
  def quat_conj(q: vec4) -> vec4:
      return vec4(-q.x, -q.y, -q.z, q.w)


  @ti.func
  def quat_rotate(q: vec4, v: vec3) -> vec3:
      qv = vec4(*v, 0.0)
      q_rot = quat_mul(q, quat_mul(qv, quat_mul(q)))
      return q_rot.xyz


  return SimpleNamespace(**locals())