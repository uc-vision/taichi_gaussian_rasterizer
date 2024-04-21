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

  mat4x2 = ti.types.matrix(4, 2, dtype=dtype)

  #
  # Gaussian datatypes
  #


  @ti.dataclass
  class Gaussian2D:
      uv        : vec2
      uv_conic  : vec3
      alpha   : dtype



  @ti.dataclass
  class Gaussian3D:
      position   : vec3
      log_scaling : vec3
      rotation    : vec4
      alpha_logit : dtype

      @ti.func
      def alpha(self):
        return sigmoid(self.alpha_logit)

      @ti.func
      def scale(self):
          return ti.math.exp(self.log_scaling)


  vec_g2d = ti.types.vector(struct_size(Gaussian2D), dtype=dtype)
  vec_g3d = ti.types.vector(struct_size(Gaussian3D), dtype=dtype)


  @ti.func
  def to_vec_g2d(uv:vec2, uv_conic:vec3, alpha:dtype) -> vec_g2d:
    return vec_g2d(*uv, *uv_conic, alpha)

  @ti.func
  def to_vec_g3d(position:vec3, log_scaling:vec3, rotation:vec4, alpha_logit:dtype) -> vec_g3d:
    return vec_g3d(*position, *log_scaling, *rotation, alpha_logit)


  @ti.func
  def unpack_vec_g3d(vec:vec_g3d) -> Gaussian3D:
    return vec[0:3], vec[3:6], vec[6:10], vec[10]

  @ti.func
  def unpack_vec_g2d(vec:vec_g2d) -> Gaussian2D:
    return vec[0:2], vec[2:5], vec[5]

  @ti.func
  def get_position_g3d(vec:vec_g3d) -> vec3:
    return vec[0:3]

  @ti.func
  def get_position_g2d(vec:vec_g2d) -> vec2:
    return vec[0:2]

  @ti.func
  def get_conic_g2d(vec:vec_g2d) -> vec3:
    return vec[2:5]


  @ti.func
  def get_cov_g2d(vec:vec_g2d) -> vec3:
    conic = get_conic_g2d(vec)
    return inverse_cov(conic)

  @ti.func
  def from_vec_g3d(vec:vec_g3d) -> Gaussian3D:
    return Gaussian3D(vec[0:3], vec[3:6], vec[6:10], vec[10])

  @ti.func
  def from_vec_g2d(vec:vec_g2d) -> Gaussian2D:
    return Gaussian2D(vec[0:2], vec[2:5], vec[5])


  @ti.func
  def unpack_activate_g3d(vec:vec_g3d):
    position, log_scaling, rotation, alpha_logit = unpack_vec_g3d(vec)
    return position, ti.exp(log_scaling), ti.math.normalize(rotation), sigmoid(alpha_logit)
  



  @ti.func
  def bounding_sphere(vec:vec_g3d, gaussian_scale: ti.template()):
    position, log_scaling = vec[0:3], vec[3:6]
    return position, ti.exp(log_scaling).max() * gaussian_scale

  # Taichi structs don't have static methods, but they can be added afterward
  Gaussian2D.vec = vec_g2d
  Gaussian2D.to_vec = to_vec_g2d
  Gaussian2D.from_vec = from_vec_g2d
  Gaussian2D.unpack = unpack_vec_g2d

  Gaussian2D.get_position = get_position_g2d
  Gaussian2D.get_conic = get_conic_g2d
  Gaussian2D.get_cov = get_cov_g2d


  Gaussian3D.vec = vec_g3d
  Gaussian3D.to_vec = to_vec_g3d
  Gaussian3D.from_vec = from_vec_g3d
  Gaussian3D.unpack = unpack_vec_g3d
  Gaussian3D.unpack_activate = unpack_activate_g3d
  Gaussian3D.get_position = get_position_g3d
  Gaussian3D.bounding_sphere = bounding_sphere



  #
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
  def gaussian_covariance_in_camera(
      T_camera_world: mat4,
      cov_rotation: vec4,
      cov_scale: vec3,
  ) -> mat3:
      """ Construct and rotate the covariance matrix in camera space
      """
      
      W = T_camera_world[:3, :3]
      R = quat_to_mat(cov_rotation)

      S = mat3([
          [cov_scale.x, 0, 0],
          [0, cov_scale.y, 0],
          [0, 0, cov_scale.z]
      ])
      # covariance matrix, 3x3, equation (6) in the paper
      # Sigma = R @ S @ S.transpose() @ R.transpose()
      # cov_uv = J @ W @ Sigma @ W.transpose() @ J.transpose()  # equation (5) in the paper
      
      m = W @ R @ S
      return m @ m.transpose() 


  @ti.func
  def get_projective_transform_jacobian(
      projection: mat3,
      position: vec3,
  ):
      f = vec2(projection[0, 0], projection[1, 1])
      c = vec2(projection[0, 2], projection[1, 2])
      x, y, z = position

      return mat2x3f([
         [f.x/z, 0, c.x/z - (c.x*z + f.x*x)/z**2],
         [0, f.y/z, c.y/z - (c.y*z + f.y*y)/z**2],
      ])

         

  @ti.func
  def project_perspective_gaussian(
      projective_transform: mat3,
      point_in_camera: vec3,
      cov_in_camera: mat3) -> mat2:
      """ Approximate the 2D gaussian covariance in image space """

      J = get_projective_transform_jacobian(
          projective_transform, point_in_camera)
      
      cov_uv = J @ cov_in_camera @ J.transpose()
      return cov_uv




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
  def cov_axes(cov:vec3):
      tr = cov.x + cov.z
      det = cov.x * cov.z - cov.y * cov.y

      gap = tr**2 - 4 * det
      sqrt_gap = ti.sqrt(ti.max(gap, 0))

      lambda1 = (tr + sqrt_gap) * 0.5
      lambda2 = (tr - sqrt_gap) * 0.5

      v1 = vec2(cov.x - lambda2, cov.y).normalized() 
      v2 = vec2(v1.y, -v1.x)

      return v1 * ti.sqrt(lambda1), v2 * ti.sqrt(lambda2)  


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
  def conic_pdf(xy: vec2, uv: vec2, uv_conic: vec3, beta:ti.template()) -> dtype:
      dx, dy = xy - uv
      a, b, c = uv_conic

      inner = (0.5 * (dx**2 * a + dy**2 * c) + dx * dy * b)
      p = ti.exp(-inner ** beta)
      return p


  

  @ti.func
  def conic_pdf_with_grad(xy: vec2, uv: vec2, uv_conic: vec3, beta:ti.template()):
      d = xy - uv
      a, b, c = uv_conic

      dx2 = d.x**2
      dy2 = d.y**2
      dxdy = d.x * d.y
      
      # should not be negative, but to prevent NaNs
      inner =  ti.math.max(0.5 * (dx2 * a + dy2 * c) + dxdy * b, 0.0)
      z = inner ** beta
      p = ti.exp(-z)

      d_inner = beta * inner ** (beta - 1) * p

      dp_duv = vec2(
          (a * d.x + b * d.y) * d_inner,
          (c * d.y + b * d.x) * d_inner
      )
      
      dp_dconic = vec3(
          -0.5  * dx2 * d_inner,
          -dxdy       * d_inner,
          -0.5  * dy2 * d_inner)

      # dp_dbeta = -z * ti.log(inner) * p if inner > 1e-8 else 0.0
      return p, dp_duv, dp_dconic





  @ti.func
  def cov_inv_basis(uv_cov: vec3, scale: dtype) -> mat2:
      basis = ti.Matrix.cols(cov_axes(uv_cov))
      return (basis * scale).inverse()


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