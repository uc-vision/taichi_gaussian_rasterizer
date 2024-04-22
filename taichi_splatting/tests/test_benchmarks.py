from taichi_splatting.benchmarks import bench_conic_projection, bench_conic_rasterizer, bench_sh

# currently takes a little too long to compile backward kernels
# def test_bench_projection():
#   args = bench_projection.parse_args([])
#   bench_projection.bench_projection(args)

def test_bench_rasterizer():
  args = bench_conic_rasterizer.parse_args([])
  bench_conic_rasterizer.bench_rasterizer(args)

def test_bench_sh():
  args = bench_sh.parse_args([])
  bench_sh.bench_sh(args)