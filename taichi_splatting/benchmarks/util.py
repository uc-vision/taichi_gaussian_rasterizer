import time
import torch
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity

def profiled_benchmark(name, f, iters=100, warmup=1):
  for _ in range(warmup):
    f()

  with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_inference"):
      for _ in tqdm(range(iters), desc=f"{name}"):
        f()
      torch.cuda.synchronize()

  result = prof.key_averages().table(sort_by="self_cuda_time_total", 
                                      row_limit=25, max_name_column_width=100)
  print(result)


  

def timed_benchmark(name, f, iters=100, warmup=10):
  for _ in range(warmup):
    f()


  start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
  start.record()
  for _ in tqdm(range(iters), desc=f"{name}"):
    f()

  end.record()
  torch.cuda.synchronize()

  elapsed = start.elapsed_time(end) / 1000.
  print(f'{name}  {iters} iterations in {elapsed:.3f}s at {iters / elapsed:.1f} iters/sec')


def benchmarked(name, f, iters=100, warmup=10, profile: bool = False):
  if profile:
    profiled_benchmark(name, f, iters, warmup)
  else:
    timed_benchmark(name, f, iters, warmup)


