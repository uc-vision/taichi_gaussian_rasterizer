
from concurrent.futures import Future, ThreadPoolExecutor
from functools import partial
import threading

import taichi as ti


class NullExecutor:
    def __init__(self, **kwargs):
        self._threads = []

    def submit(self, fn, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future
    
    def shutdown(self, wait=True):
        pass


class TaichiQueueContext:
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def __enter__(self):
    TaichiQueue.init(*self.args, **self.kwargs)

  def __exit__(self, exc_type, exc_value, traceback):
    TaichiQueue.stop()


def taichi_queue(*args, **kwargs):
  return TaichiQueueContext(*args, **kwargs)



class TaichiQueue():
  executor: ThreadPoolExecutor = None

  @classmethod
  def init(cls, *args, threaded=False, **kwargs) -> None:
    if cls.executor is not None:
      return cls.executor
    
    executor = ThreadPoolExecutor if threaded else NullExecutor

    cls.executor = executor(max_workers=1, thread_name_prefix="taichi")
    cls.run_sync(partial(ti.init, *args, **kwargs))

  @staticmethod
  def thread_id():
    executor = TaichiQueue.queue()
    threads = list(executor._threads)
    return threads[0].ident if len(threads) > 0 else None
    
  @classmethod
  def queue(cls) -> ThreadPoolExecutor:
    assert cls.executor is not None, "TaichiQueue not initialized (run TaichiQueue.init() in place of ti.init())"
    return cls.executor
  
  @staticmethod
  def _await_run(func, *args, **kwargs) -> any:
    args = [arg.result() if isinstance(arg, Future) else arg for arg in args]
    return func(*args, **kwargs)
      
  @staticmethod
  def run_async(func, *args, **kwargs) -> Future:
    return TaichiQueue.queue().submit(TaichiQueue._await_run, func, *args, **kwargs)
  
  @staticmethod
  def run_sync(func, *args, **kwargs) -> any:
  
    assert threading.get_ident() != TaichiQueue.thread_id(), "TaichiQueue.run_sync() called from worker thread (will deadlock)"
    return TaichiQueue.run_async(func, *args, **kwargs).result()
  
  @classmethod
  def stop(cls) -> None:
    executor = TaichiQueue.executor
    if executor is not None:
      cls.run_sync(ti.reset)
      executor.shutdown(wait=True)
      TaichiQueue.executor = None


def queued(kernel):
  def f(*args, **kwargs):
    return TaichiQueue.run_sync(kernel, *args, **kwargs)
  return f
