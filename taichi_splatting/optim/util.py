import torch



def get_vector_state(state:dict, param: torch.Tensor):
  if 'v' not in state:
    state['v'] = torch.zeros_like(param.view(param.shape[0], -1))
    state['m'] = torch.zeros((param.shape[0],), dtype=param.dtype, device=param.device)

  return state['v'], state['m']


def get_scalar_state(state:dict, param: torch.Tensor):
  if 'v' not in state:
    state['v'] = torch.zeros_like(param.view(param.shape[0], -1))
    state['m'] = torch.zeros_like(param.view(param.shape[0], -1))

  return state['v'], state['m']

def get_total_weight(state:dict, n:int, device:torch.device):
  if 'total_weight' not in state:
    state['total_weight'] = torch.zeros(n, device=device, dtype=torch.float32)

  return state['total_weight']


def get_running_vis(state:dict, shape:tuple, device:torch.device):
  if 'running_vis' not in state:
    state['running_vis'] = torch.zeros(shape, device=device, dtype=torch.float32)

  return state['running_vis']


def flatten_param(param: torch.Tensor):
  return param.view(param.shape[0], -1), param.grad.view(param.shape[0], -1)