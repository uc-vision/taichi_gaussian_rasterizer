import copy
from functools import cached_property
from beartype.typing import Dict
from beartype import beartype
from beartype.typing import Callable, List, Dict, Optional
from tensordict import TensorDict
import torch.optim as optim
import torch



class ParameterClass():
  """
  Maintains a group of mixed parameters/non parameters in a TensorClass,
  as well as optimizer state synchronized with the parameters.

  Note - call ParameterClass.create to create a ParameterClass instead 
  of the constructor directly.

  Supports filtering with python indexing and appending parameters.

  Parameters:
    tensors: TensorClass
    param_groups: list of parameter dicts, see torch.optim for details 
    param_state: dict of optimizer state to insert into the optimizer
  """

  def __init__(self, tensors:TensorDict, param_groups:List, param_state:Optional[Dict]=None):

    self.tensors = tensors
    self.optimizer = optim.Adam(param_groups, foreach=True, betas=(0.7, 0.999))

    if param_state is not None:
      for k, v in param_state.items():
        self.optimizer.state[self.tensors[k]] = v


  @beartype
  @staticmethod
  def create(tensors:TensorDict, learning_rates:Dict[str, float], base_lr=1.0):
    param_dict = as_parameters(tensors, learning_rates.keys())

    param_groups = [
      dict(params=[param_dict[name]], lr=lr * base_lr, name=name)
        for name, lr in learning_rates.items()
    ]

    tensors = replace_dict(tensors, **param_dict)
    return ParameterClass(tensors, param_groups)


  def zero_grad(self):
    self.optimizer.zero_grad()


  def step(self):
    self.optimizer.step()

  def keys(self):
    return self.tensors.keys()

  def optimized_keys(self):
    return {group["name"] for group in self.optimizer.param_groups}
    
  def items(self):
    return self.tensors.items()


  def __getattr__(self, name):
    if name  not in self.tensors.keys():
      raise AttributeError(f'ParameterClass has no attribute {name}')
    
    return self.tensors[name]


  def to(self, device):
    return ParameterClass(
      self.tensors.to(device), self.optimizer.param_groups, self.get_state()
    )

  def to_dict(self):
    return self.tensors.to_dict()
  
  @property
  def batch_size(self):
    return self.tensors.batch_size


    
  def _updated_state(self, f:Callable):
    def modify_state(state):
      return {k : f(v) if torch.is_tensor(v) and v.dim() > 0 else state[k]
                for k, v in state.items()}

    return {name:modify_state(self.optimizer.state[param])
              for name, param in self.tensors.items()
                if param in self.optimizer.state}
    

  def get_state(self):
    return self._updated_state(lambda x: x)


  def _updated_tensors(self, tensors):
    params = as_parameters(tensors, self.optimized_keys())
    updated_groups = [ replace_dict(group, params=[params[group["name"]]])
      for group in self.optimizer.param_groups]
    
    return replace_dict(tensors, **params), updated_groups

  def __getitem__(self, idx):
    tensors, updated_groups = self._updated_tensors(self.tensors[idx])
    state = self._updated_state(lambda x: x[idx])
    return ParameterClass(tensors, updated_groups, state)
  
  def append_tensors(self, tensors):
    n = tensors.batch_size[0]

    tensors, updated_groups = self._updated_tensors(torch.cat([self.tensors, tensors]))
    state = self._updated_state(lambda x: torch.cat(
      [x, x.new_zeros(n, *x.shape[1:])] )
    )
    return ParameterClass(tensors, updated_groups, state)

  def append(self, params:'ParameterClass'):
    tensors, updated_groups = self._updated_tensors(
        torch.cat([self.tensors, params.tensors]))
    
    state = concat_states(self.get_state(), params.get_state())
    return ParameterClass(tensors, updated_groups, state)


def as_parameters(tensors, keys):
    param_dict = {k: torch.nn.Parameter(x, requires_grad=True) 
                        for k, x in tensors.items()
                        if k in keys}

    cls = type(tensors)
    return cls.from_dict(param_dict) 


def replace_dict(d, **kwargs):
  d = copy.copy(d)
  d.update(kwargs)
  return d


def concat_states(state, other_state):
  def concat_dict(d1, d2):

    return {k: torch.concat( (d1[k], d2[k]) ) if d1[k].dim() > 0 else d1[k]
            for k in d1.keys()}

  return {k: concat_dict(state[k], other_state[k]) 
          for k in state.keys()}