import copy
from typing import Iterable, Mapping, Tuple
from beartype import beartype
from beartype.typing import Dict, Optional

from tensordict import TensorDict
import torch.optim as optim
import torch



class ParameterClass():
  """
  Maintains a group of mixed parameters/non parameters in a TensorClass,
  as well as optimizer state synchronized with the parameters.

  Supports filtering with python indexing and appending parameters.

  Parameters:
    tensors: TensorDict
    parameter_groups: dict of key -> parameter dict for parameters to be optimized, 
                      keys must exist in tensors dict
                      
    param_state: dict of key -> optimizer state to insert into the optimizer
  """

  @beartype
  def __init__(self, tensors:TensorDict, 
               parameter_groups:Dict[str, Dict], 
               optimizer_state:Optional[Tuple[TensorDict, Dict]]=None,
               optimizer = optim.Optimizer,

               **optim_kwargs):

    self.tensors:TensorDict = as_parameters(tensors, parameter_groups.keys())


    param_groups = [
      dict(params=[self.tensors[name]], name=name, **group)
        for name, group in parameter_groups.items()
    ]

    self.optimizer = optimizer(param_groups, **optim_kwargs)
    self.optim_kwargs = optim_kwargs

    if optimizer_state is not None:
      tensor_state, other_state = optimizer_state

      for k in tensor_state.keys():
          assert k in self.tensors.keys(), f"state parameter {k} not in {self.tensors.keys()}"
          self.optimizer.state[self.tensors[k]] = {**tensor_state[k].to_dict(), **other_state[k]}

  
  @property
  def parameter_groups(self):

    return {group['name']: {k: v for k, v in group.items() if k not in ['params', 'name']}
      for group in self.optimizer.param_groups}


  @property
  def learning_rates(self):
    return {group['name']: group['lr'] for group in self.optimizer.param_groups}


  @beartype
  def set_learning_rate(self, **kwargs:float):
    learning_rates = replace_dict(self.learning_rates, **kwargs)

    for group in self.optimizer.param_groups:
      group['lr'] = learning_rates[group['name']]

    return self
  
  @beartype
  def update_group(self, name:str, **kwargs):
    for group in self.optimizer.param_groups:
      if group['name'] == name:
        group.update(kwargs)
        return
    
    raise ValueError(f"Group {name} not found in optimizer")


  @beartype
  def update_groups(self, **kwargs):
    for name, params in kwargs.items():
      self.update_group(name, **params)


  def state_dict(self) -> Dict:
    return {
      'tensors': self.tensors.to_dict(),
      'optimizer': (self.tensor_state.to_dict(), self.other_state),
      'parameter_groups': self.parameter_groups
    }
  
  def __getstate__(self):
    return self.__dict__

  def __setstate__(self, state):
    self.__dict__.update(state)

  
  @staticmethod
  def from_state_dict(state:dict, optimizer=optim.Adam, **optim_kwargs) -> 'ParameterClass':
    tensor_state, other_state = state['optimizer']
    return ParameterClass(
      TensorDict.from_dict(state['tensors'], batch_dims=1), 
      parameter_groups=state['parameter_groups'],
      optimizer_state=(TensorDict.from_dict(tensor_state), other_state),
      optimizer=optimizer,
      **optim_kwargs
    )

  def zero_grad(self):
    self.optimizer.zero_grad()


  def step(self, **kwargs):
    self.optimizer.step(**kwargs)



  def keys(self):
    return self.tensors.keys()

  def optimized_keys(self):
    return self.parameter_groups.keys()
    
  def items(self):
    return self.tensors.items()
  



  def __getattr__(self, name):
    if name  not in self.tensors.keys():
      return self.__getattribute__(name)
    
    return self.tensors[name]

  def modify_tensors(self, f):
    return ParameterClass(
      f(self.tensors), 
      parameter_groups=self.parameter_groups, 
      optimizer_state=(f(self.tensor_state), self.other_state),
      optimizer=type(self.optimizer),
      **self.optim_kwargs
    )
  


  def apply(self, f):
    return self.modify(lambda x: x.apply(f))
  
  def to(self, device):
    return self.modify(lambda x: x.to(device))

  def replace(self, **kwargs):
      return ParameterClass(
      self.tensors.replace(**kwargs), 
      parameter_groups=self.parameter_groups, 
      optimizer_state=self.optimizer_state,
      optimizer=type(self.optimizer),
      **self.optim_kwargs
    )
  

  def detach(self) -> TensorDict:
    return self.tensors.detach()


  def to_dict(self):
    return self.tensors.to_dict()
  
  @property
  def batch_size(self):
    return self.tensors.batch_size
  
  @property
  def batch_dims(self):
    return self.tensors.batch_dims

  def _get_state(self, f):
    return {name:f(self.optimizer.state[param])
              for name, param in self.tensors.items()
                if param in self.optimizer.state}

  @property
  def tensor_state(self) -> TensorDict:
    def get_tensor_states(state):
      return {k : v for k, v in state.items() if torch.is_tensor(v)}
    
    return TensorDict.from_dict(
      self._get_state(get_tensor_states), 
      batch_dims=self.batch_dims
    )
  
  @property
  def other_state(self) -> Dict:
    def get_other_states(state):
      return {k: v for k, v in state.items() if not torch.is_tensor(v)}
    return self._get_state(get_other_states)

  @property
  def optimizer_state(self) -> Tuple[TensorDict, Dict]:
    return self.tensor_state, self.other_state

  @beartype
  def __getitem__(self, idx:torch.Tensor | str):

    if isinstance(idx, str):
      return self.tensors[idx]
    else:
      state = (self.tensor_state[idx], self.other_state)
      return ParameterClass(self.tensors[idx], self.parameter_groups, state, optimizer=type(self.optimizer), **self.optim_kwargs)
  
  def append_tensors(self, tensors:TensorDict, tensor_state:Optional[TensorDict]=None):
    if tensor_state is not None:
      assert tensors.shape == tensor_state.shape, f"{tensors.shape} != {tensor_state.shape}"

    assert set(tensors.keys()) == set(self.tensors.keys()), f"{tensors.keys()} != {self.tensors.keys()}"
    n = tensors.batch_size[0]

    if tensor_state is None:
      tensor_state = self.tensor_state.new_zeros(n)
      
    return ParameterClass(torch.cat([self.tensors, tensors]), 
                          self.parameter_groups, 
                          optimizer_state=(torch.cat([self.tensor_state, tensor_state]), self.other_state), 
                          optimizer=type(self.optimizer),
                          **self.optim_kwargs)

  def append(self, params:'ParameterClass'):
    return self.append_tensors(params.tensors)


@beartype
def as_parameters(tensors:TensorDict | Mapping[str, torch.Tensor], keys:Iterable[str]):
  
    param_dict = {
      k: torch.nn.Parameter(x.detach(), requires_grad=True) 
            if k in keys else x 
        for k, x in tensors.items()}

    cls = type(tensors)
    return cls.from_dict(param_dict) 


def replace_dict(d, **kwargs):
  d = copy.copy(d)
  d.update(kwargs)
  return d



