from typing import List, Tuple
import torch
import torch.nn as nn

from taichi_splatting.misc.renderer2d import point_covariance
from .renderer2d import Gaussians2D

def kl_divergence(means1:torch.Tensor, means2:torch.Tensor, cov1:torch.Tensor, cov2:torch.Tensor) -> torch.Tensor:
  """Compute KL divergence between two 2D Gaussian distributions.
  
  Args:
    means1: (N,D) tensor of means for first distribution
    means2: (N,D) tensor of means for second distribution  
    cov1: (N,D,D) tensor of covariance matrices for first distribution
    cov2: (N,D,D) tensor of covariance matrices for second distribution

  Returns:
    (N,) tensor of KL divergence values
  """
  # Compute inverse of cov2
  cov2_inv = torch.linalg.inv(cov2)
  
  # Compute trace term
  trace_term = torch.einsum('...ij,...jk->...ik', cov2_inv, cov1)
  trace_term = torch.diagonal(trace_term, dim1=-2, dim2=-1).sum(-1)
  
  # Compute quadratic term for means
  mean_diff = means2 - means1
  quad_term = torch.einsum('...i,...ij,...j->...', mean_diff, cov2_inv, mean_diff)
  
  # Combine terms
  return 0.5 * (trace_term + quad_term - 2 + torch.logdet(cov2) - torch.logdet(cov1))
  

def linear(in_features, out_features,  init_std=None):
  m = nn.Linear(in_features, out_features, bias=True)

  if init_std is not None:
    m.weight.data.normal_(0, init_std)
    
  m.bias.data.zero_()
  return m


def layer(in_features, out_features, activation=nn.Identity, norm=nn.Identity, **kwargs):
  return nn.Sequential(linear(in_features, out_features, **kwargs), 
                       norm(out_features),
                       activation(),
                       )


def mlp_body(inputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity):
  return nn.Sequential(
    layer(inputs, hidden_channels[0], activation),
    *[layer(hidden_channels[i], hidden_channels[i+1], activation, norm)  
      for i in range(len(hidden_channels) - 1)]
  )   


def mlp(inputs, outputs, hidden_channels: List[int], activation=nn.ReLU, norm=nn.Identity, 
        output_activation=nn.Identity, output_scale =None):

  output_layer = layer(hidden_channels[-1], outputs, 
                       output_activation,
                       init_std=output_scale)
  
  return nn.Sequential(
    mlp_body(inputs, hidden_channels, activation, norm),
    output_layer
  )   
