# coding: utf-8
"""
This file contains extra layers added to the VGG network.

They are :
  - image input normalizations layers
  - losses layers applied automatically after each convulution layers

These layers are taken from this tutorial:
  https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input_tensor):
  batch_size, feature_maps, height, width = input_tensor.size()
  features = input_tensor.view(batch_size * feature_maps, height * width)
  gram = torch.mm(features, features.t())

  # Normalizing the gram matrix otherwise inputs with biggest dimensions would have higher values
  return gram.div(batch_size * feature_maps * height * width)

class ContentLoss(nn.Module):
  """Loss to compare the content of input and content image.
  Mean-squared error over associated layer features
  """

  def __init__(self, target):
    super().__init__()
    self.target = target.detach()

  def forward(self, tensor):
    return F.mse_loss(tensor, self.target) # pylint: disable=attribute-defined-outside-init

class StyleLoss(nn.Module):
  """Loss to compare the style of input and style image.
  Mean-squared error over Gram matrix of associated layer features
  """

  def __init__(self, target):
    super().__init__()
    self.target = gram_matrix(target).detach()

  def forward(self, tensor):
    gram = gram_matrix(tensor)
    return F.mse_loss(gram, self.target) # pylint: disable=attribute-defined-outside-init
