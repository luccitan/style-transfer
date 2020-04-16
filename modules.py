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

class Normalization(nn.Module):
  """Normalization layer to apply on input images."""

  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    # (Comment from the tutorial: ...)
    # .view the mean and std to make them [C x 1 x 1] so that they can
    # directly work with image Tensor of shape [B x C x H x W].
    # B is batch size. C is number of channels. H is height and W is width.
    self.mean = torch.tensor(mean).view(-1, 1, 1)
    self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
    return (img - self.mean) / self.std

class ContentLoss(nn.Module):
  """ TODO """
  def __init__(self, target,):
    super(ContentLoss, self).__init__()
    # we 'detach' the target content from the tree used
    # to dynamically compute the gradient: this is a stated value,
    # not a variable. Otherwise the forward method of the criterion
    # will throw an error.
    self.target = target.detach()

  def forward(self, tensor):
    self.loss = F.mse_loss(tensor, self.target) # pylint: disable=attribute-defined-outside-init
    return tensor


class StyleLoss(nn.Module):
  """ TODO """

  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()

  def forward(self, tensor):
    gram = gram_matrix(tensor)
    self.loss = F.mse_loss(gram, self.target) # pylint: disable=attribute-defined-outside-init
    return tensor
