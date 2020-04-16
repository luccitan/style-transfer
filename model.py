# coding: utf-8
"""This file defines functions used to load, init and defines the style transfer models and losses"""

import torch.nn as nn
import torchvision.models as models

from utils import Utils
from modules import ContentLoss, StyleLoss, Normalization

def get_model_and_losses(content_image, style_image, normalization_mean=None, normalization_std=None):
  """ TODO """
  normalization_mean = normalization_mean or Utils.NORMALIZATION_MEAN
  normalization_std = normalization_std or Utils.NORMALIZATION_STD
  normalization = Normalization(normalization_mean, normalization_std).to(Utils.DEVICE)

  style_losses = []
  content_losses = []
  model = nn.Sequential(normalization)
  vgg19 = models.vgg19(pretrained=True).features.to(Utils.DEVICE).eval()

  i = 0
  for layer in vgg19.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = f"conv_{i}"
    elif isinstance(layer, nn.ReLU):
      name = f"relu_{i}"
      layer = nn.ReLU(inplace=False) # original source claims that in-place doesn't play with Style Transfer losses
    elif isinstance(layer, nn.MaxPool2d):
      name = f"pool_{i}"
    elif isinstance(layer, nn.BatchNorm2d):
      name = f"bn_{i}"
    else:
      raise RuntimeError(f"Unrecognized layer : {layer.__class__.__name__}")
    model.add_module(name, layer)

    if name in Utils.CONTENT_LAYERS:
      target = model(content_image).detach()
      content_loss = ContentLoss(target)
      model.add_module(f"content_loss_{i}", content_loss)
      content_losses.append(content_loss)

    if name in Utils.STYLE_LAYERS:
      target = model(style_image).detach()
      style_loss = StyleLoss(target)
      model.add_module(f"style_loss_{i}", style_loss)
      style_losses.append(style_loss)

  # Here we look for the last content OR style loss layer starting from the end
  # and we remove the layers that follow because they'll be useless
  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], (ContentLoss, StyleLoss)):
      break
  model = model[:(i + 1)]

  return model, style_losses, content_losses
