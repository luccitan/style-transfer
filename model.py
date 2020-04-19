# coding: utf-8
"""This file defines functions used to load, init and defines the style transfer models and losses"""

from collections import OrderedDict

import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
  """Based on the VGG-19 model, tweak ReLU and Pooling layers with pretrained convolutions layers
  to apply Neural Style Transfer"""

  def __init__(self):
    super().__init__()
    vgg19 = vgg19 = models.vgg19(pretrained=True).features.eval()
    self.sequential = nn.Sequential(OrderedDict([
      # Block 1
      ('conv1_1', vgg19[0]), ('relu1_1', nn.ReLU(inplace=False)),
      ('conv1_2', vgg19[2]), ('relu1_2', nn.ReLU(inplace=False)),
      ('pool1', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
      # Block 2
      ('conv2_1', vgg19[5]), ('relu2_1', nn.ReLU(inplace=False)),
      ('conv2_2', vgg19[7]), ('relu2_2', nn.ReLU(inplace=False)),
      ('pool2', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
      # Block 3
      ('conv3_1', vgg19[10]), ('relu3_1', nn.ReLU(inplace=False)),
      ('conv3_2', vgg19[12]), ('relu3_2', nn.ReLU(inplace=False)),
      ('conv3_3', vgg19[14]), ('relu3_3', nn.ReLU(inplace=False)),
      ('conv3_4', vgg19[16]), ('relu3_4', nn.ReLU(inplace=False)),
      ('pool3', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
      # Block 4
      ('conv4_1', vgg19[19]), ('relu4_1', nn.ReLU(inplace=False)),
      ('conv4_2', vgg19[21]), ('relu4_2', nn.ReLU(inplace=False)),
      ('conv4_3', vgg19[23]), ('relu4_3', nn.ReLU(inplace=False)),
      ('conv4_4', vgg19[25]), ('relu4_4', nn.ReLU(inplace=False)),
      ('pool4', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
      # Block 5
      ('conv5_1', vgg19[28]), ('relu5_1', nn.ReLU(inplace=False)),
      ('conv5_2', vgg19[30]), ('relu5_2', nn.ReLU(inplace=False)),
      ('conv5_3', vgg19[32]), ('relu5_3', nn.ReLU(inplace=False)),
      ('conv5_4', vgg19[34]), ('relu5_4', nn.ReLU(inplace=False)),
      ('pool5', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)),
    ]))
    # freezing the model
    for parameter in self.sequential.parameters():
      parameter.requires_grad_(False)

  def forward(self, tensor, outs=None):
    if outs is None:
      return self.sequential(tensor)
    assert isinstance(outs, list)

    outputs = {}
    for name, module in self.sequential.named_children():
      tensor = module(tensor)
      if name in outs:
        outputs[name] = tensor
    return outputs
