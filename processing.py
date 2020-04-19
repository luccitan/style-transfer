# coding: utf-8
# pylint: disable=missing-module-docstring

import torch
import torchvision.transforms as transforms
import numpy as np

from utils import Utils

PRE_TRANSFORMER = transforms.Compose([
  transforms.Resize(Utils.IMAGE_SIZE),
  transforms.ToTensor(),
  transforms.Normalize(
    mean=torch.tensor(Utils.VGG_NORMALIZATION_MEAN).to(Utils.DEVICE),
    std=torch.tensor(Utils.VGG_NORMALIZATION_STD).to(Utils.DEVICE))
])

class Processing:
  """Image processing (un)loading utils and functions"""

  @staticmethod
  def preprocessor(image):
    return PRE_TRANSFORMER(image).unsqueeze(0).to(Utils.DEVICE, torch.float)

  @staticmethod
  def postprocessor(tensor):
    image = tensor.to('cpu').clone().detach()
    image = image.numpy().squeeze().transpose(1, 2, 0)
    # Transposed to H, W, C so that we can denormalize and plot with Matplotlib
    image = image * np.array(Utils.VGG_NORMALIZATION_STD) + np.array(Utils.VGG_NORMALIZATION_MEAN)
    image = image.clip(0, 1)
    return image
