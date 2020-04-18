# coding: utf-8
# pylint: disable=missing-module-docstring

from PIL import Image

import torch
import torchvision.transforms as transforms

from utils import Utils

NORMALIZATION_MEAN = torch.tensor(Utils.VGG_NORMALIZATION_MEAN).to(Utils.DEVICE)
NORMALIZATION_STD = torch.tensor(Utils.VGG_NORMALIZATION_STD).to(Utils.DEVICE)

class Processing:
  """Image processing (un)loading utils and functions"""


  preprocessor = transforms.Compose([
    transforms.Resize(Utils.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
  ])

  postprocessor = transforms.Compose([
    transforms.Normalize(mean=-NORMALIZATION_MEAN / NORMALIZATION_STD, std=1 / NORMALIZATION_STD),
    transforms.Lambda(lambda x: x.data.clamp_(0, 1)),
    transforms.ToPILImage()
  ])

  @staticmethod
  def load_image(path):
    image = Image.open(path)
    image = Processing.preprocessor(image).unsqueeze(0).to(Utils.DEVICE, torch.float)
    return image
