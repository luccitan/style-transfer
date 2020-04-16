# coding: utf-8
"""Utilitary file containing constants, configurations, predefined parameters"""

import torch
import torchvision.transforms as transforms

class Utils:
  """ TODO """

  PROJECT = 'style_transfer'

  # Hyperparameters and model predefined constants
  EPOCHS = 300
  NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
  NORMALIZATION_STD = (0.229, 0.224, 0.225)
  CONTENT_WEIGHT = 1
  CONTENT_LAYERS = ['conv_4']
  STYLE_WEIGHT = 1000000
  STYLE_LAYERS = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
  IMAGE_SIZE = 512 if torch.cuda.is_available() else 256
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Torch image transformers
  IMAGE_UNLOADER = transforms.ToPILImage()
  IMAGE_TRANSFORMER = transforms.Compose([
    transforms.Resize(IMAGE_SIZE), # dynamic size depending on device used
    transforms.ToTensor()
  ])

  LOGGING_CONFIG = {
    'version': 1,
    'incremental': False,
    'formatters': {
      'default': {
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      }
    },
    'handlers': {
      'console': {
        'level': 'DEBUG',
        'class': 'logging.StreamHandler',
        'formatter': 'default'
      },
    },
    'loggers': {
      PROJECT: {
        'level': 'DEBUG',
        'handlers': ['console']
      }
    }
  }
