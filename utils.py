# coding: utf-8
"""Utilitary file containing constants, configurations, predefined parameters"""

import torch

class Utils:
  """ TODO """

  PROJECT = 'style_transfer'

  # Hyperparameters and model predefined constants
  VGG_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
  VGG_NORMALIZATION_STD = (0.229, 0.224, 0.225)

  # Losses layers
  CONTENT_LAYERS = {'conv4_2': 1}
  STYLE_LAYERS = {f"conv{i + 1}_1": 1e3 / 5 for i in range(5)}

  IMAGE_SIZE = 512 if torch.cuda.is_available() else 256
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
