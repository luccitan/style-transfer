# coding: utf-8
"""
Main script to run Style Transfer.
  1. It loads the content and styles images
  2. Load the VGG network and adapt it to inject content and style losses relative to content and styles images
  3. Run forward and back propagation again again starting from white noise so that it satisties the loss contraints
      better and better
  4. Output / save / plots the original and final images

Code taken, rewritten and adapted from this tutorial:
  https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
"""

import os
import time
import argparse
import logging
import logging.config
from PIL import Image

import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import Utils
from model import get_model_and_losses

logging.config.dictConfig(Utils.LOGGING_CONFIG)
LOGGER = logging.getLogger(Utils.PROJECT)

def load_image(path):
  image = Image.open(path)
  image = Utils.IMAGE_TRANSFORMER(image).unsqueeze(0) # adding a fake batch dimension of size 1
  return image.to(Utils.DEVICE, torch.float)

def show_image(image, title=''):
  plt.figure()
  image = image.cpu().clone().squeeze(0)
  image = Utils.IMAGE_UNLOADER(image)
  plt.imshow(image)
  plt.title(title)
  plt.pause(0.001)

def run_style_transfer(input_image, content_image, style_image, epochs=None, content_weight=1, style_weight=1):
  """ TODO """
  epochs = epochs or Utils.EPOCHS
  LOGGER.info(f"Loading the model and the losses")
  model, content_losses, style_losses = get_model_and_losses(content_image, style_image)
  LOGGER.debug(f"Loaded the model and the losses")

  optimizer = optim.LBFGS([input_image.requires_grad_()])

  LOGGER.info('Starting the optimization loop')

  i = 0
  progress_bar = tqdm.tqdm(total=epochs, desc='CL ? / SL ?', leave=True)
  def closure():
    input_image.data.clamp_(0, 1) # image data may be updated with values outside 0 and 1 (boundaries)
    optimizer.zero_grad()
    model(input_image)

    content_score = content_weight * sum([x.loss for x in content_losses])
    style_score = style_weight * sum([x.loss for x in style_losses])

    loss = content_score + style_score
    loss.backward()

    progress_bar.update(10)
    progress_bar.set_description(f"CL {content_score.item():3f} / SL {style_score.item():4f}")
    progress_bar.refresh()
    time.sleep(0.001)
    return loss

  while i < epochs:
    optimizer.step(closure)
    i += 1
  progress_bar.close()
  LOGGER.debug('Ended the optimization loop')

  input_image.data.clamp_(0, 1) # last out-of-the-loop boundary correction

  return input_image

def main():
  filepath = lambda p: os.path.abspath(os.path.realpath(os.path.expanduser(p)))
  parser = argparse.ArgumentParser(
    prog='run.py',
    description='Style Transfer script applying style of one image to another one')
  parser.add_argument('--style_image', '-si',
                      type=filepath,
                      metavar='STYLE_IMAGE',
                      required=True,
                      help='Path to image which style will be extracted')
  parser.add_argument('--content_image', '-ci',
                      type=filepath,
                      metavar='CONTENT_IMAGE',
                      required=True,
                      help='Path to image which content will be extracted')
  parser.add_argument('--output_image', '-o',
                      type=filepath,
                      metavar='OUTPUT_PATH',
                      required=True,
                      help='Path where the output image will be stored')
  parser.add_argument('--epochs', '-e',
                      type=int,
                      metavar='N_EPOCHS',
                      default=1,
                      help='Number of optimization steps')
  parser.add_argument('--content-weight', '-cw',
                      type=int,
                      metavar='CONTENT_WEIGHT',
                      default=Utils.CONTENT_WEIGHT,
                      help='Weight applied to the content losses')
  parser.add_argument('--style-weight', '-sw',
                      type=int,
                      metavar='STYLE_WEIGHT',
                      default=Utils.STYLE_WEIGHT,
                      help='Weight applied to the style losses')
  parser.add_argument('--white_noise', '-wn',
                      action='store_true',
                      help='If set to True, initial image is white noise and not the content image')
  args = parser.parse_args()

  style_image = load_image(args.style_image)
  content_image = load_image(args.content_image)
  assert style_image.size() == content_image.size(), "Style and content images must be the same sizes"

  if args.white_noise:
    LOGGER.debug('Preparing input image as white noise')
    input_image = torch.randn(content_image.data.size(), device=Utils.DEVICE)
  else:
    LOGGER.debug('Preparing input image as the content image')
    input_image = content_image.clone()

  LOGGER.info('Starting the Style Transfer')
  final_image = run_style_transfer(input_image, content_image, style_image,
                                   epochs=args.epochs,
                                   content_weight=args.content_weight,
                                   style_weight=args.style_weight)
  LOGGER.info('Ended the Style Transfer')

  show_image(input_image, 'Input image')
  show_image(content_image, 'Content image')
  show_image(style_image, 'Style image')
  show_image(final_image, 'Final image')

if __name__ == '__main__':
  main()
  LOGGER.info('Press a button to quit ...')
  input()
