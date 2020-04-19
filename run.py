# coding: utf-8
"""
Main script to run Style Transfer.
  1. It loads the content and styles images
  2. Load the VGG network and adapt it to inject content and style losses relative to content and styles images
  3. Run forward and back propagation again again starting from white noise so that it satisties the loss contraints
      better and better
  4. Output / save / plots the original and final images

Implementation of :
  https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
Code taken, rewritten and adapted from this source:
  https://github.com/leongatys/PytorchNeuralStyleTransfer
"""

import os
import argparse
import logging
import logging.config
from PIL import Image

import tqdm
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from model import VGG
from utils import Utils
from losses import get_content_loss, get_style_loss
from processing import Processing

logging.config.dictConfig(Utils.LOGGING_CONFIG)
LOGGER = logging.getLogger(Utils.PROJECT)

def show_image(image, title='', save_path=None):
  plt.figure()
  image = Processing.postprocessor(image)
  plt.imshow(image)
  plt.title(title)
  if save_path is not None:
    plt.savefig(save_path)
  plt.pause(0.001)

def run_style_transfer(final_image, content_image, style_image, epochs=1):
  """ TODO """
  LOGGER.info(f"Loading the model and the losses")
  model = VGG().to(Utils.DEVICE)
  optimizer = optim.LBFGS([final_image.requires_grad_()])
  style_fns = {layer: get_style_loss(out) for layer, out in model(style_image, list(Utils.STYLE_LAYERS.keys())).items()}
  content_fns = {layer: get_content_loss(out) for layer, out in model(content_image, list(Utils.CONTENT_LAYERS.keys())).items()}
  LOGGER.debug(f"Loaded the model and the losses")

  i = 0
  progress_bar = tqdm.tqdm(total=epochs, desc='CL ? / SL ?', leave=True)
  def closure():
    optimizer.zero_grad()
    outs = model(final_image, [*Utils.STYLE_LAYERS, *Utils.CONTENT_LAYERS])

    style_loss = sum([weight * style_fns[layer](outs[layer]) for layer, weight in Utils.STYLE_LAYERS.items()])
    content_loss = sum([weight * content_fns[layer](outs[layer]) for layer, weight in Utils.CONTENT_LAYERS.items()])
    loss = style_loss + content_loss
    loss.backward()
    progress_bar.set_description(f"CL {content_loss.item():3f} / SL {style_loss.item():4f}")
    progress_bar.refresh()
    return loss

  LOGGER.info('Starting the optimization loop')
  while i < epochs:
    optimizer.step(closure)
    progress_bar.update(1)
    progress_bar.refresh()
    i += 1
  progress_bar.close()
  LOGGER.debug('Ended the optimization loop')

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
  parser.add_argument('--output_folder', '-of',
                      type=filepath,
                      metavar='OUTPUT_PATH',
                      required=True,
                      help='Folder path where the output image will be stored')
  parser.add_argument('--epochs', '-e',
                      type=int,
                      metavar='N_EPOCHS',
                      default=1,
                      help='Number of optimization steps')
  parser.add_argument('--white_noise', '-wn',
                      action='store_true',
                      help='If set to True, initial image is white noise and not the content image')
  args = parser.parse_args()

  style_image = Image.open(args.style_image).convert('RGB')
  style_image = Processing.preprocessor(style_image)
  content_image = Image.open(args.content_image).convert('RGB')
  content_image = Processing.preprocessor(content_image)
  assert style_image.size() == content_image.size(), "Style and content images must be the same sizes"

  if args.white_noise:
    LOGGER.debug('Preparing input image as white noise')
    input_image = torch.randn(content_image.data.size(), device=Utils.DEVICE)
  else:
    LOGGER.debug('Preparing input image as the content image')
    input_image = content_image.clone()

  final_image = input_image.clone()
  LOGGER.info('Starting the Style Transfer')
  run_style_transfer(final_image, content_image, style_image, epochs=args.epochs)
  LOGGER.info('Ended the Style Transfer')

  os.makedirs(args.output_folder, exist_ok=True)
  show_image(input_image, 'Input image', save_path=os.path.join(args.output_folder, 'input_image'))
  show_image(content_image, 'Content image', save_path=os.path.join(args.output_folder, 'content_image'))
  show_image(style_image, 'Style image', save_path=os.path.join(args.output_folder, 'style_image'))
  show_image(final_image, 'Final image', save_path=os.path.join(args.output_folder, 'final_image'))

if __name__ == '__main__':
  main()
  LOGGER.info('Press a button to quit ...')
  input()
