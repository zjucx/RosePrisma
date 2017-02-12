########################################################################
#
# The IMG help function for RosePrisma.
#
# This file offer the operations of img
# load_image、save_image and plot_images
#
# Implemented in Python 2.7 with TensorFlow v0.11.0rc0
#
########################################################################
#
# This file is part of the RosePrisma project at:
#
# https://github.com/zjucx/RosePrisma.git
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2017 by ZjuCx
#
########################################################################

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_img(filename, max_size=None):
    im = Image.open('lena.png')
    if max_size is not None:
        factor = max_size / np.max(im.size)
        # Scale the image's height and width.
        size = np.array(im.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        im = im.resize(size, Image.LANCZOS)
    return np.float(im)

def save_img(im, filename):
    im = np.clip(im, 0.0, 255.0)
    im = im.astype(np.uint8)
    with Image.open(filename, 'wb') as file:
        Image.fromarray(im).save(file, 'jpeg')

def plot_img(im):
    im = np.clip(im, 0.0, 255.0)
    im = im.astype(np.uint8)
    Image.fromarray(im).show()

def plot_imgs(content_img, style_img, mix_img):
    # Create figure with sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Use interpolation to smooth pixels?
    smooth = True
    
    # Interpolation type.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plot the content-image.
    # Note that the pixel-values are normalized to
    # the [0.0, 1.0] range by dividing with 255.
    ax = axes.flat[0]
    ax.imshow(content_img / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plot the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mix_img / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plot the style-image
    ax = axes.flat[2]
    ax.imshow(style_img / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Remove ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()