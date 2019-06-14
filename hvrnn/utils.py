# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def concat_images(images):
    """ Concatenate images horizontally with spacing """
    spacer = np.ones([64, 1], dtype=np.float32)
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add one pixel spacing.
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size):
    """ Concatenate images horizontally and vertically with spacing """
    column_size = len(images) // row_size
    spacer_h = np.ones(
        [1, 64 * column_size + column_size - 1], dtype=np.float32)

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size * row:column_size * row + column_size]
        row_concated_images = concat_images(row_images)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size - 1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def add_noise(images, noise_type="zero_mask"):
    if noise_type == "zero_mask":
        # Add zero mask type noise
        ratio = 0.2
        return images * (np.random.random_sample(images.shape) > ratio)
    else:
        # Add salt and pepper noise
        rate = 0.1
        drop = np.random.uniform(0.0, 1.0, images.shape)
        z = np.where(drop < 0.5 * rate)
        o = np.where(np.abs(drop - 0.75 * rate) < 0.25 * rate)
        images[z] = 0.0
        images[o] = 1.0
        return images
