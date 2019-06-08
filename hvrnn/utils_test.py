# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
import utils
from scipy.misc import imsave


class UtilsTest(unittest.TestCase):
    def test_concat_images(self):
        images = np.zeros((20, 64, 64))
        concated = utils.concat_images(images)
        # (64, 1299)
        self.assertEqual(concated.shape, (64, 64 * 20 + 19))
        #imsave("concated0.png", concated)

    def test_concat_images_in_rows(self):
        images = np.zeros((30, 64, 64))
        concated = utils.concat_images_in_rows(images, 3)
        # (194, 649)
        self.assertEqual(concated.shape, (64 * 3 + 2, 64 * 10 + 9))
        #imsave("concated1.png", concated)

    def test_add_noise(self):
        images = np.ones((5, 64 * 64))
        noised_images = utils.add_noise(images)
        self.assertEqual(noised_images.shape, (5, 64 * 64))

        #noised_images = np.reshape(noised_images, [5, 64, 64])
        #imsave("noised0.png", noised_images[0])


if __name__ == '__main__':
    unittest.main()
