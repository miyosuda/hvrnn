# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
import imageio

DEBUG_SAVE_IMAGE = False


class GenBSpriteDatasetTest(unittest.TestCase):
  def test_generate(self):
    data_path = "./bsprite.npz"
    data_all = np.load(data_path)

    data_images    = data_all["images"]    # (14000, 20, 64, 64) uint8
    data_labels    = data_all["labels"]    # (14000,)            uint8
    data_pos_xs    = data_all["pos_xs"]    # (14000, 20)         float32
    data_pos_ys    = data_all["pos_ys"]    # (14000, 20)         float32
    data_speeds    = data_all["speeds"]    # (14000,)            unit8
    data_bounce_xs = data_all["bounce_xs"] # (14000, 20)         int8
    data_bounce_ys = data_all["bounce_ys"] # (14000, 20)         int8

    self.assertEqual(data_images.shape,    (14000, 20, 64, 64))
    self.assertEqual(data_labels.shape,    (14000,))
    self.assertEqual(data_pos_xs.shape,    (14000,20))
    self.assertEqual(data_pos_ys.shape,    (14000,20))
    self.assertEqual(data_speeds.shape,    (14000,))
    self.assertEqual(data_bounce_xs.shape, (14000,20))
    self.assertEqual(data_bounce_ys.shape, (14000,20))
    
    self.assertEqual(data_images.dtype,    np.uint8)
    self.assertEqual(data_labels.dtype,    np.uint8)
    self.assertEqual(data_pos_xs.dtype,    np.float32)
    self.assertEqual(data_pos_ys.dtype,    np.float32)
    self.assertEqual(data_speeds.dtype,    np.uint8)
    self.assertEqual(data_bounce_xs.dtype, np.int8)
    self.assertEqual(data_bounce_ys.dtype, np.int8)

    if DEBUG_SAVE_IMAGE:
      seq_index = 9012
      
      for i in range(20):
        img = data_images[seq_index,i]
        imageio.imwrite("b_out_{0:02}_{1:02}.png".format(seq_index, i), img)
        bx = data_bounce_xs[seq_index,i]
        by = data_bounce_ys[seq_index,i]
        if bx != 0:
          print("bx{0:02}={1}".format(i, bx))
        if by != 0:
          print("by{0:02}={1}".format(i, by))

if __name__ == '__main__':
  unittest.main()
