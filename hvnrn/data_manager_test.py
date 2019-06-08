# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from scipy.misc import imsave
from data_manager import DataManager

DEBUG_SAVE_IMAGE = False


class DataManagerTest(unittest.TestCase):
    def sub_test_get_next_train_batch(self,
                                      dataset_type="mnist",
                                      seq_length=20):
        data_manager = DataManager.get_data_manager(dataset_type)

        batch_data = data_manager.get_next_train_batch(10)

        self.assertEqual(batch_data.shape, (10, seq_length, 64, 64))
        self.assertEqual(batch_data.dtype, "float32")

        if DEBUG_SAVE_IMAGE:
            for j in range(2):
                for i in range(seq_length):
                    img = batch_data[j][i]
                    imsave("m_train_batch{0:0>2}{1:0>2}.png".format(j, i), img)

    def sub_test_get_test_batch(self, dataset_type="mnist", seq_length=20):
        data_manager = DataManager.get_data_manager(dataset_type)

        batch_data = data_manager.get_test_batch(0, 10)
        self.assertEqual(batch_data.shape, (10, seq_length, 64, 64))
        self.assertEqual(batch_data.dtype, "float32")

        if DEBUG_SAVE_IMAGE:
            for j in range(2):
                for i in range(seq_length):
                    img = batch_data[j][i]
                    imsave("m_test_batch{0:0>2}{1:0>2}.png".format(j, i), img)

    def sub_test_get_off_forecast_test_data(self,
                                            dataset_type="mnist",
                                            seq_length=20):
        data_manager = DataManager.get_data_manager(dataset_type)

        data = data_manager.get_off_forecast_test_data(0)
        self.assertEqual(data.shape, (1, seq_length, 64, 64))
        self.assertEqual(data.dtype, "float32")

        if DEBUG_SAVE_IMAGE:
            for i in range(seq_length):
                img = data[0][i]
                imsave("m_off_forecast_data{0:0>2}.png".format(i), img)

    def sub_test_get_test_data(self, dataset_type="mnist", seq_length=20):
        data_manager = DataManager.get_data_manager(dataset_type)

        data = data_manager.get_test_data(0)

        self.assertEqual(data.shape, (1, seq_length, 64, 64))
        self.assertEqual(data.dtype, "float32")

    def sub_test_common_properties(self,
                                   data_manager,
                                   seq_length=10,
                                   train_data_size=5000,
                                   test_data_size=1000,
                                   using_test_data_size=500):

        # imagesはfloat32になっている
        self.assertEqual(data_manager.raw_train_images.dtype, "uint8")
        self.assertEqual(data_manager.raw_test_images.dtype, "uint8")

        # imagesのshape確認
        self.assertEqual(data_manager.raw_train_images.shape,
                         (train_data_size, seq_length, 64, 64))
        self.assertEqual(data_manager.raw_test_images.shape,
                         (test_data_size, seq_length, 64, 64))

        # imagesの値の範囲確認
        self.assertLessEqual(np.amax(data_manager.raw_train_images), 255)
        self.assertGreaterEqual(np.amin(data_manager.raw_train_images), 0)
        self.assertLessEqual(np.amax(data_manager.raw_test_images), 255)
        self.assertGreaterEqual(np.amin(data_manager.raw_test_images), 0)

        # フィールドの確認
        self.assertEqual(data_manager.train_data_size, train_data_size)
        self.assertEqual(data_manager.test_data_size, test_data_size)
        self.assertEqual(data_manager.using_test_data_size,
                         using_test_data_size)
        self.assertEqual(data_manager.seq_length, seq_length)
        self.assertEqual(data_manager.w, 64)
        self.assertEqual(data_manager.h, 64)

    # MNIST
    def test_mnist(self):
        dataset_type = "mnist"
        seq_length = 20
        train_data_size = 9000
        test_data_size = 5000
        using_test_data_size = 500

        # 共通のテスト
        data_manager = DataManager.get_data_manager(dataset_type)

        self.sub_test_common_properties(data_manager, seq_length,
                                        train_data_size, test_data_size,
                                        using_test_data_size)

        if DEBUG_SAVE_IMAGE:
            for i in range(20):
                img = data_manager.raw_train_images[0][i]
                imsave("m_out{}.png".format(i), img)

        self.sub_test_get_next_train_batch(dataset_type, seq_length)
        self.sub_test_get_test_batch(dataset_type, seq_length)
        self.sub_test_get_off_forecast_test_data(dataset_type, seq_length)
        self.sub_test_get_test_data(dataset_type, seq_length)

        # 非共通の部分のテスト
        train_analysis_data = data_manager.train_analysis_data
        test_analysis_data = data_manager.test_analysis_data

        self.assertEqual(train_analysis_data["pos_x"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["pos_y"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["label"].shape,
                         (train_data_size, ))
        self.assertEqual(train_analysis_data["speed"].shape,
                         (train_data_size, ))
        self.assertEqual(train_analysis_data["bounce_x"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["bounce_y"].shape,
                         (train_data_size, seq_length))

        self.assertEqual(test_analysis_data["pos_x"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["pos_y"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["label"].shape, (test_data_size, ))
        self.assertEqual(test_analysis_data["speed"].shape, (test_data_size, ))
        self.assertEqual(test_analysis_data["bounce_x"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["bounce_y"].shape,
                         (test_data_size, seq_length))

    # BSprite
    def test_bsprite(self):
        dataset_type = "bsprite"
        seq_length = 20
        train_data_size = 9000
        test_data_size = 5000
        using_test_data_size = 500

        # 共通のテスト
        data_manager = DataManager.get_data_manager(dataset_type)

        self.sub_test_common_properties(data_manager, seq_length,
                                        train_data_size, test_data_size,
                                        using_test_data_size)

        if DEBUG_SAVE_IMAGE:
            for i in range(20):
                img = data_manager.raw_train_images[0][i]
                imsave("b_out{}.png".format(i), img)

        self.sub_test_get_next_train_batch(dataset_type, seq_length)
        self.sub_test_get_test_batch(dataset_type, seq_length)
        self.sub_test_get_off_forecast_test_data(dataset_type, seq_length)
        self.sub_test_get_test_data(dataset_type, seq_length)

        # 非共通の部分のテスト
        train_analysis_data = data_manager.train_analysis_data
        test_analysis_data = data_manager.test_analysis_data

        self.assertEqual(train_analysis_data["pos_x"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["pos_y"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["label"].shape,
                         (train_data_size, ))
        self.assertEqual(train_analysis_data["speed"].shape,
                         (train_data_size, ))
        self.assertEqual(train_analysis_data["bounce_x"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["bounce_y"].shape,
                         (train_data_size, seq_length))

        self.assertEqual(test_analysis_data["pos_x"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["pos_y"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["label"].shape, (test_data_size, ))
        self.assertEqual(test_analysis_data["speed"].shape, (test_data_size, ))
        self.assertEqual(test_analysis_data["bounce_x"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["bounce_y"].shape,
                         (test_data_size, seq_length))

    # Oculomotor
    def test_oculomotor(self):
        dataset_type = "oculomotor"
        seq_length = 20
        train_data_size = 9500
        test_data_size = 500
        using_test_data_size = 500

        # 共通のテスト
        data_manager = DataManager.get_data_manager(dataset_type)

        self.sub_test_common_properties(data_manager, seq_length,
                                        train_data_size, test_data_size,
                                        using_test_data_size)

        if DEBUG_SAVE_IMAGE:
            for i in range(20):
                img = data_manager.raw_train_images[0][i]
                imsave("o_out{}.png".format(i), img)

        self.sub_test_get_next_train_batch(dataset_type, seq_length)
        self.sub_test_get_test_batch(dataset_type, seq_length)
        self.sub_test_get_off_forecast_test_data(dataset_type, seq_length)
        self.sub_test_get_test_data(dataset_type, seq_length)

        # 非共通の部分のテスト
        train_analysis_data = data_manager.train_analysis_data
        test_analysis_data = data_manager.test_analysis_data

        self.assertEqual(train_analysis_data["action_x"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["action_y"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["angle_x"].shape,
                         (train_data_size, seq_length))
        self.assertEqual(train_analysis_data["angle_y"].shape,
                         (train_data_size, seq_length))

        self.assertEqual(test_analysis_data["action_x"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["action_y"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["angle_x"].shape,
                         (test_data_size, seq_length))
        self.assertEqual(test_analysis_data["angle_y"].shape,
                         (test_data_size, seq_length))


if __name__ == '__main__':
    unittest.main()
