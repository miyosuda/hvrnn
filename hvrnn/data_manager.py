# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

mnist_data_path = "../data/moving_mnist/moving_mnist1.npz"  # 数字1個のmoving mnist dataset
bsprite_data_path = "../data/bsprite/bsprite.npz"  # bouncing sprite dataset
oculomotor_data_path = "../data/oculomotor/oculomotor.npz"  # Oculomotor dataset


class DataManager(object):
    """ Load sequences of images """

    @staticmethod
    def get_data_manager(dataset_type="bsprite"):
        if dataset_type == "mnist":
            return MnistDataManager()
        elif dataset_type == "bsprite":
            return BSpriteDataManager()
        elif dataset_type == "oculomotor":
            return OculomotorDataManager()
        
    def __init__(self):
        self.train_index_pos = -1

    def convert_images(self, images):
        return images.astype(np.float32) / 255.0

    def get_next_train_batch(self, batch_size):
        if self.train_index_pos < 0 or \
           self.train_index_pos + batch_size > self.train_data_size:
            self.train_indices = list(range(self.train_data_size))
            random.shuffle(self.train_indices)
            self.train_index_pos = 0
        selected_indices = self.train_indices[self.train_index_pos:
                                              self.train_index_pos+batch_size]
        images = self.raw_train_images[selected_indices, :, :, :]
        self.train_index_pos += batch_size
        return self.convert_images(images)

    def get_test_batch(self, data_index, batch_size):
        indices = list(range(data_index, data_index + batch_size))
        images = self.raw_test_images[indices, :, :, :]
        return self.convert_images(images)

    def get_off_forecast_test_data(self, data_index):
        """ Image time seris data to cause prediction error """
        index0 = data_index * 2
        index1 = data_index * 2 + 1
        # Get 0~9 and 10~19 from different sequences.
        data0 = self.raw_test_images[index0, 0:self.seq_length // 2, :, :]
        data1 = self.raw_test_images[index1, self.seq_length // 2:, :, :]
        images = np.concatenate([data0, data1], axis=0)
        images = np.reshape(images, [1, self.seq_length, self.w, self.h])
        return self.convert_images(images)

    def get_test_data(self, data_index):
        images = self.raw_test_images[data_index, :, :, :]
        images = np.reshape(images, [1, self.seq_length, self.w, self.h])
        converted_images = self.convert_images(images)
        return converted_images

    def split_train_test_data(self, data, train_data_size):
        # Split data into train/test.
        train_data = data[0:train_data_size]
        test_data = data[train_data_size:]
        return train_data, test_data


class MnistDataManager(DataManager):
    """ Load sequences of images of moving MNIST dataset. """

    def __init__(self):
        DataManager.__init__(self)

        # Load data
        data_all = np.load(mnist_data_path)
        data_images = data_all["images"]        # (10000, 20, 64, 64) uint8
        data_labels = data_all["labels"]        # (10000, 1)          uint8
        data_pos_xs = data_all["pos_xs"]        # (10000, 20, 1)      float32
        data_pos_ys = data_all["pos_ys"]        # (10000, 20, 1)      float32
        data_speeds = data_all["speeds"]        # (10000, 1)          unit8
        data_bounce_xs = data_all["bounce_xs"]  # (14000, 20, 1)   int8
        data_bounce_ys = data_all["bounce_ys"]  # (14000, 20, 1)   int8

        # Get data dimensions
        total_data_size, self.seq_length, self.w, self.h = data_images.shape
        self.train_data_size = 9000
        self.test_data_size = 5000
        # Test data is used only 500, but created 5000.
        # (For the regression analyze, test data is prepared larger)
        self.using_test_data_size = 500

        # Image data (uint8)
        self.raw_train_images = data_images[
            0:self.train_data_size, :, :, :]  # (9000, 20, 64, 64)
        self.raw_test_images = data_images[
            self.train_data_size:, :, :, :]  # (5000, 20, 64, 64)

        # ラベル
        train_labels = data_labels[0:self.train_data_size].reshape(
            [self.train_data_size])
        test_labels = data_labels[self.train_data_size:].reshape(
            [self.test_data_size])

        # Positions.
        train_pos_xs = data_pos_xs[0:self.train_data_size].reshape(
            [self.train_data_size, self.seq_length])
        test_pos_xs = data_pos_xs[self.train_data_size:].reshape(
            [self.test_data_size, self.seq_length])

        train_pos_ys = data_pos_ys[0:self.train_data_size].reshape(
            [self.train_data_size, self.seq_length])
        test_pos_ys = data_pos_ys[self.train_data_size:].reshape(
            [self.test_data_size, self.seq_length])

        # Speed
        train_speeds = data_speeds[0:self.train_data_size].reshape(
            [self.train_data_size])
        test_speeds = data_speeds[self.train_data_size:].reshape(
            [self.test_data_size])

        # Bounce
        train_bounce_xs = data_bounce_xs[0:self.train_data_size].reshape(
            [self.train_data_size, self.seq_length])
        test_bounce_xs = data_bounce_xs[self.train_data_size:].reshape(
            [self.test_data_size, self.seq_length])

        train_bounce_ys = data_bounce_ys[0:self.train_data_size].reshape(
            [self.train_data_size, self.seq_length])
        test_bounce_ys = data_bounce_ys[self.train_data_size:].reshape(
            [self.test_data_size, self.seq_length])

        self.train_analysis_data = {
            "label": train_labels,
            "pos_x": train_pos_xs,
            "pos_y": train_pos_ys,
            "speed": train_speeds,
            "bounce_x": train_bounce_xs,
            "bounce_y": train_bounce_ys
        }

        self.test_analysis_data = {
            "label": test_labels,
            "pos_x": test_pos_xs,
            "pos_y": test_pos_ys,
            "speed": test_speeds,
            "bounce_x": test_bounce_xs,
            "bounce_y": test_bounce_ys
        }


class BSpriteDataManager(DataManager):
    """ Load sequences of images of bouncing sprite dataset. """

    def __init__(self):
        DataManager.__init__(self)

        # Load data
        data_all = np.load(bsprite_data_path)
        data_images = data_all["images"]        # (14000, 20, 64, 64) uint8
        data_labels = data_all["labels"]        # (14000,)            uint8
        data_pos_xs = data_all["pos_xs"]        # (14000, 20)         float32
        data_pos_ys = data_all["pos_ys"]        # (14000, 20)         float32
        data_speeds = data_all["speeds"]        # (14000,)            uint8
        data_bounce_xs = data_all["bounce_xs"]  # (14000, 20)         int8
        data_bounce_ys = data_all["bounce_ys"]  # (14000, 20)         int8

        # Get data dimensions
        total_data_size, self.seq_length, self.w, self.h = data_images.shape
        self.train_data_size = 9000
        self.test_data_size = 5000
        # テストデータは5000個しか利用しないが、実際には5000個ある
        # (リッジ回帰でのデータ作成用に利用する為多く用意している)
        self.using_test_data_size = 500

        # 画像データ (uint8)
        self.raw_train_images, self.raw_test_images = \
          self.split_train_test_data(data_images, self.train_data_size)
        # (9000, 20, 64, 64), (5000, 20, 64, 64)

        # ラベル
        train_labels, test_labels = self.split_train_test_data(
            data_labels, self.train_data_size)

        # 位置
        train_pos_xs, test_pos_xs = self.split_train_test_data(
            data_pos_xs, self.train_data_size)
        train_pos_ys, test_pos_ys = self.split_train_test_data(
            data_pos_ys, self.train_data_size)

        # 速度
        train_speeds, test_speeds = self.split_train_test_data(
            data_speeds, self.train_data_size)

        # Bounce
        train_bounce_xs, test_bounce_xs = self.split_train_test_data(
            data_bounce_xs, self.train_data_size)
        train_bounce_ys, test_bounce_ys = self.split_train_test_data(
            data_bounce_ys, self.train_data_size)

        self.train_analysis_data = {
            "label": train_labels,
            "pos_x": train_pos_xs,
            "pos_y": train_pos_ys,
            "speed": train_speeds,
            "bounce_x": train_bounce_xs,
            "bounce_y": train_bounce_ys
        }

        self.test_analysis_data = {
            "label": test_labels,
            "pos_x": test_pos_xs,
            "pos_y": test_pos_ys,
            "speed": test_speeds,
            "bounce_x": test_bounce_xs,
            "bounce_y": test_bounce_ys
        }


class OculomotorDataManager(DataManager):
    """ Load sequences of images of oculomotor dataset. """

    def __init__(self):
        DataManager.__init__(self)

        # Load data
        data_all = np.load(oculomotor_data_path)
        data_images = data_all["images"]   # (5000, 20, 64, 64) uint8
        data_actions = data_all["actions"] # (5000, 20, 2) float32
        data_angles  = data_all["angles"]  # (5000, 20, 2) float32

        data_action_xs = data_actions[:, :, 0]
        data_action_ys = data_actions[:, :, 1]
        data_angle_xs  = data_angles[:, :, 0]
        data_angle_ys  = data_angles[:, :, 1]

        # Get data dimensions
        total_data_size, self.seq_length, self.w, self.h = data_images.shape
        self.train_data_size = 9500
        self.test_data_size = 500
        self.using_test_data_size = 500  # adjust for bach size

        # 画像データ (uint8)
        self.raw_train_images, self.raw_test_images = \
          self.split_train_test_data(data_images, self.train_data_size)
        # (9500, 20, 64, 64), (500, 20, 64, 64)

        # Action
        train_action_xs, test_action_xs = self.split_train_test_data(
            data_action_xs, self.train_data_size)
        train_action_ys, test_action_ys = self.split_train_test_data(
            data_action_ys, self.train_data_size)

        # Angle
        train_angle_xs, test_angle_xs = self.split_train_test_data(
            data_angle_xs, self.train_data_size)
        train_angle_ys, test_angle_ys = self.split_train_test_data(
            data_angle_ys, self.train_data_size)

        self.train_analysis_data = {
            "action_x": train_action_xs,
            "action_y": train_action_ys,
            "angle_x": train_angle_xs,
            "angle_y": train_angle_ys
        }

        self.test_analysis_data = {
            "action_x": test_action_xs,
            "action_y": test_action_ys,
            "angle_x": test_angle_xs,
            "angle_y": test_angle_ys
        }
