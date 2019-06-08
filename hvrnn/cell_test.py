# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from cell import VariationalRNNCell, MerlinRNNCell


class VariationalRNNCellTest(tf.test.TestCase):
    def check_list(self, val, seq_length):
        # リストかどうかのチェックとリスト長さのチェック
        self.assertTrue(isinstance(val, list))
        self.assertEqual(len(val), seq_length)

    def test_cell_rnn_training(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 10
        seq_length = 20
        downward_type = "to_prior"
        no_td_bp = True
        filter_size = 64

        cell = VariationalRNNCell(
            layer_size,
            h_dim,
            z_dim,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=False,
            binalize_output=True,
            reuse=False)

        # ゼロ初期化された初期state
        initial_state_h = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

        # Shapeのチェック
        self.assertTrue(isinstance(initial_state_h, list))
        self.assertEqual(len(initial_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(initial_state_h[i].get_shape(),
                             (batch_size, h_dim))

        input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, seq_length, x_dim],
            name="input_data")

        with tf.variable_scope("inputs"):
            # ステップ数とバッチを入れ替え
            inputs = tf.transpose(input_data, [1, 0, 2])
            inputs = tf.reshape(inputs, [-1, x_dim])
            # (seq_length * batch_size, x_dim)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            # seq_length * (batch_size, n_hidden)
            inputs = tf.split(
                axis=0, num_or_size_splits=seq_length, value=inputs)
            # Tensorのリスト [seq_length]
            # それぞれが(batch, x_dim)

        outputs, last_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state_h)

        # outputsは、[seq_length] のリスト
        self.check_list(outputs, seq_length)

        for i in range(seq_length):
            # outputsのそれぞれの要素は、(enc_mu, enc_sigma....)のタプル
            output = outputs[i]
            inputs, enc_mus, enc_sigmas, dec_mus, prior_mus, prior_sigmas, \
              zs, hs = output

            self.check_list(inputs, layer_size)
            self.check_list(enc_mus, layer_size)
            self.check_list(enc_sigmas, layer_size)
            self.check_list(dec_mus, layer_size)
            self.check_list(prior_mus, layer_size)
            self.check_list(prior_sigmas, layer_size)
            self.check_list(zs, layer_size)
            self.check_list(hs, layer_size)

            for j in range(layer_size):
                # shapeは、(batch_size, z_dim)とか、(batch_size, x_dim)
                if j == 0:
                    # 最下階層は画像xの次元 (64*64)
                    self.assertEqual(inputs[j].get_shape(),
                                     (batch_size, x_dim))
                    self.assertEqual(dec_mus[j].get_shape(),
                                     (batch_size, x_dim))
                elif j == 1:
                    self.assertEqual(inputs[j].get_shape(),
                                     (batch_size, 16 * 16 * 64))
                    self.assertEqual(dec_mus[j].get_shape(),
                                     (batch_size, 16 * 16 * 64))
                elif j == 2:
                    self.assertEqual(inputs[j].get_shape(), (batch_size, 1024))
                    self.assertEqual(dec_mus[j].get_shape(),
                                     (batch_size, 1024))
                elif j == 3:
                    self.assertEqual(inputs[j].get_shape(), (batch_size, 256))
                    self.assertEqual(dec_mus[j].get_shape(), (batch_size, 256))

                self.assertEqual(enc_mus[j].get_shape(), (batch_size, z_dim))
                self.assertEqual(enc_sigmas[j].get_shape(),
                                 (batch_size, z_dim))
                self.assertEqual(prior_mus[j].get_shape(), (batch_size, z_dim))
                self.assertEqual(prior_sigmas[j].get_shape(),
                                 (batch_size, z_dim))
                self.assertEqual(zs[j].get_shape(), (batch_size, z_dim))
                self.assertEqual(hs[j].get_shape(), (batch_size, h_dim))

        # last_stateのhは、[(batch_size, h_dim)...]のlist
        last_state_h = last_state

        self.assertTrue(isinstance(last_state_h, list))
        self.assertEqual(len(last_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(last_state_h[i].get_shape(), (batch_size, h_dim))

    # [Generating RNN]
    def test_cell_rnn_generating(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 1
        seq_length = 20
        downward_type = "concat"
        no_td_bp = False
        filter_size = 64

        cell = VariationalRNNCell(
            layer_size,
            h_dim,
            z_dim,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=True,
            binalize_output=True,
            reuse=False)

        # ゼロ初期化された初期state
        initial_state_h = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

        # Shapeのチェック
        self.assertTrue(isinstance(initial_state_h, list))
        self.assertEqual(len(initial_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(initial_state_h[i].get_shape(),
                             (batch_size, h_dim))

        input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, seq_length, x_dim],
            name="input_data")

        initial_state = initial_state_h

        with tf.variable_scope("inputs"):
            # ステップ数とバッチを入れ替え
            inputs = tf.transpose(input_data, [1, 0, 2])
            inputs = tf.reshape(inputs, [-1, x_dim])
            # (seq_length * batch_size, x_dim)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            # seq_length * (batch_size, n_hidden)
            inputs = tf.split(
                axis=0, num_or_size_splits=seq_length, value=inputs)
            # Tensorのリスト [seq_length]
            # それぞれが(batch, x_dim)

        outputs, last_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state)

        # outputsは、[seq_length] のリスト
        self.check_list(outputs, seq_length)

        for i in range(seq_length):
            # outputsの各要素は(batch, 64*64)の次元
            output = outputs[i]
            x_out = output
            self.assertEqual(x_out.get_shape(), (batch_size, x_dim))

        # last_stateのhは、[(batch_size, h_dim)...]のlist
        last_state_h = last_state

        # last_stateのhは、(batch_size, h_dim)のshape
        self.assertTrue(isinstance(last_state_h, list))
        self.assertEqual(len(last_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(last_state_h[i].get_shape(), (batch_size, h_dim))


class MerlinRNNCellTest(tf.test.TestCase):
    def check_list(self, val, seq_length):
        # リストかどうかのチェックとリスト長さのチェック
        self.assertTrue(isinstance(val, list))
        self.assertEqual(len(val), seq_length)

    def test_cell_rnn_training(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 10
        seq_length = 20
        downward_type = "to_prior"
        no_td_bp = True
        filter_size = 64

        cell = MerlinRNNCell(
            layer_size,
            h_dim,
            z_dim,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=False,
            binalize_output=True,
            reuse=False)

        # ゼロ初期化された初期state
        initial_state_h = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

        # Shapeのチェック
        self.assertTrue(isinstance(initial_state_h, list))
        self.assertEqual(len(initial_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(initial_state_h[i].get_shape(),
                             (batch_size, h_dim))

        input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, seq_length, x_dim],
            name="input_data")

        with tf.variable_scope("inputs"):
            # ステップ数とバッチを入れ替え
            inputs = tf.transpose(input_data, [1, 0, 2])
            inputs = tf.reshape(inputs, [-1, x_dim])
            # (seq_length * batch_size, x_dim)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            # seq_length * (batch_size, n_hidden)
            inputs = tf.split(
                axis=0, num_or_size_splits=seq_length, value=inputs)
            # Tensorのリスト [seq_length]
            # それぞれが(batch, x_dim)

        outputs, last_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state_h)

        # outputsは、[seq_length] のリスト
        self.check_list(outputs, seq_length)

        for i in range(seq_length):
            # outputsのそれぞれの要素は、(enc_mu, enc_sigma....)のタプル
            output = outputs[i]
            inputs, enc_mus, enc_sigmas, dec_mus, prior_mus, prior_sigmas, \
              zs, hs = output

            self.check_list(inputs, layer_size)
            self.check_list(enc_mus, layer_size)
            self.check_list(enc_sigmas, layer_size)
            self.check_list(dec_mus, layer_size)
            self.check_list(prior_mus, layer_size)
            self.check_list(prior_sigmas, layer_size)
            self.check_list(zs, layer_size)
            self.check_list(hs, layer_size)

            for j in range(layer_size):
                # shapeは、(batch_size, z_dim)とか、(batch_size, x_dim)
                if j == 0:
                    # 最下階層は画像xの次元 (64*64)
                    self.assertEqual(inputs[j].get_shape(),
                                     (batch_size, x_dim))
                    self.assertEqual(dec_mus[j].get_shape(),
                                     (batch_size, x_dim))
                elif j == 1:
                    self.assertEqual(inputs[j].get_shape(),
                                     (batch_size, 16 * 16 * 64))
                    self.assertEqual(dec_mus[j].get_shape(),
                                     (batch_size, 16 * 16 * 64))
                elif j == 2:
                    self.assertEqual(inputs[j].get_shape(), (batch_size, 1024))
                    self.assertEqual(dec_mus[j].get_shape(),
                                     (batch_size, 1024))
                elif j == 3:
                    self.assertEqual(inputs[j].get_shape(), (batch_size, 256))
                    self.assertEqual(dec_mus[j].get_shape(), (batch_size, 256))

                self.assertEqual(enc_mus[j].get_shape(), (batch_size, z_dim))
                self.assertEqual(enc_sigmas[j].get_shape(),
                                 (batch_size, z_dim))
                self.assertEqual(prior_mus[j].get_shape(), (batch_size, z_dim))
                self.assertEqual(prior_sigmas[j].get_shape(),
                                 (batch_size, z_dim))
                self.assertEqual(zs[j].get_shape(), (batch_size, z_dim))
                self.assertEqual(hs[j].get_shape(), (batch_size, h_dim))

        # last_stateのhは、[(batch_size, h_dim)...]のlist
        last_state_h = last_state

        self.assertTrue(isinstance(last_state_h, list))
        self.assertEqual(len(last_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(last_state_h[i].get_shape(), (batch_size, h_dim))

    # [Generating RNN]
    def test_cell_rnn_generating(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 1
        seq_length = 20
        downward_type = "concat"
        no_td_bp = False
        filter_size = 64

        cell = MerlinRNNCell(
            layer_size,
            h_dim,
            z_dim,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=True,
            binalize_output=True,
            reuse=False)

        # ゼロ初期化された初期state
        initial_state_h = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

        # Shapeのチェック
        self.assertTrue(isinstance(initial_state_h, list))
        self.assertEqual(len(initial_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(initial_state_h[i].get_shape(),
                             (batch_size, h_dim))

        input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, seq_length, x_dim],
            name="input_data")

        initial_state = initial_state_h

        with tf.variable_scope("inputs"):
            # ステップ数とバッチを入れ替え
            inputs = tf.transpose(input_data, [1, 0, 2])
            inputs = tf.reshape(inputs, [-1, x_dim])
            # (seq_length * batch_size, x_dim)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            # seq_length * (batch_size, n_hidden)
            inputs = tf.split(
                axis=0, num_or_size_splits=seq_length, value=inputs)
            # Tensorのリスト [seq_length]
            # それぞれが(batch, x_dim)

        outputs, last_state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=initial_state)

        # outputsは、[seq_length] のリスト
        self.check_list(outputs, seq_length)

        for i in range(seq_length):
            # outputsの各要素は(batch, 64*64)の次元
            output = outputs[i]
            x_out = output
            self.assertEqual(x_out.get_shape(), (batch_size, x_dim))

        # last_stateのhは、[(batch_size, h_dim)...]のlist
        last_state_h = last_state

        # last_stateのhは、(batch_size, h_dim)のshape
        self.assertTrue(isinstance(last_state_h, list))
        self.assertEqual(len(last_state_h), layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(last_state_h[i].get_shape(), (batch_size, h_dim))


if __name__ == "__main__":
    tf.test.main()
