# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from model import VRNN


class VRNNTest(tf.test.TestCase):
    def check_list(self, val, seq_length):
        # リストかどうかのチェックとリスト長さのチェック
        self.assertTrue(isinstance(val, list))
        self.assertEqual(len(val), seq_length)

    def test_init_with_rnn_training(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16

        batch_size = 10
        seq_length = 20

        beta = 1.0
        cell_type = "merlin"
        downward_type = "concat"
        no_td_bp = True
        filter_size = 64
        for_generating = False
        binalize_output = True
        reuse = False

        model = VRNN(
            layer_size,
            h_dim,
            z_dim,
            batch_size,
            seq_length,
            beta=beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=for_generating,
            binalize_output=binalize_output,
            reuse=reuse)

        self.check_list(model.initial_state_h, layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(model.initial_state_h[i].get_shape(),
                             (batch_size, h_dim))

        self.check_list(model.last_state_h, layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(model.last_state_h[i].get_shape(),
                             (batch_size, h_dim))

        self.assertEqual(model.reconstr_loss.get_shape(), ())
        self.assertEqual(model.latent_loss.get_shape(), ())
        self.assertEqual(model.loss.get_shape(), ())

        # predict確認用変数の確認
        # TODO:
        #self.assertEqual(model.enc_mus.get_shape(),         (batch_size*seq_length, z_dim))
        #self.assertEqual(model.enc_sigma_sqs.get_shape(),   (batch_size*seq_length, z_dim))
        #self.assertEqual(model.dec_outs.get_shape(),        (batch_size*seq_length, 64*64*1))
        #self.assertEqual(model.prior_mus.get_shape(),       (batch_size*seq_length, z_dim))
        #self.assertEqual(model.prior_sigma_sqs.get_shape(), (batch_size*seq_length, z_dim))

        self.assertEqual(len(model.inputs), layer_size)
        self.assertEqual(len(model.enc_mus), layer_size)
        self.assertEqual(len(model.enc_sigma_sqs), layer_size)
        self.assertEqual(len(model.prior_mus), layer_size)
        self.assertEqual(len(model.prior_sigma_sqs), layer_size)
        self.assertEqual(len(model.zs), layer_size)
        self.assertEqual(len(model.hs), layer_size)
        self.assertEqual(len(model.dec_outs), layer_size)

    def test_init_with_rnn_generating(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 10
        seq_length = 20

        beta = 1.0
        cell_type = "merlin"
        downward_type = "concat"
        no_td_bp = True
        filter_size = 64
        for_generating = True
        binalize_output = True
        reuse = False

        model = VRNN(
            layer_size,
            h_dim,
            z_dim,
            batch_size,
            seq_length,
            beta=beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=for_generating,
            binalize_output=binalize_output,
            reuse=reuse)

        self.check_list(model.initial_state_h, layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(model.initial_state_h[i].get_shape(),
                             (batch_size, h_dim))

        self.check_list(model.last_state_h, layer_size)
        for i in range(layer_size):
            # 各階層分
            self.assertEqual(model.last_state_h[i].get_shape(),
                             (batch_size, h_dim))

        self.assertEqual(model.sampled_x.get_shape(), (batch_size, x_dim))

    def test_reuse(self):
        layer_size = 4
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 10
        seq_length = 20

        beta = 1.0
        cell_type = "merlin"
        downward_type = "concat"
        no_td_bp = False
        filter_size = 16

        # 正常にmodelがreuseできるかどうかの確認
        model0 = VRNN(
            layer_size,
            h_dim,
            z_dim,
            batch_size,
            seq_length,
            beta=beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=False,
            binalize_output=True,
            reuse=False)

        model1 = VRNN(
            layer_size,
            h_dim,
            z_dim,
            1,
            1,
            beta=beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=True,
            binalize_output=True,
            reuse=True)

    def test_conv_weight(self):
        layer_size = 3
        h_dim = 256
        z_dim = 16
        x_dim = 64 * 64 * 1

        batch_size = 10
        seq_length = 20

        beta = 1.0
        cell_type = "merlin"
        downward_type = "concat"
        no_td_bp = True
        filter_size = 64

        model = VRNN(
            layer_size,
            h_dim,
            z_dim,
            batch_size,
            seq_length,
            beta=beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=False,
            binalize_output=True,
            reuse=False)

        # get_conv_weight()でフィルタの中身が取ってこれているかどうかの確認
        weight = model.get_conv_weight()
        self.assertEqual(weight.get_shape(), (4, 4, 1, 64))


if __name__ == "__main__":
    tf.test.main()
