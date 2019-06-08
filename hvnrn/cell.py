# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def sample_gauss(mu, log_sigma_sq):
    eps_shape = tf.shape(mu)
    eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32)
    # z = mu + sigma * epsilon
    ret = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
    return ret


def sample_bernoulli(mu):
    shape = tf.shape(mu)
    ret = tf.where(
        tf.random_uniform(shape) - mu < 0, tf.ones(shape), tf.zeros(shape))
    return ret


class VariationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell."""

    def __init__(self,
                 layer_size,
                 h_dim,
                 z_dim,
                 downward_type,
                 no_td_bp,
                 filter_size,
                 for_generating,
                 binalize_output,
                 reuse):

        self.layer_size      = layer_size
        self.n_h             = h_dim  # hiddenの次元
        self.n_z             = z_dim  # zの次元

        self.downward_type   = downward_type
        self.no_td_bp        = no_td_bp  # Top-Down信号に対するBPをしないかどうか
        self.filter_size     = filter_size
        self.for_generating  = for_generating
        self.binalize_output = binalize_output
        self.reuse           = reuse

        self.cells = [
            tf.contrib.rnn.BasicRNNCell(self.n_h)
            for _ in range(self.layer_size)
        ]

    @property
    def state_size(self):
        # zero_state()から呼ばれる
        return [self.n_h] * self.layer_size

    @property
    def output_size(self):
        # static_rnnを自前で実装した場合最終的には使われない
        return self.n_h

    def process_downward_h_join(self, h0, h1, layer_index):
        # 下方向に来た流れをどう合流するか
        # 上の階層のhとconcatする
        with tf.variable_scope("downward_h_join" + str(layer_index)):
            if self.downward_type == "concat":
                # Concatして、fcでサイズを揃える場合
                h_c = tf.concat(axis=1, values=(h0, h1))
                h = tf.layers.dense(
                    h_c,
                    self.n_h,
                    activation=tf.nn.relu,
                    name="fc1")
            elif self.downward_type == "gated_add":
                # gate addする場合
                gate = tf.get_variable(
                    "gate",
                    shape=h1.get_shape()[1:],
                    initializer=tf.constant_initializer(0.1))
                h = h0 + tf.multiply(gate, h1)
            elif self.downward_type == "add":
                # 加算する場合
                h = h0 + h1
        return h

    # (ここは各階層同じ)
    def process_prior(self, h, layer_index):
        # hから事前分布を計算する
        with tf.variable_scope("prior" + str(layer_index)):
            prior_hidden = tf.layers.dense(
                h,
                self.n_h,  # hiddenのサイズと同じにしている
                activation=tf.nn.relu,
                name="fc1")
            prior_mu = tf.layers.dense(
                prior_hidden, self.n_z, name="mu")
            prior_log_sigma_sq = tf.layers.dense(
                prior_hidden, self.n_z, name="sigma")
        return prior_mu, prior_log_sigma_sq

    def process_phi_x_conv(self, x, layer_index):
        """ 入力xをzとconcatしてhにリカレントする為にxの次元を変える為の層 """

        # 入力xをx_1に
        with tf.variable_scope("phi_x" + str(layer_index)):
            # flatになっているので直す
            if layer_index == 0:
                x = tf.reshape(x, [-1, 64, 64, 1])
                # (-1, 64, 64, 1)
            else:
                x = tf.reshape(x, [-1, 16, 16, self.filter_size])
                # (-1, 16, 16, filter_size)

            h1 = tf.layers.conv2d(
                x,
                filters=self.filter_size,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv1")
            # (-1, 32, 32, filter_size) or (-1, 8, 8, filter_size)
            h2 = tf.layers.conv2d(
                h1,
                filters=self.filter_size,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv2")
            # (-1, 16, 16, filter_size) or (-1, 4, 4, filter_size)

            # 上にあげる為のものをとっておく
            h2_flat = tf.layers.flatten(h2)

            fc_input = h2_flat

            x_1 = tf.layers.dense(
                fc_input,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")

        return x_1, h2_flat

    def process_phi_x_fc(self, x, layer_index):
        """ 入力xをzとconcatしてhにリカレントする為にxの次元を変える為の層 """

        # 入力xをx_1に
        with tf.variable_scope("phi_x" + str(layer_index)):
            h1 = tf.layers.dense(
                x,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")
            h2 = tf.layers.dense(
                h1,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc2")
            x_1 = tf.layers.dense(
                h2,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc3")
            # (-1, 256)

        return x_1, x_1

    # (ここは各階層で異なる)
    def process_phi_x(self, x, layer_index):
        """ 入力xをzとconcatしてhにリカレントする為にxの次元を変える為の層 """
        if layer_index < 2:
            return self.process_phi_x_conv(x, layer_index)
        else:
            return self.process_phi_x_fc(x, layer_index)

    # (ここは各階層同じ)
    def process_encoder(self, x_1, h, layer_index):
        # x_1とhをconcatし、Encode
        with tf.variable_scope("encoder" + str(layer_index)):
            # x_1とhをconcat
            enc_input = tf.concat(axis=1, values=(x_1, h))
            # FC層をいれて256に
            h1 = tf.layers.dense(
                enc_input,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")

            # muとsigmaに分岐
            enc_mu = tf.layers.dense(h1, self.n_z, name="mu")
            enc_log_sigma_sq = tf.layers.dense(
                h1, self.n_z, name="sigma")

        # 学習時はzを事後分布からサンプリング
        z = sample_gauss(enc_mu, enc_log_sigma_sq)
        return z, enc_mu, enc_log_sigma_sq

    # (ここは各階層同じ)
    def process_phi_z(self, z, layer_index):
        """ zをxとconcatしてhにリカレントする為にzの次元を変える為の層 """
        # zからFCでz_1に
        with tf.variable_scope("phi_z" + str(layer_index)):
            z_1 = tf.layers.dense(
                z,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")  # (-1, 256)
        return z_1

    def process_decoder_deconv(self, z_1, h, layer_index):
        # zとhで、Decode
        with tf.variable_scope("decoder" + str(layer_index)):
            # z_1とhをconcatした後のものに対して、deconvしていく
            dec_input = tf.concat(axis=1, values=(z_1, h))  # (-1, 512)

            if layer_index == 0:
                # FC層 (deconvのinputサイズに合わせるため)
                h0 = tf.layers.dense(
                    dec_input,
                    self.filter_size * 4 * 4,
                    activation=tf.nn.relu,
                    name="fc1")
                h0 = tf.reshape(h0, [-1, 4, 4, self.filter_size])
                h1 = tf.layers.conv2d_transpose(
                    h0,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv1")  # (-1, 8, 8, 16)
                h2 = tf.layers.conv2d_transpose(
                    h1,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv2")  # (-1, 16, 16, 16)
                h3 = tf.layers.conv2d_transpose(
                    h2,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv3")  # (-1, 32, 32, 16)
                h4 = tf.layers.conv2d_transpose(
                    h3,
                    filters=1,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=None,  # Activationは最後通していない
                    name="deconv4")  # (-1, 64, 64, 1)
                # flat化しておく
                dec_out_raw = tf.reshape(h4, [-1, 64 * 64 * 1])
                return dec_out_raw
            elif layer_index == 1:
                # FC層 (deconvのinputサイズに合わせるため)
                h0 = tf.layers.dense(
                    dec_input,
                    self.filter_size * 4 * 4,
                    activation=tf.nn.relu,
                    name="fc1")
                h0 = tf.reshape(h0, [-1, 4, 4, self.filter_size])
                h1 = tf.layers.conv2d_transpose(
                    h0,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv1")
                # (-1, 8, 8, filter_size)
                h2 = tf.layers.conv2d_transpose(
                    h1,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,  # Activation通している
                    name="deconv2")
                # (-1, 16, 16, filter_size)
                # flat化しておく
                h2_shape = h2.get_shape()
                dec_out_raw = tf.reshape(
                    h2, [-1, h2_shape[1] * h2_shape[2] * h2_shape[3]])
                return dec_out_raw

    def process_decoder_fc(self, z_1, h, output_dim, layer_index):
        # zとhで、Decode
        with tf.variable_scope("decoder" + str(layer_index)):
            # z_1とhをconcatした後のものに対して、fcする
            dec_input = tf.concat(axis=1, values=(z_1, h))  # (-1, 512)

            h1 = tf.layers.dense(
                dec_input,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")
            h2 = tf.layers.dense(
                h1,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc2")
            # 下から上がってきたx_upと同じ次元に
            dec_out_raw = tf.layers.dense(
                h2,
                output_dim,
                name="fc3",
                activation=tf.nn.relu)  # Activation入れている
        return dec_out_raw

    # (ここは各階層異なる)
    def process_decoder(self, z_1, h, output_dim, layer_index):
        # zとhで、Decode
        if layer_index < 2:
            return self.process_decoder_deconv(z_1, h, layer_index)
        else:
            return self.process_decoder_fc(z_1, h, output_dim, layer_index)

    def calc_upward_x(self, param):
        # 上行フローはバックプロパゲーションしない様に
        x_up = tf.stop_gradient(param)
        return x_up

    def __call__(self, x, state, scope=None):
        new_state = [None] * self.layer_size

        inputs = []
        enc_mus = []
        enc_log_sigma_sqs = []
        dec_out_raws = []
        prior_mus = []
        prior_log_sigma_sqs = []
        zs = []
        hs = []

        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            for layer_index in range(self.layer_size):
                with tf.variable_scope("layer" + str(layer_index)):
                    # 同階層の前時刻のrnn hidden
                    h0 = state[layer_index]
                    # h0=(-1, 256)

                    if layer_index != self.layer_size - 1:
                        # 最上位階層以外
                        # 一つ上の階層のrnn hidden
                        h1 = state[layer_index + 1]

                        if self.no_td_bp:
                            # Top-Down信号に対するBack propagationをしないようにする場合
                            h1 = tf.stop_gradient(h1)

                        if self.downward_type == "to_prior":
                            # 上から来たhと合流させない場合
                            h = h0
                        else:
                            # 上からきたhと合流させる場合
                            h = self.process_downward_h_join(
                                h0, h1, layer_index)

                    else:
                        # 最上位階層
                        h = h0
                        h1 = h0

                    # 上位階層のhiddenからZの事前分布を作成
                    if self.downward_type == "to_prior":
                        # 上の階層の情報から事前分布を求める場合
                        prior_mu, prior_log_sigma_sq = self.process_prior(
                            h1, layer_index)
                    else:
                        # 同じ階層のhから事前分布を求める場合
                        prior_mu, prior_log_sigma_sq = self.process_prior(
                            h, layer_index)

                    prior_mus.append(prior_mu)
                    prior_log_sigma_sqs.append(prior_log_sigma_sq)

                    if not self.for_generating:
                        # 学習時

                        if layer_index == 0:
                            # 最下位階層は入力がx
                            x_in = x
                        else:
                            # ひとつ下の層から上がってきたものをxに
                            x_in = x_up

                        inputs.append(x_in)
                        # decoder用にinputの次元を記録しておく
                        output_dim = x_in.get_shape()[1]

                        x_1, x_1_raw = self.process_phi_x(x_in, layer_index)
                        # (-1, 256), (-1, ***)

                        # 学習時はEncoderでZの事後分布を作成
                        z, enc_mu, enc_log_sigma_sq = self.process_encoder(
                            x_1, h, layer_index)
                        enc_mus.append(enc_mu)
                        enc_log_sigma_sqs.append(enc_log_sigma_sq)

                        # 上にあげる情報
                        x_up = self.calc_upward_x(x_1_raw)

                    else:
                        # 生成時は事前分布からZをサンプリング
                        z = sample_gauss(prior_mu, prior_log_sigma_sq)
                        if layer_index == 0:
                            output_dim = 64 * 64 * 1
                        elif layer_index == 1:
                            output_dim = 16 * 16 * self.filter_size
                        elif layer_index == 2:
                            output_dim = 4 * 4 * self.filter_size
                        else:
                            output_dim = self.n_h  # hiddenのサイズと同じにしている

                    zs.append(z)

                    z_1 = self.process_phi_z(z, layer_index)

                    # Decoder処理
                    dec_out_raw = self.process_decoder(z_1, h, output_dim,
                                                       layer_index)
                    # (-1, 64*64*1) or

                    dec_out_raws.append(dec_out_raw)

                    if self.for_generating:
                        # 生成時は出力したxからリカレント入力用のxを作成する
                        if layer_index == 0:
                            dec_out = tf.nn.sigmoid(dec_out_raw)
                            if self.binalize_output:
                                # ベルヌーイ分布のサンプリングを行う (only first layer)
                                x_out = sample_bernoulli(dec_out)
                            else:
                                # そのまま出力する
                                x_out = dec_out
                        else:
                            # sigmoid取らずにそのまま
                            x_out = dec_out_raw

                        # (-1, 64*64*1)
                        x_1, _ = self.process_phi_x(x_out, layer_index)

                        if layer_index == 0:
                            # 最下層のみ最終的に出力する
                            x_out_0 = x_out

                    # ここがリカレント部分 (RNNを利用している). リカレントのoutputは利用していない.
                    rec_input = tf.concat(axis=1, values=(x_1, z_1))
                    # rec_input=(10, 512), h=(10, 512)
                    _, state2 = self.cells[layer_index](rec_input, h)

                    new_state[layer_index] = state2
                    hs.append(state2)

        if self.for_generating:
            return x_out_0, new_state
        else:
            return (inputs, enc_mus, enc_log_sigma_sqs, dec_out_raws,
                    prior_mus, prior_log_sigma_sqs, zs, hs), new_state


class MerlinRNNCell(tf.contrib.rnn.RNNCell):
    """Merlin style RNN cell."""

    def __init__(self, layer_size, h_dim, z_dim, downward_type, no_td_bp,
                 filter_size, for_generating, binalize_output, reuse):

        self.layer_size = layer_size
        self.n_h = h_dim  # hiddenの次元
        self.n_z = z_dim  # zの次元

        self.downward_type = downward_type
        self.no_td_bp = no_td_bp  # Top-Down信号に対するBPをしないかどうか
        self.filter_size = filter_size
        self.for_generating = for_generating
        self.binalize_output = binalize_output
        self.reuse = reuse

        self.cells = [
            tf.contrib.rnn.BasicRNNCell(self.n_h)
            for _ in range(self.layer_size)
        ]

    @property
    def state_size(self):
        # zero_state()から呼ばれる
        return [self.n_h] * self.layer_size

    @property
    def output_size(self):
        # static_rnnを自前で実装した場合最終的には使われない
        return self.n_h

    def process_downward_h_join(self, h0, h1, layer_index):
        # 下方向に来た流れをどう合流するか
        # 上の階層のhとconcatする
        with tf.variable_scope("downward_h_join" + str(layer_index)):
            if self.downward_type == "concat":
                # Concatして、fcでサイズを揃える場合
                h_c = tf.concat(axis=1, values=(h0, h1))
                h = tf.layers.dense(
                    h_c,
                    self.n_h,
                    activation=tf.nn.relu,
                    name="fc1")
            elif self.downward_type == "gated_add":
                # gate addする場合
                gate = tf.get_variable(
                    "gate",
                    shape=h1.get_shape()[1:],
                    initializer=tf.constant_initializer(0.1))
                h = h0 + tf.multiply(gate, h1)
            elif self.downward_type == "add":
                # 加算する場合
                h = h0 + h1
        return h

    # (ここは各階層同じ)
    def process_prior(self, h, layer_index):
        # hから事前分布を計算する
        with tf.variable_scope("prior" + str(layer_index)):
            prior_hidden = tf.layers.dense(
                h,
                self.n_h,  # hiddenのサイズと同じにしている
                activation=tf.nn.relu,
                name="fc1")
            prior_mu = tf.layers.dense(
                prior_hidden, self.n_z, name="mu")
            prior_log_sigma_sq = tf.layers.dense(
                prior_hidden, self.n_z, name="sigma")
        return prior_mu, prior_log_sigma_sq

    def process_encoder_conv(self, x, layer_index):
        """ 入力xをzとconcatしてhにリカレントする為にxの次元を変える為の層 """

        # 入力xをx_1に
        with tf.variable_scope("encoder" + str(layer_index)):
            # flatになっているので直す
            if layer_index == 0:
                x = tf.reshape(x, [-1, 64, 64, 1])
                # (-1, 64, 64, 1)
            else:
                x = tf.reshape(x, [-1, 16, 16, self.filter_size])
                # (-1, 16, 16, filter_size)

            h1 = tf.layers.conv2d(
                x,
                filters=self.filter_size,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv1")
            # (-1, 32, 32, filter_size) or (-1, 8, 8, filter_size)
            h2 = tf.layers.conv2d(
                h1,
                filters=self.filter_size,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv2")
            # (-1, 16, 16, filter_size) or (-1, 4, 4, filter_size)
            # 上にあげる為のものをとっておく
            h2_flat = tf.layers.flatten(h2)

            x_1 = tf.layers.dense(
                h2_flat,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")

        return x_1, h2_flat

    def process_encoder_fc(self, x, layer_index):
        """ 入力xをzとconcatしてhにリカレントする為にxの次元を変える為の層 """

        # 入力xをx_1に
        with tf.variable_scope("encoder" + str(layer_index)):
            h1 = tf.layers.dense(
                x,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")
            h2 = tf.layers.dense(
                h1,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc2")
            x_1 = tf.layers.dense(
                h2,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc3")
            # (-1, 256)
        return x_1, x_1

    # (ここは各階層で異なる)
    def process_encoder(self, x, layer_index):
        """ 入力xをenocode """
        if layer_index < 2:
            return self.process_encoder_conv(x, layer_index)
        else:
            return self.process_encoder_fc(x, layer_index)

    def process_posterior(self, e, h, prior_mu, prior_log_sigma_sq,
                          layer_index):
        with tf.variable_scope("posterior" + str(layer_index)):
            # e, h, prior_mu, prior_log_sigma_sq を concat
            n = tf.concat(axis=1, values=(e, h, prior_mu, prior_log_sigma_sq))
            # FC層をいれて256に
            h1 = tf.layers.dense(
                n,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")

            # muとsigmaに分岐
            enc_mu0 = tf.layers.dense(
                h1, self.n_z, name="mu")
            enc_log_sigma_sq0 = tf.layers.dense(
                h1, self.n_z, name="sigma")

            enc_mu = enc_mu0 + prior_mu
            enc_log_sigma_sq = enc_log_sigma_sq0 + prior_log_sigma_sq

        # 学習時はzを事後分布からサンプリング
        z = sample_gauss(enc_mu, enc_log_sigma_sq)
        return z, enc_mu, enc_log_sigma_sq

    # (ここは各階層同じ)
    def process_phi_z(self, z, layer_index):
        """ zをxとconcatしてhにリカレントする為にzの次元を変える為の層 """
        # zからFCでz_1に
        with tf.variable_scope("phi_z" + str(layer_index)):
            z_1 = tf.layers.dense(
                z,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")  # (-1, 256)
        return z_1

    def process_decoder_deconv(self, z, layer_index):
        # zとhで、Decode
        with tf.variable_scope("decoder" + str(layer_index)):
            if layer_index == 0:
                # FC層 (deconvのinputサイズに合わせるため)
                h0 = tf.layers.dense(
                    z,
                    self.filter_size * 4 * 4,
                    activation=tf.nn.relu,
                    name="fc1")
                h0 = tf.reshape(h0, [-1, 4, 4, self.filter_size])
                h1 = tf.layers.conv2d_transpose(
                    h0,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv1")  # (-1, 8, 8, 16)
                h2 = tf.layers.conv2d_transpose(
                    h1,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv2")  # (-1, 16, 16, 16)
                h3 = tf.layers.conv2d_transpose(
                    h2,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv3")  # (-1, 32, 32, 16)
                h4 = tf.layers.conv2d_transpose(
                    h3,
                    filters=1,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=None,  # Activationは最後通していない
                    name="deconv4")  # (-1, 64, 64, 1)
                # flat化しておく
                dec_out_raw = tf.reshape(h4, [-1, 64 * 64 * 1])
                return dec_out_raw
            elif layer_index == 1:
                # FC層 (deconvのinputサイズに合わせるため)
                h0 = tf.layers.dense(
                    z,
                    self.filter_size * 4 * 4,
                    activation=tf.nn.relu,
                    name="fc1")
                h0 = tf.reshape(h0, [-1, 4, 4, self.filter_size])
                h1 = tf.layers.conv2d_transpose(
                    h0,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,
                    name="deconv1")
                # (-1, 8, 8, filter_size)
                h2 = tf.layers.conv2d_transpose(
                    h1,
                    filters=self.filter_size,
                    kernel_size=[4, 4],
                    strides=(2, 2),
                    padding="same",
                    activation=tf.nn.relu,  # Activation通している
                    name="deconv2")
                # (-1, 16, 16, filter_size)
                # flat化しておく
                h2_shape = h2.get_shape()
                dec_out_raw = tf.reshape(
                    h2, [-1, h2_shape[1] * h2_shape[2] * h2_shape[3]])
                return dec_out_raw

    def process_decoder_fc(self, z, output_dim, layer_index):
        # zとhで、Decode
        with tf.variable_scope("decoder" + str(layer_index)):
            h1 = tf.layers.dense(
                z,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc1")
            h2 = tf.layers.dense(
                h1,
                self.n_h,  # hiddenのサイズと同じにした
                activation=tf.nn.relu,
                name="fc2")
            # 下から上がってきたx_upと同じ次元に
            dec_out_raw = tf.layers.dense(
                h2,
                output_dim,
                name="fc3",
                activation=tf.nn.relu)  # Activation入れている
        return dec_out_raw

    # (ここは各階層異なる)
    def process_decoder(self, z, output_dim, layer_index):
        # zとhで、Decode
        if layer_index < 2:
            return self.process_decoder_deconv(z, layer_index)
        else:
            return self.process_decoder_fc(z, output_dim, layer_index)

    def calc_upward_x(self, param):
        # 上行フローはバックプロパゲーションしない様に
        x_up = tf.stop_gradient(param)
        return x_up

    def __call__(self, x, state, scope=None):
        new_state = [None] * self.layer_size

        inputs = []
        enc_mus = []
        enc_log_sigma_sqs = []
        dec_out_raws = []
        prior_mus = []
        prior_log_sigma_sqs = []
        zs = []
        hs = []

        with tf.variable_scope(scope or type(self).__name__, reuse=self.reuse):
            for layer_index in range(self.layer_size):
                with tf.variable_scope("layer" + str(layer_index)):
                    # 同階層の前時刻のrnn hidden
                    h0 = state[layer_index]
                    # h0=(-1, 256)

                    if layer_index != self.layer_size - 1:
                        # 最上位階層以外
                        # 一つ上の階層のrnn hidden
                        h1 = state[layer_index + 1]

                        if self.no_td_bp:
                            # Top-Down信号に対するBack propagationをしないようにする場合
                            h1 = tf.stop_gradient(h1)

                        if self.downward_type == "to_prior":
                            # 上から来たhと合流させない場合
                            h = h0
                        else:
                            # 上からきたhと合流させる場合
                            h = self.process_downward_h_join(
                                h0, h1, layer_index)

                    else:
                        # 最上位階層
                        h = h0
                        h1 = h0

                    # 上位階層のhiddenからZの事前分布を作成
                    if self.downward_type == "to_prior":
                        # 上の階層の情報から事前分布を求める場合
                        prior_mu, prior_log_sigma_sq = self.process_prior(
                            h1, layer_index)
                    else:
                        # 同じ階層のhから事前分布を求める場合
                        prior_mu, prior_log_sigma_sq = self.process_prior(
                            h, layer_index)

                    prior_mus.append(prior_mu)
                    prior_log_sigma_sqs.append(prior_log_sigma_sq)

                    if not self.for_generating:
                        # 学習時

                        if layer_index == 0:
                            # 最下位階層は入力がx
                            x_in = x
                        else:
                            # ひとつ下の層から上がってきたものをxに
                            x_in = x_up

                        inputs.append(x_in)
                        # decoder用にinputの次元を記録しておく
                        output_dim = x_in.get_shape()[1]

                        e, e_raw = self.process_encoder(x_in, layer_index)
                        # (-1, 256), (-1, ***)

                        # 学習時はEncoderでZの事後分布を作成
                        z, enc_mu, enc_log_sigma_sq = self.process_posterior(
                            e, h, prior_mu, prior_log_sigma_sq, layer_index)
                        enc_mus.append(enc_mu)
                        enc_log_sigma_sqs.append(enc_log_sigma_sq)

                        # 上にあげる情報
                        x_up = self.calc_upward_x(e_raw)

                    else:
                        # 生成時は事前分布からZをサンプリング
                        z = sample_gauss(prior_mu, prior_log_sigma_sq)
                        if layer_index == 0:
                            output_dim = 64 * 64 * 1
                        elif layer_index == 1:
                            output_dim = 16 * 16 * self.filter_size
                        elif layer_index == 2:
                            output_dim = 4 * 4 * self.filter_size
                        else:
                            output_dim = self.n_h  # hiddenのサイズと同じにしている

                    zs.append(z)

                    # Decoder処理
                    dec_out_raw = self.process_decoder(z, output_dim,
                                                       layer_index)
                    # (-1, 64*64*1) or

                    dec_out_raws.append(dec_out_raw)

                    if self.for_generating:
                        # 生成時は出力したxからリカレント入力用のxを作成する
                        if layer_index == 0:
                            dec_out = tf.nn.sigmoid(dec_out_raw)
                            if self.binalize_output:
                                # ベルヌーイ分布のサンプリングを行う (only first layer)
                                x_out = sample_bernoulli(dec_out)
                            else:
                                # そのまま出力する
                                x_out = dec_out
                        else:
                            # sigmoid取らずにそのまま
                            x_out = dec_out_raw

                        if layer_index == 0:
                            # 最下層のみ最終的に出力する
                            x_out_0 = x_out

                    # ここがリカレント部分 (RNNを利用している). リカレントのoutputは利用していない.
                    _, state2 = self.cells[layer_index](z, h)

                    new_state[layer_index] = state2
                    hs.append(state2)

        if self.for_generating:
            return x_out_0, new_state
        else:
            return (inputs, enc_mus, enc_log_sigma_sqs, dec_out_raws,
                    prior_mus, prior_log_sigma_sqs, zs, hs), new_state
