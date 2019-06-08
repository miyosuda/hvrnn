# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import utils


class Trainer(object):
    def __init__(self, data_manager, train_model, generate_model,
                 predict_model, learning_rate, use_denoising):

        self.data_manager = data_manager
        self.train_model = train_model
        self.generate_model = generate_model
        self.predict_model = predict_model
        self.use_denoising = use_denoising

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.train_model.loss)

        # Tensorboard記録用
        self.generated_image_ph = tf.placeholder(
            tf.float32, shape=[1, 64 * 3 + 2, 64 * 10 + 9, 1])
        self.generated_image_op = tf.summary.image("generated_image",
                                                   self.generated_image_ph, 1)

    def train(self, sess, summary_writer, batch_size, step):
        seq_length = self.data_manager.seq_length
        w = self.data_manager.w
        h = self.data_manager.h
        batch_data = self.data_manager.get_next_train_batch(batch_size)
        batch_data = np.reshape(batch_data, [batch_size, seq_length, w * h])

        if self.use_denoising:
            # ノイズを載せたもの
            input_batch_data = utils.add_noise(batch_data)
        else:
            input_batch_data = batch_data

        # 毎バッチごとにstateの初期化をするので、stateの更新をしていない
        out = sess.run(
            [
                self.train_op, self.train_model.loss,
                self.train_model.reconstr_loss, self.train_model.latent_loss,
                self.train_model.last_state_h,
                self.train_model.summary_op_frequent
            ],
            feed_dict={
                self.train_model.input_data: input_batch_data,
                self.train_model.target_data: batch_data
            })
        _, loss, reconstr_loss, latent_loss, _, summary_str = out

        if step % 10 == 0:
            summary_writer.add_summary(summary_str, step)
        """
        # weightの表示はしないことに
        if step % 10000 == 0:
          # weightのヒストグラム更新
          summary_str_infrequent = sess.run( self.train_model.summary_op_infrequent )
          summary_writer.add_summary(summary_str_infrequent, step)
        """

    def record_loss(self, summary_writer, tag, value, step):
        summary_str = tf.Summary(
            value=[tf.Summary.Value(tag=tag, simple_value=value)])
        summary_writer.add_summary(summary_str, step)

    def test(self, sess, summary_writer, batch_size, step):
        test_data_size = self.data_manager.using_test_data_size

        seq_length = self.data_manager.seq_length
        w = self.data_manager.w
        h = self.data_manager.h

        all_reconstr_losses = []
        all_latent_losses = []

        for i in range(0, test_data_size, batch_size):
            batch_data = self.data_manager.get_test_batch(i, batch_size)
            batch_data = np.reshape(batch_data,
                                    [batch_size, seq_length, w * h])

            if self.use_denoising:
                # ノイズを載せたもの
                input_batch_data = utils.add_noise(batch_data)
            else:
                input_batch_data = batch_data

            # 毎バッチごとにstateの初期化をするので、stateの更新をしていない
            out = sess.run(
                [
                    self.train_model.reconstr_losses,
                    self.train_model.latent_losses,
                    self.train_model.last_state_h
                ],
                feed_dict={
                    self.train_model.input_data: input_batch_data,
                    self.train_model.target_data: batch_data
                })
            reconstr_losses, latent_losses, _ = out
            all_reconstr_losses.append(reconstr_losses)
            all_latent_losses.append(latent_losses)

        mean_reconstr_losses = np.mean(all_reconstr_losses, axis=0)
        mean_latent_losses = np.mean(all_latent_losses, axis=0)

        # Summary記録
        for i in range(len(mean_reconstr_losses)):
            self.record_loss(summary_writer, "test_rconstr_loss{}".format(i),
                             mean_reconstr_losses[i], step)
            self.record_loss(summary_writer, "test_latent_loss{}".format(i),
                             mean_latent_losses[i], step)

            print(
                "layer:{0}, reconstr_loss={1:.2f}, latent_loss={2:.4f}".format(
                    i, mean_reconstr_losses[i], mean_latent_losses[i]))

    def generate(self, sess, generate_size, summary_writer=None, step=None):
        # Stateをリセットする
        last_state_h = sess.run(self.generate_model.initial_state_h)

        samples = []

        for i in range(generate_size):
            feed_dict = {}
            for j, last_state_h_j in enumerate(last_state_h):
                feed_dict[self.generate_model.initial_state_h[
                    j]] = last_state_h[j]

            out = sess.run(
                [
                    self.generate_model.sampled_x,
                    self.generate_model.last_state_h
                ],
                feed_dict=feed_dict)
            x, last_state_h = out
            x = np.reshape(x, [64, 64])
            samples.append(x)

        if summary_writer is not None:
            # Tensorboardへの記録
            concated_samples = utils.concat_images_in_rows(samples, 3)
            # (194, 649)
            concated_samples = np.reshape(concated_samples,
                                          [1, 64 * 3 + 2, 64 * 10 + 9, 1])
            # (1, 194, 649, 1)
            summary_str = sess.run(
                self.generated_image_op,
                feed_dict={self.generated_image_ph: concated_samples})
            summary_writer.add_summary(summary_str, step)
        return samples

    def calculate_sub(self, sess, images):
        """ 内部状態ビジュアライズ用に内部状態を計算する """

        # (1, 20, 64, 64)
        seq_length = images.shape[1]

        errors = []
        inputs = []
        enc_mus = []
        enc_sigma_sqs = []
        dec_outs = []
        dec_sigma_sqs = []
        prior_mus = []
        prior_sigma_sqs = []

        # Stateをリセットする

        last_state_h = sess.run(self.predict_model.initial_state_h)

        for i in range(seq_length):
            x = images[0, i, :, :]
            x = np.reshape(x, [1, 1, 64 * 64 * 1])

            # feed_dictの準備
            feed_dict = {}
            feed_dict[self.predict_model.input_data] = x
            for j, last_state_h_j in enumerate(last_state_h):
                feed_dict[self.predict_model.initial_state_h[
                    j]] = last_state_h[j]

            out = sess.run(
                [
                    self.predict_model.last_state_h,
                    self.predict_model.prediction_errors,
                    self.predict_model.inputs, self.predict_model.enc_mus,
                    self.predict_model.enc_sigma_sqs,
                    self.predict_model.dec_outs, self.predict_model.prior_mus,
                    self.predict_model.prior_sigma_sqs
                ],
                feed_dict=feed_dict)
            last_state_h, prediction_error, \
              input_, enc_mu, enc_sigma_sq, dec_out, prior_mu, prior_sigma_sq = out

            # prediction_error [(1, 16), (1, 16)]
            # enc_mu           [(1, 16), (1, 16)]
            # enc_sigma_sq     [(1, 16), (1, 16)]
            # dec_out          [(1, 4096), (1, 1024)]
            # prior_mu         [(1, 16), (1,16)]
            # prior_sigma_sq   [(1, 16), (1, 16)]

            def squeeze_vals(vals):
                # (1,16)とかを(16)にsqueeeする
                return [np.squeeze(val) for val in vals]

            # それぞれ現在全階層分の配列になっている
            input_ = squeeze_vals(input_)
            prediction_error = squeeze_vals(prediction_error)
            enc_mu = squeeze_vals(enc_mu)
            enc_sigma_sq = squeeze_vals(enc_sigma_sq)
            dec_out = squeeze_vals(dec_out)
            prior_mu = squeeze_vals(prior_mu)
            prior_sigma_sq = squeeze_vals(prior_sigma_sq)

            errors.append(prediction_error)
            inputs.append(input_)
            enc_mus.append(enc_mu)
            enc_sigma_sqs.append(enc_sigma_sq)
            dec_outs.append(dec_out)
            prior_mus.append(prior_mu)
            prior_sigma_sqs.append(prior_sigma_sq)

        def seperate(vals_seqs):
            # [seq, 16]みたいなndarrayがlayer_size分ならんだ配列にする
            layer_size = len(vals_seqs[0])
            arr = [[] for _ in range(layer_size)]
            for vals in vals_seqs:
                for i, val in enumerate(vals):
                    arr[i].append(val)
            # 各要素をndarrayにしておく
            return [np.array(a) for a in arr]

        def seperate_dec_out(vals_seqs):
            # [seq, 16]みたいなndarrayがlayer_size分ならんだ配列にするが、
            # 第一階層のみ、[seq, 64, 64]にする
            layer_size = len(vals_seqs[0])
            arr = [[] for _ in range(layer_size)]
            for vals in vals_seqs:
                for i, val in enumerate(vals):
                    arr[i].append(val)
            ret = []
            for i in range(layer_size):
                if i == 0:
                    a = np.reshape(np.array(arr[i]), [seq_length, 64, 64])
                else:
                    a = np.array(arr[i])
                ret.append(a)
            return ret

        return \
          seperate(errors), \
          seperate_dec_out(inputs), \
          seperate(enc_mus), \
          seperate(enc_sigma_sqs), \
          seperate_dec_out(dec_outs), \
          seperate(prior_mus), \
          seperate(prior_sigma_sqs)

    def calculate(self, sess, data_index, off_forecast):
        """ 内部状態ビジュアライズ用に内部状態を計算する """

        if off_forecast:
            # 予測がはずれる様なデータの場合
            images = self.data_manager.get_off_forecast_test_data(
                data_index=data_index)
        else:
            # 予測がはずれないデータの場合
            images = self.data_manager.get_test_data(data_index=data_index)
        #images = utils.add_noise(images) # ノイズを入れる
        return self.calculate_sub(sess, images)

    def calculate_for_analyze(self, sess, data_index):
        """ 内部状態ビジュアライズ用に内部状態を計算する """
        images, _, _, _ = self.data_manager.get_check_plain_data(
            data_index=data_index)
        return self.calculate_sub(sess, images)

    def forecast(self, sess, data_index):
        images = self.data_manager.get_test_data(data_index=data_index)

        last_state_h = sess.run(self.predict_model.initial_state_h)

        samples = []

        seq_length = self.data_manager.seq_length

        for i in range(seq_length // 2):
            x = images[0, i, :, :]
            x = np.reshape(x, [1, 1, 64 * 64 * 1])

            # feed_dictの準備
            feed_dict = {}
            feed_dict[self.predict_model.input_data] = x
            for j, last_state_h_j in enumerate(last_state_h):
                feed_dict[self.predict_model.initial_state_h[
                    j]] = last_state_h[j]

            out = sess.run(
                [self.predict_model.last_state_h, self.predict_model.dec_outs],
                feed_dict=feed_dict)

            last_state_h, dec_out = out

            np.reshape(x, [64, 64])

        for i in range(seq_length // 2):
            feed_dict = {}
            for j, last_state_h_j in enumerate(last_state_h):
                feed_dict[self.generate_model.initial_state_h[
                    j]] = last_state_h[j]

            out = sess.run(
                [
                    self.generate_model.sampled_x,
                    self.generate_model.last_state_h
                ],
                feed_dict=feed_dict)
            x, last_state_h = out
            x = np.reshape(x, [64, 64])
            samples.append(x)

        orignal_images = np.reshape(images,
                                    [seq_length, 64, 64])  # (20, 64, 64)
        forecasted_images = np.array(samples)  # (10, 64, 64)

        return orignal_images, forecasted_images

    def collect_analysis_data(self, sess, layer_size, data_size):
        batch_size = 10

        seq_length = self.data_manager.seq_length
        w = self.data_manager.w
        h = self.data_manager.h

        all_zses = [[] for _ in range(layer_size)]
        all_hses = [[] for _ in range(layer_size)]

        for i in range(0, data_size, batch_size):
            # テストデータセットからデータを取ってくる
            batch_data = self.data_manager.get_test_batch(
                i, batch_size)
            # batch_data = (batch_size, seq_length, 64, 64)
            batch_data = np.reshape(batch_data,
                                    [batch_size, seq_length, w * h])

            # 毎バッチごとにstateの初期化をするので、stateの更新をしていない
            out = sess.run(
                [
                    self.train_model.zs, self.train_model.hs,
                    self.train_model.last_state_h
                ],
                feed_dict={self.train_model.input_data: batch_data})
            zs, hs, _ = out
            # (layer_size, 200, 16)

            for j in range(layer_size):
                # 3階層をそれぞれ別の器に入れていく
                all_zses[j].append(zs[j])
                # (200, 16) を各階層枚にappendしていく
                all_hses[j].append(hs[j])

        all_zses = np.array(all_zses)
        all_hses = np.array(all_hses)
        #(layer_size, 5000/10, seq_length*batch_size, 16)
        latent_size = all_zses.shape[3]
        h_size = all_hses.shape[3]

        all_zses = all_zses.reshape([layer_size, -1, seq_length, latent_size])
        all_hses = all_hses.reshape([layer_size, -1, seq_length, h_size])

        return all_zses, all_hses
