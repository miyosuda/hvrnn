# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from data_manager import DataManager
from model import VRNN
from trainer import Trainer


class TrainerTest(tf.test.TestCase):
    def create_trainer(self, layer_size, h_size, latent_size, seq_length):
        batch_size = 10
        beta = 1.0
        cell_type = "merlin"
        learning_rate = 1e-4
        downward_type = "to_prior"
        no_td_bp = True
        filter_size = 64

        use_denoising = False

        train_model = VRNN(
            layer_size,
            h_size,
            latent_size,
            batch_size,
            seq_length,
            beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=False,
            binalize_output=True,
            reuse=False)
        generate_model = VRNN(
            layer_size,
            h_size,
            latent_size,
            1,
            1,
            beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=True,
            binalize_output=True,
            reuse=True)
        predict_model = VRNN(
            layer_size,
            h_size,
            latent_size,
            1,
            1,
            beta,
            cell_type=cell_type,
            downward_type=downward_type,
            no_td_bp=no_td_bp,
            filter_size=filter_size,
            for_generating=False,
            binalize_output=True,
            reuse=True)

        data_manager = DataManager.get_data_manager(dataset_type="bsprite")
        seq_length = data_manager.seq_length
        trainer = Trainer(data_manager, train_model, generate_model,
                          predict_model, learning_rate, use_denoising)

        return trainer

    def test_calculate(self):
        layer_size = 2
        h_size = 256
        latent_size = 16
        seq_length = 20

        trainer = self.create_trainer(layer_size, h_size, latent_size,
                                      seq_length)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            data_index = 0
            off_forecast = False

            out = trainer.calculate(sess, data_index, off_forecast)

            errors, inputs, enc_mus, enc_sigma_sqs, dec_outs, prior_mus, prior_sigma_sqs = out

            # 各要素はlayer_size分の配列になっている
            for i in range(layer_size):
                self.assertEqual(errors[i].shape, (seq_length, latent_size))
                self.assertEqual(enc_mus[i].shape, (seq_length, latent_size))
                self.assertEqual(enc_sigma_sqs[i].shape,
                                 (seq_length, latent_size))
                self.assertEqual(prior_mus[i].shape, (seq_length, latent_size))
                self.assertEqual(prior_sigma_sqs[i].shape,
                                 (seq_length, latent_size))

                # dec_out, input_だけ各階層でshapeが違う
                if i == 0:
                    self.assertEqual(dec_outs[i].shape, (seq_length, 64, 64))
                    self.assertEqual(inputs[i].shape, (seq_length, 64, 64))
                else:
                    self.assertEqual(dec_outs[i].shape,
                                     (seq_length, 64 * 16 * 16))
                    self.assertEqual(inputs[i].shape,
                                     (seq_length, 64 * 16 * 16))

    def test_forecast(self):
        layer_size = 3
        h_size = 256
        latent_size = 16
        seq_length = 20

        trainer = self.create_trainer(layer_size, h_size, latent_size,
                                      seq_length)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            orignal_images, forecasted_images = trainer.forecast(
                sess, data_index=0)

            self.assertEqual(orignal_images.shape, (seq_length, 64, 64))
            self.assertEqual(forecasted_images.shape,
                             (seq_length // 2, 64, 64))

    def test_collect_analysis_data(self):
        layer_size = 3
        h_size = 256
        latent_size = 16
        seq_length = 20

        trainer = self.create_trainer(layer_size, h_size, latent_size,
                                      seq_length)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            data_size = 200

            zses, hses = trainer.collect_analysis_data(sess, layer_size,
                                                       data_size)

            self.assertEqual(zses.shape,
                             (layer_size, data_size, seq_length, latent_size))
            self.assertEqual(hses.shape,
                             (layer_size, data_size, seq_length, h_size))


if __name__ == "__main__":
    tf.test.main()
