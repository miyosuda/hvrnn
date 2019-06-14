# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from cell import VariationalRNNCell, MerlinRNNCell


def static_rnn(cell, inputs, initial_state):

    outputs = []

    with tf.variable_scope("rnn") as varscope:
        state = initial_state

        for time, input_ in enumerate(inputs):
            if time > 0:
                varscope.reuse_variables()
            (output, state) = cell(input_, state)
            outputs.append(output)

        return (outputs, state)


class VRNN():
    def __init__(self,
                 layer_size,
                 h_size,
                 latent_size,
                 batch_size,
                 seq_length,
                 beta,
                 cell_type,
                 downward_type,
                 no_td_bp,
                 filter_size,
                 for_generating,
                 binalize_output,
                 reuse):

        self.cell_type = cell_type

        def _extract_and_stack(output_seqs, param_index, layer_index,
                               param_name):
            with tf.variable_scope(param_name + str(layer_index), reuse=reuse):
                # extract values like enc_mu and stack them along the sequence.
                # enc_muなどを取り出して、sequence方向にstackする.
                x = tf.stack([
                    output[param_index][layer_index] for output in output_seqs
                ])
                # (seq_length, batch_size, z_dim) or (seq_length, batch_size, x_dim)
                x = tf.transpose(x, [1, 0, 2])
                # (batch_size, seq_length, z_dim) or (batch_size, seq_length, x_dim)
                x = tf.reshape(x, [batch_size * seq_length, -1])
                # (batch_size * seq_length, z_dim) or (batch_size * seq_length, x_dim)
                # (The last -1 is treated as z_dim or x_dim)
                return x

        def _calc_reconstr_losses(ys, first_y, dec_out_raws):
            # Negative log likelihood
            with tf.variable_scope("reconstr", reuse=reuse):
                ret = []
                for i in range(len(ys)):
                    if i == 0:
                        y = first_y
                    else:
                        y = ys[i]
                    dec_out_raw = dec_out_raws[i]
                    # y, dec_out_raw  (batch_size*seq_length, 64*64*1)

                    if i == 0:
                        # Cross entropy loss for the 1st layer.
                        result = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=y, logits=dec_out_raw)
                        result = tf.reduce_sum(result, 1)
                        # reduce_sumの中身は(batch_size*seq_length, 64*64*1)
                        result = tf.reduce_mean(result)
                    else:
                        # L2 loss
                        result = tf.nn.l2_loss(y - dec_out_raw) / (
                            batch_size * seq_length)
                    ret.append(result)
                return ret

        def _calc_latent_losses(mu1s, log_sigma1_sqs, mu2s, log_sigma2_sqs,
                                beta):
            with tf.variable_scope("latent", reuse=reuse):
                ret = []
                for i in range(len(mu1s)):
                    mu1 = mu1s[i]
                    log_sigma1_sq = log_sigma1_sqs[i]
                    mu2 = mu2s[i]
                    log_sigma2_sq = log_sigma2_sqs[i]

                    result = tf.reduce_sum(
                        0.5 *
                        (log_sigma2_sq - log_sigma1_sq +
                         tf.exp(log_sigma1_sq - log_sigma2_sq) +
                         tf.square(mu1 - mu2) / tf.exp(log_sigma2_sq) - 1), 1)
                    result = tf.reduce_mean(result) * beta
                    ret.append(result)
                return ret

        def _calc_prediction_errors(mu1s, log_sigma1_sqs, mu2s,
                                    log_sigma2_sqs):
            results = []

            with tf.variable_scope("prediction_error", reuse=reuse):
                for i in range(len(mu1s)):
                    mu1 = mu1s[i]
                    log_sigma1_sq = log_sigma1_sqs[i]
                    mu2 = mu2s[i]
                    log_sigma2_sq = log_sigma2_sqs[i]

                    result = 0.5 * (
                        log_sigma2_sq - log_sigma1_sq +
                        tf.exp(log_sigma1_sq - log_sigma2_sq) +
                        tf.square(mu1 - mu2) / tf.exp(log_sigma2_sq) - 1)
                    results.append(result)
                # 'resuts' is the list of prediction errors for each hierarchcal layer.
                return results

        if cell_type == "vrnn":
            cell = VariationalRNNCell(
                layer_size,
                h_size,
                latent_size,
                downward_type=downward_type,
                no_td_bp=no_td_bp,
                filter_size=filter_size,
                for_generating=for_generating,
                binalize_output=binalize_output,
                reuse=reuse)
        elif cell_type == "merlin":
            cell = MerlinRNNCell(
                layer_size,
                h_size,
                latent_size,
                downward_type=downward_type,
                no_td_bp=no_td_bp,
                filter_size=filter_size,
                for_generating=for_generating,
                binalize_output=binalize_output,
                reuse=reuse)

        self.cell = cell

        # Place holder for the input
        self.input_data = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, seq_length, 64 * 64 * 1],
            name="input_data")
        self.target_data = tf.placeholder(
            dtype=tf.float32,
            shape=[batch_size, seq_length, 64 * 64 * 1],
            name="input_data")

        # Zero initialized initial state
        self.initial_state_h = cell.zero_state(
            batch_size=batch_size, dtype=tf.float32)

        with tf.variable_scope("inputs", reuse=reuse):
            # Swap sequence size and batch size.
            inputs = tf.transpose(self.input_data, [1, 0, 2])
            # (seq_length, batch_size, 64*64*1)
            inputs = tf.reshape(inputs, [-1, 64 * 64 * 1])
            # (seq_length * batch_size, 64*64*1)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            # seq_length * (batch_size, n_hidden)
            inputs = tf.split(
                axis=0, num_or_size_splits=seq_length, value=inputs)
            # 'inputs' is the list of tensors [seq_length], and each element is (batch, 64*64*1)

        with tf.variable_scope("target", reuse=reuse):
            target = tf.reshape(self.target_data, [-1, 64 * 64 * 1])
            # (batch_size * seq_length, 64*64*1)

        output_seqs, last_state = static_rnn(
            cell, inputs, initial_state=self.initial_state_h)
        self.last_state_h = last_state

        if not for_generating:
            # 'output_seqs' is a list of [seq_length].
            # Each element of the list is a tuple of ([enc_mu], [enc_sigma], ....).
            # Values like 'enc_mu' are 、[layer_size].
            # Each element of them has the shape of (batch_size, z_dim).
            # 'last_state' is [(batch_size, dim_h), ...].

            outputs_reshape = []
            names = [
                "input", "enc_mu", "enc_log_sigma_sq", "dec_out_raw",
                "prior_mu", "prior_log_sigma_sq", "z", "h"
            ]

            # 'output_seqs' is a list with sequence size.
            for param_index, param_name in enumerate(names):
                xs = []
                for layer_index in range(layer_size):
                    x = _extract_and_stack(output_seqs, param_index,
                                           layer_index, param_name)
                    xs.append(x)
                outputs_reshape.append(xs)

            inputs, enc_mus, enc_log_sigma_sqs, dec_out_raws, prior_mus, prior_log_sigma_sqs, zs, hs \
              = outputs_reshape

            # Each of them is [(batch_size * seq_length, z_dim or x_dim), ... x layer size]

            self.inputs = inputs
            self.enc_mus = enc_mus
            self.enc_sigma_sqs = [
                tf.exp(enc_log_sigma_sq)
                for enc_log_sigma_sq in enc_log_sigma_sqs
            ]
            self.prior_mus = prior_mus
            self.prior_sigma_sqs = [
                tf.exp(prior_log_sigma_sq)
                for prior_log_sigma_sq in prior_log_sigma_sqs
            ]
            self.zs = zs
            self.hs = hs

            # Decoder output
            self.dec_outs = []
            for i in range(layer_size):
                if i == 0:
                    dec_out = tf.nn.sigmoid(dec_out_raws[i])
                else:
                    dec_out = dec_out_raws[i]
                self.dec_outs.append(dec_out)

            # Loss calculation
            self.reconstr_losses = _calc_reconstr_losses(
                inputs, target, dec_out_raws)
            self.latent_losses = _calc_latent_losses(
                enc_mus, enc_log_sigma_sqs, prior_mus, prior_log_sigma_sqs,
                beta)

            self.reconstr_loss = tf.reduce_sum(tf.stack(self.reconstr_losses))
            self.latent_loss = tf.reduce_sum(tf.stack(self.latent_losses))
            self.loss = self.reconstr_loss + self.latent_loss

            self.prediction_errors = _calc_prediction_errors(
                enc_mus, enc_log_sigma_sqs, prior_mus, prior_log_sigma_sqs)

            # For Tensorboard
            # Loss visualization
            summary_ops = []
            for i in range(layer_size):
                reconstr_loss_summary_op = tf.summary.scalar(
                    "reconstr_loss{}".format(i), self.reconstr_losses[i])
                latent_loss_summary_op = tf.summary.scalar(
                    "latent_loss{}".format(i), self.latent_losses[i])
                summary_ops.append(reconstr_loss_summary_op)
                summary_ops.append(latent_loss_summary_op)
            total_reconstr_loss_summary_op = tf.summary.scalar(
                "total_reconstr_loss", self.reconstr_loss)
            total_latent_loss_summary_op = tf.summary.scalar(
                "total_latent_loss", self.latent_loss)
            summary_ops.append(total_reconstr_loss_summary_op)
            summary_ops.append(total_latent_loss_summary_op)

            self.summary_op_frequent = tf.summary.merge(summary_ops)

            # Weight visualization
            if cell_type == "vrnn":
                scope_name = "rnn/VariationalRNNCell"
            elif cell_type == "merlin":
                scope_name = "rnn/MerlinRNNCell"

            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
            histogram_ops = []
            for v in variables:
                histogram_op = tf.summary.histogram(v.name, v)
                histogram_ops.append(histogram_op)
            self.summary_op_infrequent = tf.summary.merge(histogram_ops)

        else:
            # For generation
            self.sampled_x = output_seqs[0]

    def get_conv_weight(self):
        # Get 1st conv layers weights for weight visualization
        if self.cell_type == "vrnn":
            with tf.variable_scope("rnn/VariationalRNNCell", reuse=True):
                weight = tf.get_variable("layer0/phi_x0/conv1/kernel")
        elif self.cell_type == "merlin":
            with tf.variable_scope("rnn/MerlinRNNCell", reuse=True):
                weight = tf.get_variable("layer0/encoder0/conv1/kernel")
        return weight
