# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os

from data_manager import DataManager
from model import VRNN
import options

oculomotor_data_path = "../data/oculomotor/oculomotor.npz"  # Oculomotor dataset

flags = options.get_options()


def sample_gauss(mu, log_sigma_sq):
    eps_shape = tf.shape(mu)
    eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32)
    # z = mu + sigma * epsilon
    ret = tf.add(mu, tf.multiply(tf.sqrt(tf.exp(log_sigma_sq)), eps))
    return ret


class DecodeModel():
    def __init__(self, latent_size, h_size, learning_rate, use_h, scope):
        with tf.variable_scope(scope):
            self.build(latent_size, h_size, use_h)
            self.create_loss_optimizer(learning_rate)

    def build(self, latent_size, h_size, use_h):
        self.input_mu = tf.placeholder(dtype=tf.float32,
                                       shape=[None, latent_size],
                                       name="input_mu")
        self.input_sigma_sq = tf.placeholder(dtype=tf.float32,
                                             shape=[None, latent_size],
                                             name="input_sigma_sq")
        self.input_h = tf.placeholder(dtype=tf.float32,
                                      shape=[None, h_size],
                                      name="input_h")
        self.target_action = tf.placeholder(dtype=tf.float32,
                                            shape=[None, 2],
                                            name="target_action")
        
        z = self.sample_gauss(self.input_mu, self.input_sigma_sq)

        if use_h:
            h0 = tf.concat(axis=1, values=(z, self.input_h))
        else:
            h0 = z
        h1 = tf.layers.dense(h0,
                             256,
                             activation=tf.nn.relu,
                             name="fc1")
        h2 = tf.layers.dense(h1,
                             256,
                             activation=tf.nn.relu,
                             name="fc2")
        self.out_action = tf.layers.dense(h2,
                                          2,
                                          name="fc3")

    def sample_gauss(self, mu, sigma_sq):
        eps_shape = tf.shape(mu)
        eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32)
        # z = mu + sigma * epsilon
        ret = tf.add(mu, tf.multiply(tf.sqrt(sigma_sq), eps))
        return ret

    def create_loss_optimizer(self, learning_rate):
        self.loss = tf.nn.l2_loss(self.target_action - self.out_action)
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.loss_summary_op = tf.summary.scalar("loss", self.loss)



class DecodeDataManager():
    def __init__(self):
        base_data_path = flags.save_dir + "/decode_base_data.npz"
        base_data = np.load(base_data_path)
        
        data_mus       = base_data["mu"]       # (3, 10000, 20, 16) float32
        data_sigma_sqs = base_data["sigma_sq"] # (3, 10000, 20, 16) float32
        data_hs        = base_data["h"]        # (3, 10000, 20, 256) float32
        del base_data

        data = np.load(oculomotor_data_path)
        data_actions = data["actions"]         # (10000, 20, 2) float32
        del data

        layer_size  = data_mus.shape[0]
        latent_size = data_mus.shape[3]
        h_size      = data_hs.shape[3]
        action_size = data_actions.shape[2]

        # 開始フレームを指定
        start_frame = 5

        data_mus       = data_mus[:, :, start_frame:, :]
        data_sigma_sqs = data_sigma_sqs[:, :, start_frame:, :]
        data_hs        = data_hs[:, :, start_frame:, :]
        data_actions   = data_actions[:, start_frame:, :]

        # シーケンスをばらす
        data_mus       = data_mus.reshape([layer_size, -1, latent_size])
        data_sigma_sqs = data_sigma_sqs.reshape([layer_size, -1, latent_size])
        data_hs        = data_hs.reshape([layer_size, -1, h_size])
        data_actions   = data_actions.reshape([-1, action_size])

        data_size = len(data_actions)
        self.test_data_size = data_size // 10
        self.train_data_size = data_size - self.test_data_size

        # data_size=150000, train=135000, test=15000

        # TrainとTestに分ける
        self.train_mus, self.test_mus = self.split_layered_train_test_data(
            data_mus, self.train_data_size)
        self.train_sigma_sqs, self.test_sigma_sqs = self.split_layered_train_test_data(
            data_sigma_sqs, self.train_data_size)
        self.train_hs, self.test_hs = self.split_layered_train_test_data(
            data_hs, self.train_data_size)
        self.train_actions, self.test_actions = self.split_train_test_data(
            data_actions, self.train_data_size)

        self.prepare_train_indices()

    def prepare_train_indices(self):
        self.train_indices = list(range(self.train_data_size))
        random.shuffle(self.train_indices)
        self.train_pos = 0

    def split_train_test_data(self, data, train_data_size):
        # 引数のデータをtrain, testに分割する
        train_data = data[0:train_data_size]
        test_data = data[train_data_size:]
        return train_data, test_data

    def split_layered_train_test_data(self, data, train_data_size):
        # 引数のデータをtrain, testに分割する (layer版)
        train_data = data[:, 0:train_data_size, :]
        test_data = data[:, train_data_size:, :]
        return train_data, test_data

    def get_next_train_batch(self, layer_index, batch_size):
        selected_indices = self.train_indices[self.train_pos:self.train_pos+batch_size]
        self.train_pos += batch_size
        if self.train_pos >= self.train_data_size:
            self.prepare_train_indices()

        mus       = self.train_mus[layer_index, selected_indices, :]
        sigma_sqs = self.train_sigma_sqs[layer_index, selected_indices, :]
        hs        = self.train_hs[layer_index, selected_indices, :]
        actions   = self.train_actions[selected_indices, :]
        return mus, sigma_sqs, hs, actions

    def get_test_batch(self, layer_index, data_index, batch_size):
        indices = list(range(data_index, data_index + batch_size))

        mus       = self.test_mus[layer_index, indices, :]
        sigma_sqs = self.test_sigma_sqs[layer_index, indices, :]
        hs        = self.test_hs[layer_index, indices, :]
        actions   = self.test_actions[indices, :]
        return mus, sigma_sqs, hs, actions


class DecodeBaseDataManager():
    def __init__(self):
        # Load data
        data_all = np.load(oculomotor_data_path)
        data_images = data_all["images"]   # (10000, 20, 64, 64) uint8
        data_actions = data_all["actions"] # (10000, 20, 2) float32

        self.raw_images = data_images
        self.actions    = data_actions

        # Get data dimensions
        self.data_size, self.seq_length, self.w, self.h = data_images.shape

    def convert_images(self, images):
        return images.astype(np.float32) / 255.0

    def get_batch(self, data_index, batch_size):
        indices = list(range(data_index, data_index + batch_size))
        images = self.raw_images[indices, :, :, :]
        return self.convert_images(images)


def load_checkpoints(sess):
    saver = tf.train.Saver()
    checkpoint_dir = flags.save_dir + "/checkpoints"

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # checkpointからロード
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded checkpoint: {0}".format(
            checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")

def save_result_txt(file_path, loss):
    f = open(file_path, "w")

    line = "{0}".format(loss)
    f.write(line)
    f.write("\n")
    f.flush()
    f.close()


def train_decode(layer_index=0, use_h=False):
    # TODO: option指定
    learning_rate = 1e-3
    steps = 30 * (10**3)
    batch_size = 100

    if use_h:
        exp_name = "l{}_hz".format(layer_index)
    else:
        exp_name = "l{}_z".format(layer_index)

    checkpoint_dir = flags.save_dir + "/decode/checkpoints/{}".format(exp_name)
    log_dir = flags.save_dir + "/decode/log/{}".format(exp_name)
    loss_file_path = flags.save_dir + "/decode/loss_{}.txt".format(exp_name)
    decoded_actions_path = flags.save_dir + "/decode/actions_{}".format(exp_name)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    data_manager = DecodeDataManager()
    model = DecodeModel(flags.latent_size, flags.h_size, learning_rate, use_h, exp_name)

    sess = tf.Session()
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())

    # Train
    for step in range(steps):
        mus, sigma_sqs, hs, actions = data_manager.get_next_train_batch(
            layer_index, batch_size)

        out = sess.run(
            [
                model.train_op,
                model.loss_summary_op
            ],
            feed_dict={
                model.input_mu : mus,
                model.input_sigma_sq : sigma_sqs,
                model.input_h : hs,
                model.target_action : actions
            })
        _, summary_str = out

        if step % 10 == 0:
            summary_writer.add_summary(summary_str, step)

    # Save
    saver.save(sess, checkpoint_dir + '/' + 'checkpoint', global_step=steps)

    # Test
    test_data_size = data_manager.test_data_size
    test_loss = 0.0

    decoded_actions = []

    for pos in range(0, test_data_size, batch_size):
        mus, sigma_sqs, hs, actions = data_manager.get_test_batch(
            layer_index, pos, batch_size)
        out = sess.run(
            [
                model.loss,
                model.out_action
            ],
            feed_dict={
                model.input_mu : mus,
                model.input_sigma_sq : sigma_sqs,
                model.input_h : hs,
                model.target_action : actions
            })
        out_loss, out_actions = out
        test_loss += out_loss
        decoded_actions.append(out_actions)

    test_loss /= test_data_size

    print("test_loss: layer={}: {}".format(layer_index, test_loss))

    decoded_actions = np.array(decoded_actions, dtype=np.float32)
    # (150, 100, 2)
    decoded_actions = decoded_actions.reshape([-1, 2])
    # (15000, 2)
    
    save_result_txt(loss_file_path, test_loss)

    np.savez_compressed(decoded_actions_path,
                        action=decoded_actions)
    
    sess.close()


def collect_decode_base_data(sess, model, data_manager, layer_size):
    batch_size = 10

    seq_length = data_manager.seq_length
    data_size = data_manager.data_size
    
    w = data_manager.w
    h = data_manager.h
    
    all_mus       = [[] for _ in range(layer_size)]
    all_sigma_sqs = [[] for _ in range(layer_size)]
    all_hs        = [[] for _ in range(layer_size)]

    for i in range(0, data_size, batch_size):
        if i % 100 == 0:
            print("processed: {}".format(i))
        # テストデータセットからデータを取ってくる
        batch_data = data_manager.get_batch(i, batch_size)
        
        # batch_data = (batch_size, seq_length, 64, 64)
        batch_data = np.reshape(batch_data,
                                [batch_size, seq_length, w * h])

        # 毎バッチごとにstateの初期化をするので、stateの更新をしていない
        out = sess.run( [model.enc_mus,
                         model.enc_sigma_sqs,
                         model.hs,
                         model.last_state_h ],
                        feed_dict={
                            model.input_data: batch_data
                        })
        mus, sigma_sqs, hs, _ = out
        # (layer_size, 200, 16)

        for j in range(layer_size):
            # 3階層をそれぞれ別の器に入れていく
            all_mus[j].append(mus[j])
            all_sigma_sqs[j].append(sigma_sqs[j])
            all_hs[j].append(hs[j])

    all_mus       = np.array(all_mus)
    all_sigma_sqs = np.array(all_sigma_sqs)
    all_hs        = np.array(all_hs)

    #(layer_size, 10000/10, seq_length*batch_size, 16)
    latent_size = all_mus.shape[3]
    h_size      = all_hs.shape[3]

    all_mus       = all_mus.reshape(      [layer_size, -1, seq_length, latent_size])
    all_sigma_sqs = all_sigma_sqs.reshape([layer_size, -1, seq_length, latent_size])
    all_hs        = all_hs.reshape(       [layer_size, -1, seq_length, h_size])

    return all_mus, all_sigma_sqs, all_hs


def collect_data():
    print("collecting data")

    base_data_manager = DecodeBaseDataManager()    
    
    dataset_type    = flags.dataset_type
    layer_size      = flags.layer_size
    h_size          = flags.h_size
    latent_size     = flags.latent_size
    batch_size      = flags.batch_size
    beta            = flags.beta
    cell_type       = flags.cell_type
    binalize_output = dataset_type == "bsprite"
    downward_type   = flags.downward_type
    no_td_bp        = flags.no_td_bp
    filter_size     = flags.filter_size

    seq_length = base_data_manager.seq_length

    model = VRNN(
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
        binalize_output=binalize_output,
        reuse=False)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # ウェイトを初期化する
    sess.run(tf.global_variables_initializer())

    # ウェイトデータのロード
    load_checkpoints(sess)

    # 学習後のデータを集める
    mus, sigma_sqs, hs = collect_decode_base_data(sess, model, base_data_manager, layer_size)
    
    file_path = flags.save_dir + "/decode_base_data"

    # 圧縮して保存
    np.savez_compressed(file_path,
                        mu=mus,
                        sigma_sq=sigma_sqs,
                        h=hs)

    sess.close()

def main():
    if flags.collect_data:
        collect_data()

    for i in range(3):
        train_decode(layer_index=i, use_h=True)
    for i in range(3):
        train_decode(layer_index=i, use_h=False)


if __name__ == '__main__':
    main()
