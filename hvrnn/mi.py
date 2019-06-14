# -*- coding: utf-8 -*-
# Calculate mutual information between z and the factor using MINE.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data_manager import DataManager
import options

flags = options.get_options()


class ConvMineModel(object):
    """ MINE model with X input as image and using CNN. """

    def __init__(self, name, y_size, filter_size=16):
        self.filter_size = filter_size
        self.lr = 0.0001

        self.create_network(name, y_size)

    def create_network(self, name, y_size):
        with tf.variable_scope(name):
            self.x_in = tf.placeholder(tf.float32, [None, 64, 64])
            self.y_in = tf.placeholder(tf.float32, [None, y_size])
            self.y_shuffle_in = tf.random_shuffle(self.y_in, name="y_shuffle")

            x_in_reshaped = tf.reshape(self.x_in, [-1, 64, 64, 1])
            h1 = tf.layers.conv2d(
                x_in_reshaped,
                filters=self.filter_size,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv1")
            # (-1, 32, 32, 16)
            h2 = tf.layers.conv2d(
                h1,
                filters=self.filter_size,
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv2")
            # (-1, 16, 16, 16)
            h3 = tf.layers.conv2d(
                h2,
                filters=self.filter_size // 4,  # filter_size小さく
                kernel_size=[4, 4],
                strides=(2, 2),
                padding="same",
                activation=tf.nn.relu,
                name="conv3")
            # (-1, 8, 8, 4)
            x_flat = tf.layers.flatten(h3)
            # (-1, 256)

            fc0_1 = tf.layers.dense(self.y_in, 128, name="fc1", reuse=False)
            fc0_2 = tf.layers.dense(fc0_1, 256, name="fc2", reuse=False)

            fc1_1 = tf.layers.dense(
                self.y_shuffle_in, 128, name="fc1", reuse=True)
            fc1_2 = tf.layers.dense(fc1_1, 256, name="fc2", reuse=True)

            pred_xy = tf.layers.dense(
                tf.nn.relu(x_flat + fc0_2), 1, name="fc3", reuse=False)
            # x, y  -> should be mazimized.

            pred_x_y = tf.layers.dense(
                tf.nn.relu(x_flat + fc1_2), 1, name="fc3", reuse=True)
            # x, y_shuffle -> should be minimized.

            self.dual0 = tf.reduce_mean(pred_xy)
            self.dual1 = tf.log(tf.reduce_mean(tf.exp(pred_x_y)))

            self.loss = -(self.dual0 - self.dual1)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.loss_summary_op = tf.summary.scalar("loss", self.loss)
            self.dual0_summary_op = tf.summary.scalar("dual0", self.dual0)
            self.dual1_summary_op = tf.summary.scalar("dual1", self.dual1)

            self.summary_op = tf.summary.merge([
                self.loss_summary_op, self.dual0_summary_op,
                self.dual1_summary_op
            ])


class FCMineModel(object):
    """ MINE model to calculate factor and each dimension of z. """

    def __init__(self, name, unit_size=10):
        self.lr = 0.0001
        self.unit_size = unit_size
        self.create_network(name)

    def create_network(self, name):
        with tf.variable_scope(name):
            self.x_in = tf.placeholder(tf.float32, [None, 1])
            self.y_in = tf.placeholder(tf.float32, [None, 1])
            self.y_shuffle_in = tf.random_shuffle(self.y_in, name="y_shuffle")

            x_h1 = tf.layers.dense(
                self.x_in, self.unit_size, name="x_fc1", reuse=False)

            y_h1 = tf.layers.dense(
                self.y_in, self.unit_size, name="y_fc1", reuse=False)

            ys_h1 = tf.layers.dense(
                self.y_shuffle_in, self.unit_size, name="y_fc1", reuse=True)

            pred_xy = tf.layers.dense(
                tf.nn.relu(x_h1 + y_h1), 1, name="fc_xy", reuse=False)
            # x, y  -> should be maximized.

            pred_x_y = tf.layers.dense(
                tf.nn.relu(x_h1 + ys_h1), 1, name="fc_xy", reuse=True)
            # x, y_shuffle -> should be minimized.

            self.dual0 = tf.reduce_mean(pred_xy)
            self.dual1 = tf.log(tf.reduce_mean(tf.exp(pred_x_y)))

            self.loss = -(self.dual0 - self.dual1)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            self.loss_summary_op = tf.summary.scalar("loss", self.loss)
            self.dual0_summary_op = tf.summary.scalar("dual0", self.dual0)
            self.dual1_summary_op = tf.summary.scalar("dual1", self.dual1)

            self.summary_op = tf.summary.merge([
                self.loss_summary_op, self.dual0_summary_op,
                self.dual1_summary_op
            ])


def calc_entropy_discrete(values):
    """ Calculate entropy for discrete random variable. """
    ids, counts = np.unique(values, return_counts=True)
    size = np.sum(counts)
    probs = counts / size

    h = 0.0
    for prob in probs:
        h -= prob * np.ma.log(prob)
    return h


def calc_entropy_continuous(values, bin_size=300):
    """ Calculate entropy for continuous random variable. """
    pds, bins = np.histogram(values, bins=bin_size, density=True)
    dx = bins[1] - bins[0]

    h = 0.0
    for pd in pds:
        h -= pd * np.ma.log(pd) * dx
    return h


def analyze_mi(x, y, model, is_image, summary_writer, step_size):
    # x: (75000, 64, 64) or (75000)
    # y: (75000, 16)     or (75000)

    data_size = x.shape[0]

    batch_size = 250

    indices = list(range(data_size))
    random.shuffle(indices)

    pos = 0

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Train
    for i in range(step_size):
        selected_indices = indices[pos:pos + batch_size]
        pos += batch_size
        if pos >= data_size:
            random.shuffle(indices)
            pos = 0

        if is_image:
            xs = x[selected_indices, :, :]
        else:
            xs = x[selected_indices, :]
        ys = y[selected_indices, :]

        out = sess.run(
            [
                model.train_op, model.loss, model.dual0, model.dual1,
                model.summary_op
            ],
            feed_dict={
                model.x_in: xs,
                model.y_in: ys,
            })
        _, loss_ret, dual0_ret, dual1_ret, summary_str = out
        if i % 10 == 0:
            summary_writer.add_summary(summary_str, i)

    # Calc final MI value.
    mi_values = []
    dual0_values = []
    dual1_values = []

    eval_batch_count = len(x) // batch_size
    pos = 0

    for i in range(eval_batch_count):
        selected_indices = indices[pos:pos + batch_size]
        pos += batch_size
        if is_image:
            xs = x[selected_indices, :, :]
        else:
            xs = x[selected_indices, :]
        ys = y[selected_indices, :]

        out = sess.run(
            [model.loss, model.dual0, model.dual1],
            feed_dict={
                model.x_in: xs,
                model.y_in: ys,
            })
        loss_ret, dual0_ret, dual1_ret = out

        # Average
        mi_values.append(-loss_ret)
        dual0_values.append(dual0_ret)
        dual1_values.append(dual1_ret)

    mi = np.mean(mi_values)
    dual0 = np.mean(dual0_values)
    dual1 = np.mean(dual1_values)
    print(mi, dual0_ret, dual1_ret)
    sess.close()

    return mi, dual0, dual1


def analyze_image_and_vector_mi(x, y, name, summary_writer, step_size):
    """ Calculate mutual info b/w image and vector. """
    print("analyzing mutual information: {}".format(name))

    y_size = y.shape[2]

    x = np.reshape(x, [-1, x.shape[2], x.shape[3]])  # (75000, 64, 64)
    y = np.reshape(y, [-1, y_size])  # (75000, 16) or (75000, 256)

    model = ConvMineModel(name, y_size=y_size)

    return analyze_mi(x, y, model, True, summary_writer, step_size)


def analyze_scalar_and_scalar_mi(x, y, name, summary_writer, step_size):
    """ Calculate mutual info b/w scalers. """
    # x: (5000, n, 1) ?
    # y: (5000, n)
    print("analyzing mutual information: {}".format(name))

    x = np.reshape(x, [-1, 1])  # (75000)
    y = np.reshape(y, [-1, 1])  # (75000)

    model = FCMineModel(name)

    return analyze_mi(x, y, model, False, summary_writer, step_size)


def record_mi(f, name, mi, dual0, dual1):
    """ Save mutual info into file. """
    line = "{0}: {1:.3f}, {2:.3f}, {3:.3f}".format(name, mi, dual0, dual1)
    f.write(line)
    f.write("\n")
    f.flush()


def save_grid_mi_figure(datas, param_names, layer_index):
    """ Grid display of mutual info. """
    plt.figure()

    datas = np.array(datas)

    # Make output size 640x240
    fig, ax = plt.subplots(figsize=(6.4, 2.4))

    # Fit vertical size of the color bar as that of the heatmap.
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="2%", pad=0.05)

    fig.add_axes(ax_cb)

    # Show heatmap
    im = ax.imshow(datas)

    # Make colorbar.
    cbar = ax.figure.colorbar(im, cax=ax_cb)

    # Show Y axis label.
    ax.set_yticks(np.arange(datas.shape[0]))
    ax.set_yticklabels(param_names)

    # Remove x axis
    ax.tick_params(labelbottom=False, bottom=False)
    ax.set_xticklabels([])

    # Show X axis label.
    ax.set_xlabel("Z")

    # Show title.
    ax.set_title("Normalized Mutual Info: Layer{}".format(layer_index + 1))

    file_name = "mi_grid_norm_{}.png".format(layer_index)
    file_path = flags.save_dir + "/analysis/" + file_name
    plt.savefig(file_path)
    plt.close()


def analyze_input_mi(data_manager, z, h, start_frame, end_frame,
                     summary_writer, step_size):
    """ Calculate mutual info b/w input image with z and h """

    test_images = data_manager.raw_test_images
    # (5000, 20, 64, 64)

    # Skip first some frames.
    z = z[:, :, start_frame:end_frame, :]  # (3, 5000, n, 16)
    h = h[:, :, start_frame:end_frame, :]  # (3, 5000, n, 256)
    x = test_images[:, start_frame:end_frame, :, :]  # (5000, n, 64, 64)

    file_name = flags.save_dir + "/mi_x_z_h.txt"
    f = open(file_name, "w")

    layer_size = z.shape[0]

    for i in range(layer_size):
        name = "mi_z_{}".format(i)
        mi, dual0, dual1 = analyze_image_and_vector_mi(
            x, z[i], name, summary_writer, step_size)
        record_mi(f, name, mi, dual0, dual1)

        name = "mi_h_{}".format(i)
        mi, dual0, dual1 = analyze_image_and_vector_mi(
            x, h[i], name, summary_writer, step_size)
        record_mi(f, name, mi, dual0, dual1)

    f.close()


def analyze_factor_mi(data_manager, z, start_frame, end_frame, summary_writer,
                      step_size):
    """ Calculate mutual info b/w each factor of the input and each dimension of z. """

    test_analysis_data = data_manager.test_analysis_data

    pos_x = test_analysis_data["pos_x"]  # (5000, 20)
    pos_y = test_analysis_data["pos_y"]  # (5000, 20)
    label = test_analysis_data["label"]  # (5000,)
    speed = test_analysis_data["speed"]  # (5000,)
    bounce_x = test_analysis_data["bounce_x"]  # (5000, 20)
    bounce_y = test_analysis_data["bounce_y"]  # (5000, 20)

    seq_length = pos_x.shape[1]

    # Extend label and speed as sequence length.
    label = np.repeat(label, seq_length).reshape([-1, seq_length])
    speed = np.repeat(speed, seq_length).reshape([-1, seq_length])

    # Remove first some frames.
    z = z[:, :, start_frame:end_frame, :]  # (3, 5000, n, 16)

    pos_x = pos_x[:, start_frame:end_frame]  # (5000, n)
    pos_y = pos_y[:, start_frame:end_frame]  # (5000, n)
    label = label[:, start_frame:end_frame]  # (5000, n)
    speed = speed[:, start_frame:end_frame]  # (5000, n)
    bounce_x = bounce_x[:, start_frame:end_frame]  # (5000, n)
    bounce_y = bounce_y[:, start_frame:end_frame]  # (5000, n)

    factors = [pos_x, pos_y, label, speed, bounce_x, bounce_y]
    factor_names = ["pos_x", "pos_y", "label", "speed", "bounce_x", "bounce_y"]
    param_names = ["Pos X", "Pos Y", "Label", "Speed", "Bounce X", "Bounce Y"]

    sample_size = z.shape[1]  # 5000

    file_name = flags.save_dir + "/mi_f_z.txt"
    f = open(file_name, "w")

    z_dim = z.shape[3]

    layer_size = z.shape[0]
    factor_mi_raw_results = np.empty(
        [layer_size, len(factors), z_dim], np.float32)
    factor_mi_normalized_results = np.empty(
        [layer_size, len(factors), z_dim], np.float32)

    # Calculate entropy of the factor.
    # Calculate entropy of continuous factor.
    pos_x_h = calc_entropy_continuous(
        pos_x.reshape([-1]), bin_size=300)  # 3.68
    pos_y_h = calc_entropy_continuous(
        pos_y.reshape([-1]), bin_size=300)  # 3.67
    # Calculate entropy of discrete factor.
    label_h = calc_entropy_discrete(label.reshape([-1]))  # 2.30
    speed_h = calc_entropy_discrete(speed.reshape([-1]))  # 1.61
    bounce_x_h = calc_entropy_discrete(bounce_x.reshape([-1]))  # 0.31
    bounce_y_h = calc_entropy_discrete(bounce_y.reshape([-1]))  # 0.31

    factor_hs = [pos_x_h, pos_y_h, label_h, speed_h, bounce_x_h, bounce_y_h]

    for i in range(layer_size):
        for factor, factor_name, j in zip(factors, factor_names,
                                          range(len(factors))):
            for k in range(z_dim):
                name = "mi_f_{}_{}_z{}".format(i, factor_name, k)
                z_layer = z[i]  # z of each layer
                z_i = z_layer[:, :, k]  # one dim of z
                mi, dual0, dual1 = analyze_scalar_and_scalar_mi(
                    factor, z_i, name, summary_writer, step_size)
                record_mi(f, name, mi, dual0, dual1)
                factor_mi_raw_results[i, j, k] = mi
                # Nomalize mutual info with entropy of each factor.
                factor_mi_normalized_results[i, j, k] = mi / factor_hs[j]
    f.close()

    for i in range(layer_size):
        save_grid_mi_figure(factor_mi_normalized_results[i], param_names, i)

    np.save(flags.save_dir + "/mi_data_raw", factor_mi_raw_results)
    np.save(flags.save_dir + "/mi_data_normalized",
            factor_mi_normalized_results)


def analyze(data_manager):
    """ Mutual info analysis. """
    print("analyzing data")

    data_path = flags.save_dir + "/analysis_data.npz"
    data = np.load(data_path)

    z = data["z"]  # (layer_size, 5000, seq_length, latent_size))
    h = data["h"]  # (layer_size, 5000, seq_length, latent_size))

    # Set start and end frame
    start_frame = 5
    end_frame = 20

    log_dir = flags.save_dir + "/log"
    summary_writer = tf.summary.FileWriter(log_dir, None)

    step_size = 100000

    # Analyze mutual info b/w input X and z,h.
    #analyze_input_mi(data_manager, z, h, start_frame, end_frame, summary_writer, step_size)

    # Calculate mutual info b/w factor and z.
    analyze_factor_mi(data_manager, z, start_frame, end_frame, summary_writer,
                      step_size)


def main():
    if not os.path.exists(flags.save_dir + "/analysis"):
        os.mkdir(flags.save_dir + "/analysis")

    data_manager = DataManager.get_data_manager(flags.dataset_type)

    dataset_type = flags.dataset_type

    analyze(data_manager)


if __name__ == '__main__':
    main()
