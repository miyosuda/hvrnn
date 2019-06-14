# -*- coding: utf-8 -*-
# 
# Calculate correlation coefficient and decoding accuracy.
# (Note: correlation coefficient between the label factor and z or h is not adequate,
# so mutual information (mi.py) should be used instead.)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# for Ridge rigression
import sklearn.linear_model as lm
from sklearn.metrics import mean_squared_error, r2_score

# for SVM
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data_manager import DataManager
from trainer import Trainer
from model import VRNN
import options

# Descretized values like label takes long time to decode (because they use
# SVM for classification), so set 'analyze_discrete' False for skipping.
analyze_discrete = True

flags = options.get_options()


def load_checkpoints(sess):
    saver = tf.train.Saver()
    checkpoint_dir = flags.save_dir + "/checkpoints"

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # Load from checkpoint.
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Loaded checkpoint: {0}".format(
            checkpoint.model_checkpoint_path))
    else:
        print("Could not find old checkpoint")


def save_figure(data, title, ylabel, file_name):
    plt.figure()
    plt.ylim([0, 1])

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)

    plt.plot(
        data,
        marker="o",
        linestyle="solid",
    )

    xlabels = list(range(0, len(data)))
    plt.xticks(xlabels, xlabels)

    file_path = flags.save_dir + "/analysis/" + file_name
    plt.savefig(file_path)

    plt.close()


def save_corr_figure(data, title_name, param_name, layer_index):
    plt.figure()
    plt.ylim([0, 1])

    plt.title("Correlation {}: Layer{}".format(title_name, layer_index))
    plt.xlabel("Z index")
    plt.ylabel("correlation")

    xlabels = list(range(0, len(data)))
    colors = list(matplotlib.colors.TABLEAU_COLORS.values())
    plt.bar(xlabels, data, color=colors)

    file_name = "corr_{}_{}.png".format(param_name, layer_index)
    file_path = flags.save_dir + "/analysis/" + file_name
    plt.savefig(file_path)

    plt.close()


def save_grid_corr_figure(datas, param_names, layer_index):
    # Grid display for the correlation coefficient.
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
    ax.set_title("Correlation: Layer{}".format(layer_index))

    file_name = "corr_grid_{}.png".format(layer_index)
    file_path = flags.save_dir + "/analysis/" + file_name
    plt.savefig(file_path)
    plt.close()


def split_train_test_data(data, train_data_size):
    # Split data for train and test.
    train_data = data[0:train_data_size]
    test_data = data[train_data_size:]
    return train_data, test_data


def analyze_continuous_timeseries_values(layer_index, train_xs, train_values,
                                         test_xs, test_values):
    print("layer: {}".format(layer_index))

    train_x = train_xs[layer_index]
    train_x_shape = train_x.shape  # (800, 7, 16)

    latent_size = train_x_shape[2]

    train_x = train_x.reshape([-1, latent_size])  # (800*7, 16)
    train_y = train_values.reshape([-1])  # (800*7)

    alpha = 0.5
    clf = lm.Ridge(alpha=alpha)
    clf.fit(train_x, train_y)

    test_x = test_xs[layer_index]
    test_x_shape = test_x.shape
    test_x = test_x.reshape([-1, latent_size])  # (200*7, 16)
    test_y = test_values.reshape([-1])  # (200*7)

    pred_y = clf.predict(test_x)
    error = mean_squared_error(test_y, pred_y)
    r2 = r2_score(test_y, pred_y)

    print("variance score: {:.2f}".format(r2))

    return r2


def analyze_continuous_single_value(layer_index, train_xs, train_value,
                                    test_xs, test_value):
    seq_length = train_xs.shape[2]

    # repeat taregt values to sequence step size to align size.
    train_values = np.repeat(train_value, seq_length)  # (800*7)
    test_values = np.repeat(test_value, seq_length)

    return analyze_continuous_timeseries_values(
        layer_index, train_xs, train_values, test_xs, test_values)


def analyze_discrete_timeseries_value(layer_index, train_xs, train_values,
                                      test_xs, test_values):
    print("layer: {}".format(layer_index))

    train_x = train_xs[layer_index]
    train_x_shape = train_x.shape  # (4000, 15, 16)
    train_x = train_x.reshape([-1, train_x_shape[2]])  # (60000, 16)
    train_y = train_values.reshape([-1])

    test_x = test_xs[layer_index]
    test_x_shape = test_x.shape
    test_x = test_x.reshape([-1, test_x_shape[2]])  # (15000, 16)
    test_y = test_values.reshape([-1])

    # SVM with RBF kernel.
    #C      = 1.0
    #kernel = 'rbf'
    #gamma  = 0.01
    #clf = SVC(C=C, kernel=kernel, gamma=gamma)

    # For Linear SVN
    #clf = LinearSVC(multi_class='ovr',
    #                penalty='l2',
    #                loss='hinge',
    #                dual=True,
    #                tol=1e-3)

    # Logistic Regessionを使った場合
    clf = lm.LogisticRegression(
        solver='liblinear',
        C=100,
        tol=0.001,
        fit_intercept=True,
        multi_class='ovr',
        penalty='l2')

    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)

    accuracy = accuracy_score(test_y, pred_y)
    print("accuracy={:.2f}".format(accuracy))

    return accuracy


def analyze_discrete_single_value(layer_index, train_xs, train_value, test_xs,
                                  test_value):
    seq_length = train_xs.shape[2]
    train_values = np.repeat(train_value, seq_length)
    test_values = np.repeat(test_value, seq_length)

    return analyze_discrete_timeseries_value(
        layer_index, train_xs, train_values, test_xs, test_values)


def save_analysis_result(**kwargs):
    keys = kwargs.keys()

    lines = []
    for key in keys:
        value = kwargs[key]
        line = "{}={}".format(key, value)
        lines.append(line + "\n")

    file_name = flags.save_dir + "/analysis/analysis_result.txt"
    f = open(file_name, "w")
    f.writelines(lines)
    f.close()


def analyze_timeseries_values_corr(layer_index, test_xs, test_values):
    # Calculate correlation coefficient between z and the factor that changes along
    # time time serirs.
    test_x = test_xs[layer_index]
    test_x_shape = test_x.shape
    latent_size = test_x_shape[2]
    test_x = test_x.reshape([-1, latent_size])  # (200*7, 16)
    test_y = test_values.reshape([-1])  # (200*7)

    corrs = []
    for i in range(latent_size):
        x = test_x[:, i].reshape([-1])
        corr = np.corrcoef(x, test_y)[0, 1]
        corrs.append(np.abs(corr))  # 相関係数の絶対値を記録
    return corrs


def analyze_single_value_corr(layer_index, test_xs, test_value):
    # Calculate correlation coefficient between z and the factor that does not change
    # during the
    test_x = test_xs[layer_index]
    test_x_shape = test_x.shape
    latent_size = test_x_shape[2]
    test_x = test_x.reshape([-1, latent_size])  # (200*7, 16)
    test_y = np.repeat(test_value, test_x_shape[1])  # (200*7)

    corrs = []
    for i in range(latent_size):
        x = test_x[:, i].reshape([-1])
        corr = np.corrcoef(x, test_y)[0, 1]
        assert (x.shape == test_y.shape)
        corrs.append(np.abs(corr))  # 相関係数の絶対値を記録
    return corrs


def analyze_face(data_manager):
    print("analyzing data")

    if not os.path.exists(flags.save_dir + "/analysis"):
        os.mkdir(flags.save_dir + "/analysis")

    data_path = flags.save_dir + "/analysis_data.npz"
    data = np.load(data_path)

    z = data["z"]  # (layer_size, 1000, seq_length, latent_size))
    h = data["h"]  # (layer_size, 1000, seq_length, h_size))
    layer_size = z.shape[0]

    test_analysis_data = data_manager.test_analysis_data

    pan = test_analysis_data["pan"]  # (1000, 10)
    roll = test_analysis_data["roll"]  # (1000, 10)
    pan_alpha = test_analysis_data["pan_alpha"]  # (1000,)
    roll_alpha = test_analysis_data["roll_alpha"]  # (1000,)

    # Normalize vlues
    pan = pan % (2.0 * np.pi)
    roll = roll % (2.0 * np.pi)

    # Set start and end frame.
    start_frame = 2
    end_frame = 4

    z = z[:, :, start_frame:end_frame, :]  # (3, 1000, n, 16)
    h = h[:, :, start_frame:end_frame, :]  # (3, 1000, n, 256)
    pan = pan[:, start_frame:end_frame]  # (1000, n)
    roll = roll[:, start_frame:end_frame]  # (1000, n)

    sample_size = z.shape[1]  # 1000

    # Split original data into train and test.
    test_data_size = sample_size // 5  # 200
    train_data_size = sample_size - test_data_size  # 800

    train_z = z[:, 0:train_data_size, :, :]  # (3, 800, n, 16)
    test_z = z[:, train_data_size:, :, :]  # (3, 200, n, 16)
    train_h = h[:, 0:train_data_size, :, :]  # (3, 800, n, 256)
    test_h = h[:, train_data_size:, :, :]  # (3, 200, n, 256)

    train_pan, test_pan = split_train_test_data(pan, train_data_size)
    train_roll, test_roll = split_train_test_data(roll, train_data_size)
    train_pan_alpha, test_pan_alpha = split_train_test_data(
        pan_alpha, train_data_size)
    train_roll_alpha, test_roll_alpha = split_train_test_data(
        roll_alpha, train_data_size)

    #(800,n) (200,n)
    #(800,) (200,)

    # Calculate correlation coefficient
    for i in range(layer_size):
        corr_pan = analyze_timeseries_values_corr(i, test_z, test_pan)
        corr_roll = analyze_timeseries_values_corr(i, test_z, test_roll)
        corr_pan_alpha = analyze_single_value_corr(i, test_z, test_pan_alpha)
        corr_roll_alpha = analyze_single_value_corr(i, test_z, test_roll_alpha)

        save_corr_figure(corr_pan, "Pan", "pan", i)
        save_corr_figure(corr_roll, "Roll", "roll", i)
        save_corr_figure(corr_pan_alpha, "Pan Alpha", "roll_alpha", i)
        save_corr_figure(corr_roll_alpha, "Roll Alpha", "pan_alpha", i)

    # Calculate decoding accuracies.
    z_accs_pan = []
    z_accs_roll = []
    z_accs_pan_alpha = []
    z_accs_roll_alpha = []

    h_accs_pan = []
    h_accs_roll = []
    h_accs_pan_alpha = []
    h_accs_roll_alpha = []

    for i in range(layer_size):
        # Decode from z
        z_acc_pan = analyze_continuous_timeseries_values(
            i, train_z, train_pan, test_z, test_pan)
        z_acc_roll = analyze_continuous_timeseries_values(
            i, train_z, train_roll, test_z, test_roll)

        z_acc_pan_alpha = analyze_continuous_single_value(
            i, train_z, train_pan_alpha, test_z, test_pan_alpha)
        z_acc_roll_alpha = analyze_continuous_single_value(
            i, train_z, train_roll_alpha, test_z, test_roll_alpha)

        # Decode from h
        h_acc_pan = analyze_continuous_timeseries_values(
            i, train_h, train_pan, test_h, test_pan)
        h_acc_roll = analyze_continuous_timeseries_values(
            i, train_h, train_roll, test_h, test_roll)

        h_acc_pan_alpha = analyze_continuous_single_value(
            i, train_h, train_pan_alpha, test_h, test_pan_alpha)
        h_acc_roll_alpha = analyze_continuous_single_value(
            i, train_h, train_roll_alpha, test_h, test_roll_alpha)

        z_accs_pan.append(z_acc_pan)
        z_accs_roll.append(z_acc_roll)
        z_accs_pan_alpha.append(z_acc_pan_alpha)
        z_accs_roll_alpha.append(z_acc_roll_alpha)

        h_accs_pan.append(h_acc_pan)
        h_accs_roll.append(h_acc_roll)
        h_accs_pan_alpha.append(h_acc_pan_alpha)
        h_accs_roll_alpha.append(h_acc_roll_alpha)

    save_figure(z_accs_pan, "Pan (z)", "Accuracy (R^2)", "z_acc_pan.png")
    save_figure(z_accs_roll, "Roll (z)", "Accuracy (R^2)", "z_acc_roll.png")
    save_figure(z_accs_pan_alpha, "Pan Alpha (z)", "Accuracy (R^2)",
                "z_acc_pan_alpha.png")
    save_figure(z_accs_roll_alpha, "Roll Alpha (z)", "Accuracy (R^2)",
                "z_acc_roll_alpha.png")

    save_figure(h_accs_pan, "Pan (h)", "Accuracy (R^2)", "h_acc_pan.png")
    save_figure(h_accs_roll, "Roll (h)", "Accuracy (R^2)", "h_acc_roll.png")
    save_figure(h_accs_pan_alpha, "Pan Alpha (h)", "Accuracy (R^2)",
                "h_acc_pan_alpha.png")
    save_figure(h_accs_roll_alpha, "Roll Alpha (h)", "Accuracy (R^2)",
                "h_acc_roll_alpha.png")

    print("[Pan (z)]")
    print(z_accs_pan)

    print("[Roll (z)]")
    print(z_accs_roll)

    print("[Pan Alpha (z)]")
    print(z_accs_pan_alpha)

    print("[Roll Alpha (z)]")
    print(z_accs_roll_alpha)

    print("[Pan (h)]")
    print(h_accs_pan)

    print("[Roll (h)]")
    print(h_accs_roll)

    print("[Pan Alpha (h)]")
    print(h_accs_pan_alpha)

    print("[Roll Alpha (h)]")
    print(h_accs_roll_alpha)

    # Save result to file
    save_analysis_result(
        z_pan=z_accs_pan,
        z_roll=z_accs_roll,
        z_pan_alpha=z_accs_pan_alpha,
        z_roll_alpha=z_accs_roll_alpha,
        h_pan=h_accs_pan,
        h_roll=h_accs_roll,
        h_pan_alpha=h_accs_pan_alpha,
        h_roll_alpha=h_accs_roll_alpha)


def analyze_mnist(data_manager):
    print("analyzing data")

    if not os.path.exists(flags.save_dir + "/analysis"):
        os.mkdir(flags.save_dir + "/analysis")

    data_path = flags.save_dir + "/analysis_data.npz"
    data = np.load(data_path)

    z = data["z"]  # (layer_size, 5000, seq_length, latent_size))
    h = data["h"]  # (layer_size, 5000, seq_length, latent_size))
    layer_size = z.shape[0]

    test_analysis_data = data_manager.test_analysis_data

    pos_x = test_analysis_data["pos_x"]  # (5000, 20)
    pos_y = test_analysis_data["pos_y"]  # (5000, 20)
    label = test_analysis_data["label"]  # (5000,)
    speed = test_analysis_data["speed"]  # (5000,)
    bounce_x = test_analysis_data["bounce_x"]  # (5000,)
    bounce_y = test_analysis_data["bounce_y"]  # (5000,)

    # Set start and end frame
    start_frame = 5
    end_frame = 20

    z = z[:, :, start_frame:end_frame, :]  # (3, 5000, n, 16)
    h = h[:, :, start_frame:end_frame, :]  # (3, 5000, n, 16)
    pos_x = pos_x[:, start_frame:end_frame]  # (5000, n)
    pos_y = pos_y[:, start_frame:end_frame]  # (5000, n)
    bounce_x = bounce_x[:, start_frame:end_frame]  # (5000, n)
    bounce_y = bounce_y[:, start_frame:end_frame]  # (5000, n)

    sample_size = z.shape[1]  # 5000

    # Split original data into train and test.
    test_data_size = sample_size // 5  # 1000
    train_data_size = sample_size - test_data_size  # 4000

    train_z = z[:, 0:train_data_size, :, :]  # (3, 4000, n, 16)
    test_z = z[:, train_data_size:, :, :]  # (3, 1000, n, 16)
    train_h = h[:, 0:train_data_size, :, :]  # (3, 4000, n, 256)
    test_h = h[:, train_data_size:, :, :]  # (3, 1000, n, 256)

    train_pos_x, test_pos_x = split_train_test_data(pos_x, train_data_size)
    train_pos_y, test_pos_y = split_train_test_data(pos_y, train_data_size)
    train_label, test_label = split_train_test_data(label, train_data_size)
    train_speed, test_speed = split_train_test_data(speed, train_data_size)
    train_bounce_x, test_bounce_x = split_train_test_data(
        bounce_x, train_data_size)
    train_bounce_y, test_bounce_y = split_train_test_data(
        bounce_y, train_data_size)

    #(4000,n) (1000,n)
    #(4000,) (1000,)

    # Calculate correlation coefficient
    for i in range(layer_size):
        # correlation with z
        z_corr_pos_x = analyze_timeseries_values_corr(i, test_z, test_pos_x)
        z_corr_pos_y = analyze_timeseries_values_corr(i, test_z, test_pos_y)
        z_corr_label = analyze_single_value_corr(i, test_z, test_label)
        z_corr_speed = analyze_single_value_corr(i, test_z, test_speed)
        z_corr_bounce_x = analyze_timeseries_values_corr(
            i, test_z, test_bounce_x)
        z_corr_bounce_y = analyze_timeseries_values_corr(
            i, test_z, test_bounce_y)

        save_corr_figure(z_corr_pos_x, "Pos X", "z_pos_x", i)
        save_corr_figure(z_corr_pos_y, "Pos Y", "z_pos_y", i)
        save_corr_figure(z_corr_label, "Label", "z_label", i)
        save_corr_figure(z_corr_speed, "Speed", "z_speed", i)
        save_corr_figure(z_corr_bounce_x, "Bounce X", "z_bounce_x", i)
        save_corr_figure(z_corr_bounce_y, "Bounce Y", "z_bounce_y", i)

        # Show correlation coefficient as grid.
        save_grid_corr_figure([
            z_corr_pos_x, z_corr_pos_y, z_corr_label, z_corr_speed,
            z_corr_bounce_x, z_corr_bounce_y
        ], ["Pos X", "Pos Y", "Label", "Speed", "Bounce X", "Bounce Y"], i)

        # Correlation with h
        h_corr_pos_x = analyze_timeseries_values_corr(i, test_h, test_pos_x)
        h_corr_pos_y = analyze_timeseries_values_corr(i, test_h, test_pos_y)
        h_corr_label = analyze_single_value_corr(i, test_h, test_label)
        h_corr_speed = analyze_single_value_corr(i, test_h, test_speed)
        h_corr_bounce_x = analyze_timeseries_values_corr(
            i, test_h, test_bounce_x)
        h_corr_bounce_y = analyze_timeseries_values_corr(
            i, test_h, test_bounce_y)

        save_corr_figure(h_corr_pos_x, "Pos X", "h_pos_x", i)
        save_corr_figure(h_corr_pos_y, "Pos Y", "h_pos_y", i)
        save_corr_figure(h_corr_label, "Label", "h_label", i)
        save_corr_figure(h_corr_speed, "Speed", "h_speed", i)
        save_corr_figure(h_corr_bounce_x, "Bounce X", "h_bounce_x", i)
        save_corr_figure(h_corr_bounce_y, "Bounce Y", "h_bounce_y", i)

    # Calculate decoding accuracies
    z_accs_pos_x = []
    z_accs_pos_y = []
    z_accs_speed = []
    if analyze_discrete:
        z_accs_label = []
        z_accs_bounce_x = []
        z_accs_bounce_y = []

    h_accs_pos_x = []
    h_accs_pos_y = []
    h_accs_speed = []
    if analyze_discrete:
        h_accs_label = []
        h_accs_bounce_x = []
        h_accs_bounce_y = []

    for i in range(layer_size):
        z_acc_pos_x = analyze_continuous_timeseries_values(
            i, train_z, train_pos_x, test_z, test_pos_x)
        z_acc_pos_y = analyze_continuous_timeseries_values(
            i, train_z, train_pos_y, test_z, test_pos_y)
        z_acc_speed = analyze_continuous_single_value(i, train_z, train_speed,
                                                      test_z, test_speed)
        if analyze_discrete:
            z_acc_label = analyze_discrete_single_value(
                i, train_z, train_label, test_z, test_label)
            z_acc_bounce_x = analyze_discrete_timeseries_value(
                i, train_z, train_bounce_x, test_z, test_bounce_x)
            z_acc_bounce_y = analyze_discrete_timeseries_value(
                i, train_z, train_bounce_y, test_z, test_bounce_y)

        h_acc_pos_x = analyze_continuous_timeseries_values(
            i, train_h, train_pos_x, test_h, test_pos_x)
        h_acc_pos_y = analyze_continuous_timeseries_values(
            i, train_h, train_pos_y, test_h, test_pos_y)
        h_acc_speed = analyze_continuous_single_value(i, train_h, train_speed,
                                                      test_h, test_speed)

        if analyze_discrete:
            h_acc_label = analyze_discrete_single_value(
                i, train_h, train_label, test_h, test_label)
            h_acc_bounce_x = analyze_discrete_timeseries_value(
                i, train_h, train_bounce_x, test_h, test_bounce_x)
            h_acc_bounce_y = analyze_discrete_timeseries_value(
                i, train_h, train_bounce_y, test_h, test_bounce_y)

        z_accs_pos_x.append(z_acc_pos_x)
        z_accs_pos_y.append(z_acc_pos_y)
        z_accs_speed.append(z_acc_speed)
        if analyze_discrete:
            z_accs_label.append(z_acc_label)
            z_accs_bounce_x.append(z_acc_bounce_x)
            z_accs_bounce_y.append(z_acc_bounce_y)

        h_accs_pos_x.append(h_acc_pos_x)
        h_accs_pos_y.append(h_acc_pos_y)
        h_accs_speed.append(h_acc_speed)
        if analyze_discrete:
            h_accs_label.append(h_acc_label)
            h_accs_bounce_x.append(h_acc_bounce_x)
            h_accs_bounce_y.append(h_acc_bounce_y)

    save_figure(z_accs_pos_x, "Pos X (z)", "Accuracy (R^2)", "z_acc_pos_x.png")
    save_figure(z_accs_pos_y, "Pos Y (z)", "Accuracy (R^2)", "z_acc_pos_y.png")
    save_figure(z_accs_speed, "Speed (z)", "Accuracy (R^2)", "z_acc_speed.png")
    if analyze_discrete:
        save_figure(z_accs_label, "Label (z)", "Accuracy", "z_acc_label.png")
        save_figure(z_accs_bounce_x, "Bounce X (z)", "Accuracy",
                    "z_acc_bounce_x.png")
        save_figure(z_accs_bounce_y, "Bounce Y (z)", "Accuracy",
                    "z_acc_bounce_y.png")

    save_figure(h_accs_pos_x, "Pos X (h)", "Accuracy (R^2)", "h_acc_pos_x.png")
    save_figure(h_accs_pos_y, "Pos Y (h)", "Accuracy (R^2)", "h_acc_pos_y.png")
    save_figure(h_accs_speed, "Speed (h)", "Accuracy (R^2)", "h_acc_speed.png")
    if analyze_discrete:
        save_figure(h_accs_label, "Label (h)", "Accuracy", "h_acc_label.png")
        save_figure(h_accs_bounce_x, "Bounce X (h)", "Accuracy",
                    "h_acc_bounce_x.png")
        save_figure(h_accs_bounce_y, "Bounce Y (h)", "Accuracy",
                    "h_acc_bounce_y.png")

    print("[Pos X (z)]")
    print(z_accs_pos_x)

    print("[Pos Y (z)]")
    print(z_accs_pos_y)

    print("[Speed (z)]")
    print(z_accs_speed)

    if analyze_discrete:
        print("[Label (z)]")
        print(z_accs_label)
        print("[Bounce X (z)]")
        print(z_accs_bounce_x)
        print("[Bounce Y (z)]")
        print(z_accs_bounce_y)

    print("[Pos X (h)]")
    print(h_accs_pos_x)

    print("[Pos Y (h)]")
    print(h_accs_pos_y)

    print("[Speed (h)]")
    print(h_accs_speed)

    if analyze_discrete:
        print("[Label (h)]")
        print(h_accs_label)
        print("[Bounce X (h)]")
        print(h_accs_bounce_x)
        print("[Bounce Y (h)]")
        print(h_accs_bounce_y)

    # Save result to file
    if analyze_discrete:
        save_analysis_result(
            z_pos_x=z_accs_pos_x,
            z_pos_y=z_accs_pos_y,
            z_speed=z_accs_speed,
            z_label=z_accs_label,
            z_bounce_x=z_accs_bounce_x,
            z_bounce_y=z_accs_bounce_y,
            h_pos_x=h_accs_pos_x,
            h_pos_y=h_accs_pos_y,
            h_speed=h_accs_speed,
            h_label=h_accs_label,
            h_bounce_x=h_accs_bounce_x,
            h_bounce_y=h_accs_bounce_y)

    else:
        save_analysis_result(
            z_pos_x=z_accs_pos_x,
            z_pos_y=z_accs_pos_y,
            z_speed=z_accs_speed,
            h_pos_x=h_accs_pos_x,
            h_pos_y=h_accs_pos_y,
            h_speed=h_accs_speed)


def analyze_bsprite(data_manager):
    print("analyzing data")

    if not os.path.exists(flags.save_dir + "/analysis"):
        os.mkdir(flags.save_dir + "/analysis")

    data_path = flags.save_dir + "/analysis_data.npz"
    data = np.load(data_path)

    z = data["z"]  # (layer_size, 5000, seq_length, latent_size))
    h = data["h"]  # (layer_size, 5000, seq_length, latent_size))
    layer_size = z.shape[0]

    test_analysis_data = data_manager.test_analysis_data

    pos_x = test_analysis_data["pos_x"]  # (5000, 20)
    pos_y = test_analysis_data["pos_y"]  # (5000, 20)
    label = test_analysis_data["label"]  # (5000,)
    speed = test_analysis_data["speed"]  # (5000,)
    bounce_x = test_analysis_data["bounce_x"]  # (5000,)
    bounce_y = test_analysis_data["bounce_y"]  # (5000,)

    # Set start and end frame
    start_frame = 5
    end_frame = 20

    z = z[:, :, start_frame:end_frame, :]  # (3, 5000, n, 16)
    h = h[:, :, start_frame:end_frame, :]  # (3, 5000, n, 16)
    pos_x = pos_x[:, start_frame:end_frame]  # (5000, n)
    pos_y = pos_y[:, start_frame:end_frame]  # (5000, n)
    bounce_x = bounce_x[:, start_frame:end_frame]  # (5000, n)
    bounce_y = bounce_y[:, start_frame:end_frame]  # (5000, n)

    sample_size = z.shape[1]  # 5000

    # Split original data into train and test.
    test_data_size = sample_size // 5  # 1000
    train_data_size = sample_size - test_data_size  # 4000

    train_z = z[:, 0:train_data_size, :, :]  # (3, 4000, n, 16)
    test_z = z[:, train_data_size:, :, :]  # (3, 1000, n, 16)
    train_h = h[:, 0:train_data_size, :, :]  # (3, 4000, n, 256)
    test_h = h[:, train_data_size:, :, :]  # (3, 1000, n, 256)

    train_pos_x, test_pos_x = split_train_test_data(pos_x, train_data_size)
    train_pos_y, test_pos_y = split_train_test_data(pos_y, train_data_size)
    train_label, test_label = split_train_test_data(label, train_data_size)
    train_speed, test_speed = split_train_test_data(speed, train_data_size)
    train_bounce_x, test_bounce_x = split_train_test_data(
        bounce_x, train_data_size)
    train_bounce_y, test_bounce_y = split_train_test_data(
        bounce_y, train_data_size)

    #(4000,n) (1000,n)
    #(4000,) (1000,)

    # Calculate correlation coefficient
    for i in range(layer_size):
        # correlation with z
        z_corr_pos_x = analyze_timeseries_values_corr(i, test_z, test_pos_x)
        z_corr_pos_y = analyze_timeseries_values_corr(i, test_z, test_pos_y)
        z_corr_label = analyze_single_value_corr(i, test_z, test_label)
        z_corr_speed = analyze_single_value_corr(i, test_z, test_speed)
        z_corr_bounce_x = analyze_timeseries_values_corr(
            i, test_z, test_bounce_x)
        z_corr_bounce_y = analyze_timeseries_values_corr(
            i, test_z, test_bounce_y)

        save_corr_figure(z_corr_pos_x, "Pos X", "z_pos_x", i)
        save_corr_figure(z_corr_pos_y, "Pos Y", "z_pos_y", i)
        save_corr_figure(z_corr_label, "Label", "z_label", i)
        save_corr_figure(z_corr_speed, "Speed", "z_speed", i)
        save_corr_figure(z_corr_bounce_x, "Bounce X", "z_bounce_x", i)
        save_corr_figure(z_corr_bounce_y, "Bounce Y", "z_bounce_y", i)

        # Show correlation coefficient as grid.
        save_grid_corr_figure([
            z_corr_pos_x, z_corr_pos_y, z_corr_label, z_corr_speed,
            z_corr_bounce_x, z_corr_bounce_y
        ], ["Pos X", "Pos Y", "Label", "Speed", "Bounce X", "Bounce Y"], i)

        # Correlation with h
        h_corr_pos_x = analyze_timeseries_values_corr(i, test_h, test_pos_x)
        h_corr_pos_y = analyze_timeseries_values_corr(i, test_h, test_pos_y)
        h_corr_label = analyze_single_value_corr(i, test_h, test_label)
        h_corr_speed = analyze_single_value_corr(i, test_h, test_speed)
        h_corr_bounce_x = analyze_timeseries_values_corr(
            i, test_h, test_bounce_x)
        h_corr_bounce_y = analyze_timeseries_values_corr(
            i, test_h, test_bounce_y)

        save_corr_figure(h_corr_pos_x, "Pos X", "h_pos_x", i)
        save_corr_figure(h_corr_pos_y, "Pos Y", "h_pos_y", i)
        save_corr_figure(h_corr_label, "Label", "h_label", i)
        save_corr_figure(h_corr_speed, "Speed", "h_speed", i)
        save_corr_figure(h_corr_bounce_x, "Bounce X", "h_bounce_x", i)
        save_corr_figure(h_corr_bounce_y, "Bounce Y", "h_bounce_y", i)

    # Calculate decoding accuracies
    z_accs_pos_x = []
    z_accs_pos_y = []
    z_accs_speed = []
    if analyze_discrete:
        z_accs_label = []
        z_accs_bounce_x = []
        z_accs_bounce_y = []

    h_accs_pos_x = []
    h_accs_pos_y = []
    h_accs_speed = []
    if analyze_discrete:
        h_accs_label = []
        h_accs_bounce_x = []
        h_accs_bounce_y = []

    for i in range(layer_size):
        z_acc_pos_x = analyze_continuous_timeseries_values(
            i, train_z, train_pos_x, test_z, test_pos_x)
        z_acc_pos_y = analyze_continuous_timeseries_values(
            i, train_z, train_pos_y, test_z, test_pos_y)
        z_acc_speed = analyze_continuous_single_value(i, train_z, train_speed,
                                                      test_z, test_speed)
        if analyze_discrete:
            z_acc_label = analyze_discrete_single_value(
                i, train_z, train_label, test_z, test_label)
            z_acc_bounce_x = analyze_discrete_timeseries_value(
                i, train_z, train_bounce_x, test_z, test_bounce_x)
            z_acc_bounce_y = analyze_discrete_timeseries_value(
                i, train_z, train_bounce_y, test_z, test_bounce_y)

        h_acc_pos_x = analyze_continuous_timeseries_values(
            i, train_h, train_pos_x, test_h, test_pos_x)
        h_acc_pos_y = analyze_continuous_timeseries_values(
            i, train_h, train_pos_y, test_h, test_pos_y)
        h_acc_speed = analyze_continuous_single_value(i, train_h, train_speed,
                                                      test_h, test_speed)

        if analyze_discrete:
            h_acc_label = analyze_discrete_single_value(
                i, train_h, train_label, test_h, test_label)
            h_acc_bounce_x = analyze_discrete_timeseries_value(
                i, train_h, train_bounce_x, test_h, test_bounce_x)
            h_acc_bounce_y = analyze_discrete_timeseries_value(
                i, train_h, train_bounce_y, test_h, test_bounce_y)

        z_accs_pos_x.append(z_acc_pos_x)
        z_accs_pos_y.append(z_acc_pos_y)
        z_accs_speed.append(z_acc_speed)
        if analyze_discrete:
            z_accs_label.append(z_acc_label)
            z_accs_bounce_x.append(z_acc_bounce_x)
            z_accs_bounce_y.append(z_acc_bounce_y)

        h_accs_pos_x.append(h_acc_pos_x)
        h_accs_pos_y.append(h_acc_pos_y)
        h_accs_speed.append(h_acc_speed)
        if analyze_discrete:
            h_accs_label.append(h_acc_label)
            h_accs_bounce_x.append(h_acc_bounce_x)
            h_accs_bounce_y.append(h_acc_bounce_y)

    save_figure(z_accs_pos_x, "Pos X (z)", "Accuracy (R^2)", "z_acc_pos_x.png")
    save_figure(z_accs_pos_y, "Pos Y (z)", "Accuracy (R^2)", "z_acc_pos_y.png")
    save_figure(z_accs_speed, "Speed (z)", "Accuracy (R^2)", "z_acc_speed.png")
    if analyze_discrete:
        save_figure(z_accs_label, "Label (z)", "Accuracy", "z_acc_label.png")
        save_figure(z_accs_bounce_x, "Bounce X (z)", "Accuracy",
                    "z_acc_bounce_x.png")
        save_figure(z_accs_bounce_y, "Bounce Y (z)", "Accuracy",
                    "z_acc_bounce_y.png")

    save_figure(h_accs_pos_x, "Pos X (h)", "Accuracy (R^2)", "h_acc_pos_x.png")
    save_figure(h_accs_pos_y, "Pos Y (h)", "Accuracy (R^2)", "h_acc_pos_y.png")
    save_figure(h_accs_speed, "Speed (h)", "Accuracy (R^2)", "h_acc_speed.png")
    if analyze_discrete:
        save_figure(h_accs_label, "Label (h)", "Accuracy", "h_acc_label.png")
        save_figure(h_accs_bounce_x, "Bounce X (h)", "Accuracy",
                    "h_acc_bounce_x.png")
        save_figure(h_accs_bounce_y, "Bounce Y (h)", "Accuracy",
                    "h_acc_bounce_y.png")

    print("[Pos X (z)]")
    print(z_accs_pos_x)

    print("[Pos Y (z)]")
    print(z_accs_pos_y)

    print("[Speed (z)]")
    print(z_accs_speed)

    if analyze_discrete:
        print("[Label (z)]")
        print(z_accs_label)
        print("[Bounce X (z)]")
        print(z_accs_bounce_x)
        print("[Bounce Y (z)]")
        print(z_accs_bounce_y)

    print("[Pos X (h)]")
    print(h_accs_pos_x)

    print("[Pos Y (h)]")
    print(h_accs_pos_y)

    print("[Speed (h)]")
    print(h_accs_speed)

    if analyze_discrete:
        print("[Label (h)]")
        print(h_accs_label)
        print("[Bounce X (h)]")
        print(h_accs_bounce_x)
        print("[Bounce Y (h)]")
        print(h_accs_bounce_y)

    # Save result to file
    if analyze_discrete:
        save_analysis_result(
            z_pos_x=z_accs_pos_x,
            z_pos_y=z_accs_pos_y,
            z_speed=z_accs_speed,
            z_label=z_accs_label,
            z_bounce_x=z_accs_bounce_x,
            z_bounce_y=z_accs_bounce_y,
            h_pos_x=h_accs_pos_x,
            h_pos_y=h_accs_pos_y,
            h_speed=h_accs_speed,
            h_label=h_accs_label,
            h_bounce_x=h_accs_bounce_x,
            h_bounce_y=h_accs_bounce_y)
    else:
        save_analysis_result(
            z_pos_x=z_accs_pos_x,
            z_pos_y=z_accs_pos_y,
            z_speed=z_accs_speed,
            h_pos_x=h_accs_pos_x,
            h_pos_y=h_accs_pos_y,
            h_speed=h_accs_speed)

        
def analyze_oculomotor(data_manager):
    print("analyzing data")

    if not os.path.exists(flags.save_dir + "/analysis"):
        os.mkdir(flags.save_dir + "/analysis")

    data_path = flags.save_dir + "/analysis_data.npz"
    data = np.load(data_path)

    z = data["z"]  # (layer_size, 500, seq_length, latent_size))
    h = data["h"]  # (layer_size, 500, seq_length, latent_size))
    layer_size = z.shape[0]

    test_analysis_data = data_manager.test_analysis_data

    action_x = test_analysis_data["action_x"] # (500, 20)
    action_y = test_analysis_data["action_y"] # (500, 20)
    angle_x  = test_analysis_data["angle_x"]  # (500, 20)
    angle_y  = test_analysis_data["angle_y"]  # (500, 20)

    # Set start and end frame
    start_frame = 5
    end_frame = 20

    z = z[:, :, start_frame:end_frame, :]  # (3, 500, n, 16)
    h = h[:, :, start_frame:end_frame, :]  # (3, 500, n, 16)
    action_x = action_x[:, start_frame:end_frame] # (500, n)
    action_y = action_y[:, start_frame:end_frame] # (500, n)
    angle_x  = angle_x[:, start_frame:end_frame]  # (500, n)
    angle_y  = angle_y[:, start_frame:end_frame]  # (500, n)

    sample_size = z.shape[1]  # 500

    # Here only correlation coefficient is analyzed, so there is no train/test split.

    # Calcuate correlation coefficient.
    for i in range(layer_size):
        # Correlation with z
        z_corr_action_x = analyze_timeseries_values_corr(i, z, action_x)
        z_corr_action_y = analyze_timeseries_values_corr(i, z, action_y)
        z_corr_angle_x  = analyze_timeseries_values_corr(i, z, angle_x)
        z_corr_angle_y  = analyze_timeseries_values_corr(i, z, angle_y)

        save_corr_figure(z_corr_action_x, "Action X", "z_action_x", i)
        save_corr_figure(z_corr_action_y, "Action Y", "z_action_y", i)
        save_corr_figure(z_corr_angle_x,  "Angle X",  "z_angle_x", i)
        save_corr_figure(z_corr_angle_y,  "Angle Y",  "z_angle_y", i)

        # Show correlation coefficient as grid.
        save_grid_corr_figure([
            z_corr_action_x,
            z_corr_action_y,
            z_corr_angle_x,
            z_corr_angle_y
        ], ["Action X", "Action Y", "Angle X", "Angle Y"], i)

        # Correlation with h
        h_corr_action_x = analyze_timeseries_values_corr(i, h, action_x)
        h_corr_action_y = analyze_timeseries_values_corr(i, h, action_y)
        h_corr_angle_x  = analyze_timeseries_values_corr(i, h, angle_x)
        h_corr_angle_y  = analyze_timeseries_values_corr(i, h, angle_y)

        save_corr_figure(h_corr_action_x, "Action X", "h_action_x", i)
        save_corr_figure(h_corr_action_y, "Action Y", "h_action_y", i)
        save_corr_figure(h_corr_angle_x,  "Angle X",  "h_angle_x", i)
        save_corr_figure(h_corr_angle_y,  "Angle Y",  "h_angle_y", i)        
        
    

def collect_data(data_manager):
    """ Collect analysis data """
    print("collecting data")
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

    seq_length = data_manager.seq_length

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
        binalize_output=binalize_output,
        reuse=False)

    trainer = Trainer(data_manager, train_model, None, None,
                      flags.learning_rate, flags.use_denoising)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Collect data for regression analysis
    data_size = data_manager.test_data_size

    # Initialize weight
    sess.run(tf.global_variables_initializer())

    # Load weight data
    load_checkpoints(sess)

    # Collect trained data.
    zses, hses = trainer.collect_analysis_data(sess, layer_size, data_size)

    file_path = flags.save_dir + "/analysis_data"

    # Compress and save.
    np.savez_compressed(file_path, z=zses, h=hses)


def main():
    data_manager = DataManager.get_data_manager(flags.dataset_type)

    if flags.collect_data:
        collect_data(data_manager)

    dataset_type = flags.dataset_type

    if dataset_type == "face":
        analyze_face(data_manager)
    elif dataset_type == "mnist":
        analyze_mnist(data_manager)
    elif dataset_type == "bsprite":
        analyze_bsprite(data_manager)
    elif dataset_type == "oculomotor":
        analyze_oculomotor(data_manager)


if __name__ == '__main__':
    main()
