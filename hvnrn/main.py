# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import cv2
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from data_manager import DataManager
from trainer import Trainer
from model import VRNN
import utils
import options

flags = options.get_options()


def load_checkpoints(sess):
    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_dir = flags.save_dir + "/checkpoints"

    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if checkpoint and checkpoint.model_checkpoint_path:
        # checkpointからロード
        saver.restore(sess, checkpoint.model_checkpoint_path)
        # ファイル名から保存時のstep数を復元
        tokens = checkpoint.model_checkpoint_path.split("-")
        step = int(tokens[1])
        print("Loaded checkpoint: {0}, step={1}".format(
            checkpoint.model_checkpoint_path, step))
        return saver, step + 1
    else:
        print("Could not find old checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        return saver, 0


def save_checkponts(sess, saver, global_step):
    checkpoint_dir = flags.save_dir + "/checkpoints"
    saver.save(
        sess, checkpoint_dir + '/' + 'checkpoint', global_step=global_step)
    print("Checkpoint saved")


def save_plt(plt, file_name, dir_name):
    file_path = dir_name + "/" + file_name
    plt.savefig(file_path)


def save_figure(data,
                file_name,
                dir_name,
                range_y=None,
                ylabel=None,
                title=None):
    # 最初の2ステップはまだRNNが安定しておらず極端な値を出すので飛ばす
    initial_skip_timesteps = 2

    plt.figure()
    if range_y is not None:
        # When specifying y range, apply like range_y=[0.0,1.0]
        plt.ylim(range_y)

    dim = data.shape[1]  # 次元数
    for i in range(dim):
        v = data[initial_skip_timesteps:, i]
        label = "z{}".format(i)
        plt.plot(v, label=label)

    # レジェンドの表示
    plt.legend(
        bbox_to_anchor=(1.005, 1),
        loc='upper left',
        borderaxespad=0,
        fontsize=8)

    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xlabel("Timestep")
    xlocs = list(range(0, data.shape[0] - initial_skip_timesteps, 2))
    xlabels = list(range(initial_skip_timesteps, data.shape[0], 2))
    plt.xticks(xlocs, xlabels)
    save_plt(plt, file_name, dir_name)
    plt.close()


def train(sess, trainer, saver, summary_writer, start_step):
    # コマンドライン引数を保存しておく
    options.save_flags(flags)

    for i in range(start_step, flags.steps):
        # 学習
        trainer.train(
            sess, summary_writer, batch_size=flags.batch_size, step=i)

        if i % flags.save_interval == flags.save_interval - 1:
            # 保存
            save_checkponts(sess, saver, i)

        if i % flags.generate_interval == flags.generate_interval - 1:
            # 生成の確認
            generate(sess, trainer, summary_writer, step=i)

        if i % flags.predict_interval == flags.predict_interval - 1:
            # 予測誤差の確認
            predict(sess, trainer)
            # 時系列入力後の予測精度の確認
            forecast(sess, trainer)

        if i % flags.test_interval == flags.test_interval - 1:
            # テスト
            trainer.test(
                sess, summary_writer, batch_size=flags.batch_size, step=i)


def generate(sess, trainer, summary_writer=None, step=None):
    """ 時系列画像データの生成. """
    print("generate data")

    image_dir = flags.save_dir + "/generated"
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    samples = trainer.generate(sess, 30, summary_writer, step)

    for i in range(len(samples)):
        x = samples[i]
        file_path = image_dir + "/img{0:0>2}.png".format(i)
        imsave(file_path, x)

    generated_concated = utils.concat_images_in_rows(samples, 3)
    file_path = image_dir + "/generated.png"
    imsave(file_path, generated_concated)


def predict_sub(sess, trainer, data_index, off_forecast):
    """ 予測誤差の確認. """
    print("predict data")

    if off_forecast:
        prefix = "off"
    else:
        prefix = "on"

    image_dir = flags.save_dir + "/predicted_{}_{}".format(prefix, data_index)

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    out = trainer.calculate(
        sess, data_index=data_index, off_forecast=off_forecast)

    errors, inputs, enc_mus, enc_sigma_sqs, dec_outs, prior_mus, prior_sigma_sqs = out

    layer_size = len(errors)

    for i in range(layer_size):
        save_figure(
            errors[i],
            "predict_pp_kl{}.png".format(i),
            image_dir,
            ylabel="KLD",
            title="Posterior/Prior KLD: Layer{}".format(i))
        save_figure(
            enc_mus[i],
            "predict_enc_mu{}.png".format(i),
            image_dir,
            ylabel="Mean",
            title="Posterior mean: Layer{}".format(i))
        save_figure(
            enc_sigma_sqs[i],
            "predict_enc_sigma_sq{}.png".format(i),
            image_dir,
            ylabel="Variance",
            title="Posterior variance: Layer{}".format(i))
        save_figure(
            prior_mus[i],
            "predict_prior_mu{}.png".format(i),
            image_dir,
            ylabel="Mean",
            title="Prior mean: Layer{}".format(i))
        save_figure(
            prior_sigma_sqs[i],
            "predict_prior_sigma_sq{}.png".format(i),
            image_dir,
            ylabel="Variance",
            title="Prior variance: Layer{}".format(i))

        if i == 0:
            # 入力画像と再構成画像の表示
            dec_out_i = dec_outs[i]
            seq_length = inputs[0].shape[0]
            for j in range(seq_length):
                img_data = inputs[0][j, :, :]
                img_pred = dec_out_i[j, :, :]
                img = np.hstack([img_data, img_pred])
                imsave(image_dir + "/img{0:0>2}.png".format(j), img)

    # 連結したオリジナルを保存しておく
    org_concated = utils.concat_images(inputs[0])
    imsave(image_dir + "/org_concated_{0:0>2}.png".format(data_index),
           org_concated)


def predict(sess, trainer):
    predict_sub(sess, trainer, 0, True)


def predict_all(sess, trainer):
    predict_off_indices = [0, 1]
    predict_on_indices = [10, 11, 12, 13, 14]

    for index in predict_off_indices:
        predict_sub(sess, trainer, index, True)

    for index in predict_on_indices:
        predict_sub(sess, trainer, index, False)


def calc_graph_ranges(values):
    values = np.array(values)
    layer_size = values.shape[1]

    ranges = []

    for i in range(layer_size):
        # 時系列の0時刻目は値が大きくずれるのでレンジから外す
        layer_values = values[:, i, 1:]
        max_value = np.max(layer_values)
        min_value = np.min(layer_values)
        ranges.append((min_value, max_value))
    return ranges


def forecast_sub(sess, trainer, data_index):
    image_dir = flags.save_dir + "/forecast"

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    original_images, forecasted_images = trainer.forecast(
        sess, data_index=data_index)

    seq_length = len(original_images)

    # 画像を連結する
    original_concated_before = utils.concat_images(
        original_images[0:seq_length // 2])
    original_concated_after = utils.concat_images(
        original_images[seq_length // 2:])
    forecasted_concated_after = utils.concat_images(forecasted_images)

    imsave(image_dir + "/original_before{0:0>2}.png".format(data_index),
           original_concated_before)
    imsave(image_dir + "/original_after{0:0>2}.png".format(data_index),
           original_concated_after)
    imsave(image_dir + "/forecast_after{0:0>2}.png".format(data_index),
           forecasted_concated_after)


def forecast(sess, trainer):
    # 最初の10フレームを入れてその後10フレームをgenerateして予測する
    print("forecast data")
    forecast_sub(sess, trainer, 2)
    forecast_sub(sess, trainer, 3)


def visualize_weights(sess, model):
    """ Conv weightの可視化. """

    conv_w = model.get_conv_weight()
    w = sess.run(conv_w)

    # show graph of W_conv1
    plt.figure()

    filter_size = w.shape[3]

    fig, axes = plt.subplots(
        filter_size // 16,
        16,
        figsize=(12, 2),
        subplot_kw={
            'xticks': [],
            'yticks': []
        })
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for ax, i in zip(axes.flat, range(filter_size)):
        out_ch = i
        img = w[:, :, 0, out_ch]
        ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')

    file_path = flags.save_dir + "/weights.png"
    plt.savefig(file_path)


def evaluate_forecast(sess, trainer):
    """ 予測性能の定量評価を行う """

    print("evaluate forecast")

    rmss = []
    for i in range(trainer.data_manager.using_test_data_size):
        original_images, forecasted_images = trainer.forecast(
            sess, data_index=i)
        seq_length = len(original_images)

        for j in range(seq_length // 2):
            original = original_images[j + seq_length // 2]
            forecasted = forecasted_images[j]
            rms = np.sqrt(((forecasted - original)**2).mean())
            rmss.append(rms)

    result = np.mean(rmss)
    file_name = flags.save_dir + "/forecast_eval.txt"
    f = open(file_name, "w")
    f.write("rms={}".format(result))
    f.close()


def main(argv):
    if not os.path.exists(flags.save_dir):
        os.mkdir(flags.save_dir)

    dataset_type = flags.dataset_type
    layer_size = flags.layer_size
    h_size = flags.h_size
    latent_size = flags.latent_size
    batch_size = flags.batch_size
    beta = flags.beta
    cell_type = flags.cell_type
    binalize_output = dataset_type == "bsprite"
    downward_type = flags.downward_type
    no_td_bp = flags.no_td_bp
    filter_size = flags.filter_size

    data_manager = DataManager.get_data_manager(dataset_type)

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
        binalize_output=binalize_output,
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
        binalize_output=binalize_output,
        reuse=True)

    trainer = Trainer(data_manager, train_model, generate_model, predict_model,
                      flags.learning_rate, flags.use_denoising)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # For Tensorboard log
    log_dir = flags.save_dir + "/log"
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Load checkpoints
    saver, start_step = load_checkpoints(sess)

    if flags.training:
        # 学習
        train(sess, trainer, saver, summary_writer, start_step)
    else:
        # 生成
        generate(sess, trainer)

    # 予測誤差の確認
    predict_all(sess, trainer)

    # 最初の10フレームを入れた後に次の10フレームをgenerateする
    forecast(sess, trainer)

    # foracast性能の定量評価
    evaluate_forecast(sess, trainer)

    # weightの可視化
    visualize_weights(sess, train_model)


if __name__ == '__main__':
    tf.app.run()
