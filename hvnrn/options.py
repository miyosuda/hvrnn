# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime as dt

# 無視するflagsのキー
ignore_keys = set(["h", "help", "helpfull", "helpshort"])


def get_options():
    tf.app.flags.DEFINE_string("dataset_type", "mnist",
                               "dataset type. face, mnist or bsprite")
    tf.app.flags.DEFINE_string("save_dir", "saved",
                               "checkpoints,log,options save directory")
    tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
    tf.app.flags.DEFINE_float("beta", 1.0, "vae beta parameter")
    tf.app.flags.DEFINE_string("cell_type", "vrnn",
                               "RNN cell type. 'vrnn' or 'merlin'")
    tf.app.flags.DEFINE_integer("steps", 15 * (10**5), "training steps")
    tf.app.flags.DEFINE_integer("save_interval", 5000, "saving interval")
    tf.app.flags.DEFINE_integer("test_interval", 100000, "test interval")
    tf.app.flags.DEFINE_integer("generate_interval", 5000, "generate interval")
    tf.app.flags.DEFINE_integer("predict_interval", 10000, "predict interval")
    tf.app.flags.DEFINE_integer("h_size", 256, "vrnn hidden size")
    tf.app.flags.DEFINE_integer("latent_size", 16, "vrnn latent size")
    tf.app.flags.DEFINE_integer("batch_size", 10, "batch size")
    tf.app.flags.DEFINE_boolean("training", True, "whether to train or not")
    tf.app.flags.DEFINE_boolean(
        "no_td_bp", False, "whether to stop back-prop for top-down signal.")
    tf.app.flags.DEFINE_boolean("use_denoising", False,
                                "whether to use denoising or not'")
    tf.app.flags.DEFINE_string(
        "downward_type", "to_prior",
        "downward stream type. 'add', 'gated_add', 'concat', 'to_prior'")
    tf.app.flags.DEFINE_integer("filter_size", 64, "conv filter size.")
    tf.app.flags.DEFINE_integer("layer_size", 3, "hierarchy layer size")
    tf.app.flags.DEFINE_string("desc", "hvrnn experiment", "experiment description")

    # analyze/decodeのみで利用
    tf.app.flags.DEFINE_boolean("collect_data", True,
                                "whether to collect analysis/decode data")

    return tf.app.flags.FLAGS


def save_flags(flags):
    dic = flags.__flags

    lines = []

    # 現在時刻の記録
    time_str = dt.now().strftime('# %Y-%m-%d %H:%M')
    lines.append(time_str + "\n")

    for key in sorted(dic.keys()):
        if key in ignore_keys:
            # "helpfull"などのキーは無視しないといけない
            continue

        if hasattr(dic[key], "value"):
            # for TF 1.5+
            value = dic[key].value
        else:
            # for ~ TF 1.4
            value = dic[key]
        line = "{}={}".format(key, value)
        lines.append(line + "\n")

    file_name = flags.save_dir + "/options.txt"
    f = open(file_name, "w")
    f.writelines(lines)
    f.close()
