# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

PLOT_DIR = "plot"


def read_result_file(path):
    f = open(path)
    lines = f.readlines()  # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()

    ret = OrderedDict()

    for line in lines:
        line = line.strip()
        tokens = line.split("=")
        key = tokens[0]
        value_strs = (re.sub(r'\[|\]', "", tokens[1])).split(",")
        values = [float(value_str.strip()) for value_str in value_strs]
        ret[key] = values

    return ret


def save_graph(result, var_name, title, ylabel, sub_name):
    data0 = result["z_" + var_name]
    data1 = result["h_" + var_name]

    plt.figure()
    plt.ylim([0, 1])

    plt.title(title)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)

    xlabels = list(range(0, len(data0)))
    plt.xticks(xlabels, xlabels)

    plt.plot(data0, marker="o", linestyle="solid", label="z")
    plt.plot(data1, marker="o", linestyle="solid", label="h")
    plt.legend(loc='lower right', borderaxespad=1, fontsize=8)
    plt.title(title)
    plt.savefig(PLOT_DIR + "/plot_" + sub_name + "_" + var_name + ".png")
    plt.close()


def main():
    if not os.path.exists(PLOT_DIR):
        os.mkdir(PLOT_DIR)
    """
  path_mnist_3  = "saved_mnist_layer3/analysis/analysis_result.txt"
  path_mnist_3c = "saved_mnist_layer3_concat/analysis/analysis_result.txt"
  path_face_3   = "saved_face_layer3/analysis/analysis_result.txt"
  
  res_mnist_3 = read_result_file(path_mnist_3)
  save_graph(res_mnist_3, "pos_x", "Pos X", "Accuracy (R^2)", "m3_tdp")
  save_graph(res_mnist_3, "pos_y", "Pos Y", "Accuracy (R^2)", "m3_tdp")
  save_graph(res_mnist_3, "speed", "Speed", "Accuracy (R^2)", "m3_tdp")
  save_graph(res_mnist_3, "label", "Label", "Accuracy",       "m3_tdp")

  res_mnist_3c = read_result_file(path_mnist_3c)
  save_graph(res_mnist_3c, "pos_x", "Pos X", "Accuracy (R^2)", "m3_tdc")
  save_graph(res_mnist_3c, "pos_y", "Pos Y", "Accuracy (R^2)", "m3_tdc")
  save_graph(res_mnist_3c, "speed", "Speed", "Accuracy (R^2)", "m3_tdc")
  save_graph(res_mnist_3c, "label", "Label", "Accuracy",       "m3_tdc")

  res_face_3 = read_result_file(path_face_3)
  save_graph(res_face_3, "pan",        "Pan",        "Accuracy (R^2)", "f3_tdp")
  save_graph(res_face_3, "roll",       "Roll",       "Accuracy (R^2)", "f3_tdp")
  save_graph(res_face_3, "pan_alpha",  "Pan Alpha",  "Accuracy (R^2)", "f3_tdp")
  save_graph(res_face_3, "roll_alpha", "Roll Alpha", "Accuracy (R^2)", "f3_tdp")
  """
    """
  path_bsprite_3  = "saved_bsprite_layer3_f16/analysis/analysis_result.txt"
  res_bsprite_3 = read_result_file(path_bsprite_3)
  save_graph(res_bsprite_3, "pos_x",        "Pos X",     "Accuracy (R^2)", "b3_tdp")
  save_graph(res_bsprite_3, "pos_y",        "Pos Y",     "Accuracy (R^2)", "b3_tdp")
  save_graph(res_bsprite_3, "label",        "Label",     "Accuracy (R^2)", "b3_tdp")
  save_graph(res_bsprite_3, "speed",        "Speed",     "Accuracy",       "b3_tdp")
  save_graph(res_bsprite_3, "bounce_x",     "Bounce X",  "Accuracy",       "b3_tdp")
  save_graph(res_bsprite_3, "bounce_y",     "Bounce Y",  "Accuracy",       "b3_tdp")
  """

    path_bsprite_3c = "saved_bsprite_layer3_f16_concat/analysis/analysis_result.txt"
    res_bsprite_3c = read_result_file(path_bsprite_3c)
    save_graph(res_bsprite_3c, "pos_x", "Pos X", "Accuracy (R^2)", "b3_tdc")
    save_graph(res_bsprite_3c, "pos_y", "Pos Y", "Accuracy (R^2)", "b3_tdc")
    save_graph(res_bsprite_3c, "label", "Label", "Accuracy (R^2)", "b3_tdc")
    save_graph(res_bsprite_3c, "speed", "Speed", "Accuracy", "b3_tdc")
    save_graph(res_bsprite_3c, "bounce_x", "Bounce X", "Accuracy", "b3_tdc")
    save_graph(res_bsprite_3c, "bounce_y", "Bounce Y", "Accuracy", "b3_tdc")


if __name__ == '__main__':
    main()
