# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# moving mnistデータセットの生成スクリプト

import numpy as np
import os

from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt


output_dir = "movement"


def save_movement_graph(test_data_index,
                        pos_xs, pos_ys,
                        bounce_xs, bounce_ys):
  
  seq_index = test_data_index + 9000
  pos_x    = pos_xs[seq_index]
  pos_y    = pos_ys[seq_index]
  bounce_x = bounce_xs[seq_index]
  bounce_y = bounce_ys[seq_index]

  bounce_x = bounce_x * 16 + 32
  bounce_y = bounce_y * 16 + 32

  plt.figure()
  plt.ylim([0-5, 64+5])

  plt.plot(pos_x, label="x")
  plt.plot(pos_y, label="y")
  plt.plot(bounce_x, linestyle="dashed", label="bx")
  plt.plot(bounce_y, linestyle="dashed", label="by")
  
  # レジェンドの表示
  plt.legend(bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0, fontsize=8)

  plt.title("Movement")
  plt.xlabel("Timestep")
  
  xlocs   = list(range(0, 20, 2))
  xlabels = list(range(0, 20, 2))
  plt.xticks(xlocs, xlabels)

  file_path = output_dir + "/move_{0:03}.png".format(test_data_index)
  plt.savefig(file_path)
  
  plt.close()
  
  

def main():
  data_path = "./bsprite.npz"
  data_all = np.load(data_path)

  data_images    = data_all["images"]    # (14000, 20, 64, 64) uint8
  data_pos_xs    = data_all["pos_xs"]    # (14000, 20)         float32
  data_pos_ys    = data_all["pos_ys"]    # (14000, 20)         float32
  data_bounce_xs = data_all["bounce_xs"] # (14000, 20)         int8
  data_bounce_ys = data_all["bounce_ys"] # (14000, 20)         int8


  if not os.path.exists(output_dir):
    os.mkdir(output_dir) 

  save_movement_graph(10, data_pos_xs, data_pos_ys, data_bounce_xs, data_bounce_ys)
  save_movement_graph(11, data_pos_xs, data_pos_ys, data_bounce_xs, data_bounce_ys)
  save_movement_graph(12, data_pos_xs, data_pos_ys, data_bounce_xs, data_bounce_ys)
  save_movement_graph(13, data_pos_xs, data_pos_ys, data_bounce_xs, data_bounce_ys)
  save_movement_graph(14, data_pos_xs, data_pos_ys, data_bounce_xs, data_bounce_ys)
  save_movement_graph(15, data_pos_xs, data_pos_ys, data_bounce_xs, data_bounce_ys)  


if __name__ == '__main__':
  main()
