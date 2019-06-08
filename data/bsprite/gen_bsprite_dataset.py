# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# moving mnistデータセットの生成スクリプト

from PIL import Image, ImageDraw
import sys
import os
import math
import numpy as np
import gzip


def get_sprite_images(image_sz):
  # 円
  image = Image.new('L', (image_sz, image_sz), (0))
  draw = ImageDraw.Draw(image)
  draw.ellipse((0, 0, image_sz-1, image_sz-1), fill=(255))
  data_circle = np.asarray(image)

  # 正方形
  image = Image.new('L', (image_sz, image_sz), (0))
  draw = ImageDraw.Draw(image)
  draw.polygon(((0, 0),
                (0, image_sz-1),
                (image_sz-1, image_sz-1),
                (image_sz-1, 0)), fill=(255))
  data_rect = np.asarray(image)

  # 三角形
  image = Image.new('L', (image_sz, image_sz), (0))
  draw = ImageDraw.Draw(image)
  draw.polygon((((image_sz-1)//2, 0),
                (image_sz-1, image_sz-1),
                (0, image_sz-1)), fill=(255))
  data_triangle = np.asarray(image)

  return (data_circle, data_rect, data_triangle)


def paste_image(canvas, image, position):
  px = int(round(position[0]))
  py = int(round(position[1]))

  canvas_w = canvas.shape[0]
  canvas_h = canvas.shape[1]

  image_w = image.shape[0]
  image_h = image.shape[1]

  for tx in range(image_w):
    for ty in range(image_h):
      cx = px + tx
      cy = py + ty
      if cx < 0 or cx >= canvas_w or cy < 0 or cy >= canvas_h:
        continue
      # ここがx,y逆になるので注意
      canvas[cy,cx] = image[ty,tx]
  
  
def generate_sequence(seq_length,
                      width,
                      height,
                      lims,
                      initial_veloc,
                      initial_position,
                      image):
  """ 1シーケンス分の画像とpositionを得る """
  
  ret_images    = np.zeros((seq_length, height, width), dtype=np.uint8)
  ret_positions = np.zeros((seq_length, 2),             dtype=np.float32)
  ret_bounce_xs = np.zeros((seq_length),                dtype=np.uint8)
  ret_bounce_ys = np.zeros((seq_length),                dtype=np.uint8)

  veloc    = np.copy(initial_veloc)
  position = np.copy(initial_position)
  
  for frame_idx in range(seq_length):
    # 最終的なキャンバス
    canvas = np.zeros((height, width), dtype=np.uint8)

    paste_image(canvas, image, position)
    ret_positions[frame_idx] = position

    # update positions based on velocity
    next_pos = position + veloc
    
    # 壁での跳ね返り
    next_x = next_pos[0]
    vx = veloc[0]
    if next_x < -2.0:
      vx = -vx
      ret_bounce_xs[frame_idx] = -1
    elif next_x > lims[0] + 2.0:
      vx = -vx
      ret_bounce_xs[frame_idx] = 1
    else:
      ret_bounce_xs[frame_idx] = 0
    
    next_y = next_pos[1]
    vy = veloc[1]
    if next_y < -2.0:
      vy = -vy
      ret_bounce_ys[frame_idx] = -1
    elif next_y > lims[1] + 2.0:
      vy = -vy
      ret_bounce_ys[frame_idx] = 1
    else:
      ret_bounce_ys[frame_idx] = 0
    
    veloc = (vx, vy)
    
    # 位置更新
    position = position + veloc
    
    ret_images[frame_idx] = canvas

  # bounceを次フレームに移動する (バウンスした次フレームの検知とする場合)
  for i in list(reversed(range(seq_length))):
    if i > 0:
      ret_bounce_xs[i] = ret_bounce_xs[i-1]
      ret_bounce_ys[i] = ret_bounce_ys[i-1]
    else:
      ret_bounce_xs[i] = 0
      ret_bounce_ys[i] = 0
    
  return ret_images, ret_positions, ret_bounce_xs, ret_bounce_ys


# generates and returns video frames in uint8 array
def generate_data():
  shape      = (64,64)
  seq_length = 20
  data_size  = 14000
  image_sz   = 11

  sprite_images = get_sprite_images(image_sz)
  # ([11, 11], [11, 11], [11, 11]), uint8

  width, height = shape
  x_lim = width - image_sz
  y_lim = height - image_sz
  lims = (x_lim, y_lim)

  # 最終的な書き出すデータセット
  dataset_images    = np.empty((data_size, seq_length, height, width), dtype=np.uint8)
  dataset_labels    = np.empty((data_size),                            dtype=np.uint8)
  dataset_pos_xs    = np.empty((data_size, seq_length),                dtype=np.float32)
  dataset_pos_ys    = np.empty((data_size, seq_length),                dtype=np.float32)
  dataset_speeds    = np.empty((data_size),                            dtype=np.uint8)
  dataset_bounce_xs = np.empty((data_size, seq_length),                dtype=np.int8)
  dataset_bounce_ys = np.empty((data_size, seq_length),                dtype=np.int8)

  # 通常データ生成
  for seq_idx in range(data_size):
    if seq_idx % 500 == 0:
      print("process seq: {}".format(seq_idx))
    
    # randomly generate direc/speed/position, calculate velocity vector
    # 方向 (-pi ~ pi)
    direc = np.pi * (np.random.rand() * 2.0 - 1.0)
    # スピード
    speed = np.random.randint(5) + 2

    dataset_speeds[seq_idx] = speed

    # 速度ベクトル
    initial_veloc = (speed * math.cos(direc), speed * math.sin(direc))

    # 現在位置 (各数字に対して(x,y))
    initial_position = (np.random.rand() * x_lim, np.random.rand() * y_lim)
    
    label = seq_idx % len(sprite_images)
    image = sprite_images[label]
    
    dataset_labels[seq_idx] = label

    out = generate_sequence(seq_length, 
                            width,
                            height,
                            lims,
                            initial_veloc,
                            initial_position,
                            image)
    ret_images, ret_positions, ret_bounce_xs, ret_bounce_ys = out
                                                            
    dataset_images[seq_idx]    = ret_images
    dataset_pos_xs[seq_idx]    = ret_positions[:,0]
    dataset_pos_ys[seq_idx]    = ret_positions[:,1]
    dataset_bounce_xs[seq_idx] = ret_bounce_xs
    dataset_bounce_ys[seq_idx] = ret_bounce_ys
    
  return dataset_images, dataset_labels, dataset_pos_xs, dataset_pos_ys, dataset_speeds, \
    dataset_bounce_xs, dataset_bounce_ys


def main():
  # 乱数のシードを固定
  np.random.seed(seed=1)
  
  out = generate_data()
  
  images, labels, pos_xs, pos_ys, speeds, bounce_xs, bounce_ys = out
  # uint8, uint8, float32, float32, uint8, int8, int8

  file_name = "bsprite"

  # .npzを省いたパス
  file_path = os.path.join(".", file_name)

  # 圧縮して保存
  np.savez_compressed(file_path,
                      images=images,
                      labels=labels,
                      pos_xs=pos_xs,
                      pos_ys=pos_ys,
                      speeds=speeds,
                      bounce_xs=bounce_xs,
                      bounce_ys=bounce_ys)


if __name__ == '__main__':
  main()
