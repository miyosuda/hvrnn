# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# moving mnistデータセットの生成スクリプト

from PIL import Image
import sys
import os
import math
import numpy as np
import gzip

if sys.version_info[0] == 2:
  from urllib import urlretrieve
else:
  from urllib.request import urlretrieve

DATA_DIR = "./base_data"

# helper functions
def img_to_array(im):
  w, h = im.size # 28, 28
  arr = im.getdata()
  c = np.product(arr.size) // (w * h)
  return np.asarray(arr, dtype=np.float32).reshape((h, w)) / 255.0

def get_picture_array(data, index):
  """ 画像をuint8にして取得 """
  ret = (data[index] * 255.0).clip(0.0, 255.0).astype(np.uint8)
  return ret


# loads mnist from web on demand
def load_dataset():
  def download(file_name, file_path, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % file_name)
    if not os.path.exists(DATA_DIR):
      os.mkdir(DATA_DIR)
    urlretrieve(source + file_name, file_path)
    
  def load_sub(file_name, offset):
    file_path = os.path.join(DATA_DIR, file_name)    
    if not os.path.exists(file_path):
      download(file_name, file_path)
    with gzip.open(file_path, 'rb') as f:
      data = np.frombuffer(f.read(), np.uint8, offset=offset)
    return data
  
  def load_mnist_images(file_name):
    data = load_sub(file_name, offset=16)
    # data = (47040000,)
    data = data.reshape(-1, 28, 28)
    return data / np.float32(255)

  def load_mnist_labels(file_name):
    data = load_sub(file_name, offset=8)
    # data = (60000,)
    return data

  images = load_mnist_images('train-images-idx3-ubyte.gz')
  labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
  return (images, labels)


def generate_sequence(seq_length,
                      nums_per_image,
                      width,
                      height,
                      lims,
                      initial_veloc,
                      initial_positions,
                      mnist_images):
  """ 1シーケンス分の画像とpositionを得る """
  ret_images    = np.empty((seq_length, width, height),     dtype=np.uint8)
  ret_positions = np.empty((seq_length, nums_per_image, 2), dtype=np.float32)
  ret_bounces   = np.empty((seq_length, nums_per_image, 2), dtype=np.int8)

  veloc     = np.copy(initial_veloc)
  positions = np.copy(initial_positions)

  for frame_idx in range(seq_length):
    # 各数字に対してのキャンバス
    # 最終的なキャンバス
    canvas = np.zeros((1, width, height), dtype=np.float32)
    
    for digit_idx in range(nums_per_image):
      canv = Image.new('L', (width, height))
      canv.paste(mnist_images[digit_idx],
                 tuple(map(lambda p: int(round(p)), positions[digit_idx])))
      canvas += img_to_array(canv)
    # 位置をセット
    ret_positions[frame_idx] = positions

    # 次の位置を算出
    next_pos = [list(map(sum, zip(p,v))) for p,v in zip(positions, veloc)]

    # bounce off wall if a we hit one
    for digit_idx, pos in enumerate(next_pos):
      # 各文字に関して
      x = pos[0]
      if x < -2.0:
        ret_bounces[frame_idx][digit_idx][0] = -1
        veloc[digit_idx][0] = -veloc[digit_idx][0]
      elif x > lims[0] + 2.0:
        ret_bounces[frame_idx][digit_idx][0] = 1
        veloc[digit_idx][0] = -veloc[digit_idx][0]
      else:
        ret_bounces[frame_idx][digit_idx][0] = 0
        
      y = pos[1]
      if y < -2.0:
        ret_bounces[frame_idx][digit_idx][1] = -1
        veloc[digit_idx][1] = -veloc[digit_idx][1]
      elif y > lims[1] + 2.0:
        ret_bounces[frame_idx][digit_idx][1] = 1
        veloc[digit_idx][1] = -veloc[digit_idx][1]
      else:
        ret_bounces[frame_idx][digit_idx][1] = 0

    # 位置更新
    positions = [list(map(sum, zip(p,v))) for p,v in zip(positions, veloc)]
    # copy additive canvas to data array
    ret_images[frame_idx] = (canvas * 255).clip(0,255).astype(np.uint8)

  # bounceを次フレームに移動する (バウンスした次フレームの検知とする場合)
  for frame_idx in list(reversed(range(seq_length))):
    for digit_idx in range(nums_per_image):
      for i in range(2):
        if frame_idx > 0:
          ret_bounces[frame_idx][digit_idx][i] = ret_bounces[frame_idx-1][digit_idx][i]
        else:
          ret_bounces[frame_idx][digit_idx][i] = 0

  return ret_images, ret_positions, ret_bounces


def get_images_and_labels(indices, images, labels, image_sz):
  mnist_images = []
  mnist_labels = []
  
  for index in indices:
    img_arr = get_picture_array(images, index)
    label = labels[index]
    img = Image.fromarray(img_arr)
    img = img.resize((image_sz, image_sz), Image.ANTIALIAS)
    
    mnist_images.append(img)
    mnist_labels.append(label)
  return mnist_images, mnist_labels


def find_first_label_index(labels, target_label):
  for i in range(len(labels)):
    if labels[i] == target_label:
      return i
  return -1

# generates and returns video frames in uint8 array
def generate_data(shape=(64,64),
                  seq_length=20,
                  data_size=14000,
                  image_sz=28,
                  nums_per_image=2): # 1画像に数字を何個入れるか
  
  images, labels = load_dataset()
  # (60000, 28, 28), (60000)

  width, height = shape
  x_lim = width - image_sz
  y_lim = height - image_sz
  lims = (x_lim, y_lim)

  # 最終的な書き出すデータセット
  # (Originalのmoving mnist datasetにshapeを合わしていない)
  dataset_images    = np.empty((data_size, seq_length, width, height),  dtype=np.uint8)
  dataset_labels    = np.empty((data_size, nums_per_image),             dtype=np.uint8)
  dataset_pos_xs    = np.empty((data_size, seq_length, nums_per_image), dtype=np.float32)
  dataset_pos_ys    = np.empty((data_size, seq_length, nums_per_image), dtype=np.float32)
  dataset_bounce_xs = np.empty((data_size, seq_length, nums_per_image), dtype=np.int8)
  dataset_bounce_ys = np.empty((data_size, seq_length, nums_per_image), dtype=np.int8)
  dataset_speeds    = np.empty((data_size, nums_per_image),             dtype=np.uint8)

  # 通常データ生成
  for seq_idx in range(data_size):
    if seq_idx % 500 == 0:
      print("process seq: {}".format(seq_idx))
    
    # randomly generate direc/speed/position, calculate velocity vector
    # 方向 (-pi ~ pi) (各数字に関して)
    direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
    # スピード (各数字に関して)
    speeds = np.random.randint(5, size=nums_per_image) + 2

    dataset_speeds[seq_idx] = speeds

    # 速度ベクトル
    veloc = [(v*math.cos(d), v*math.sin(d)) for d,v in zip(direcs, speeds)]

    # 現在位置 (各数字に対して(x,y))
    positions = [(np.random.rand()*x_lim, np.random.rand()*y_lim) for _ in range(nums_per_image)]

    # 1画像内の数字の数だけ集める
    indices = np.random.randint(0, images.shape[0], nums_per_image)
    mnist_images, mnist_labels = get_images_and_labels(indices, images, labels, image_sz)
    
    dataset_labels[seq_idx] = mnist_labels
    
    out = generate_sequence(seq_length, nums_per_image, 
                            width, height, lims,
                            veloc, positions, mnist_images)
    
    ret_images, ret_positions, ret_bounces = out
    
    dataset_images[seq_idx]    = ret_images
    dataset_pos_xs[seq_idx]    = ret_positions[:,:,0]
    dataset_pos_ys[seq_idx]    = ret_positions[:,:,1]
    dataset_bounce_xs[seq_idx] = ret_bounces[:,:,0]
    dataset_bounce_ys[seq_idx] = ret_bounces[:,:,1]
    
  return dataset_images, dataset_labels, dataset_pos_xs, dataset_pos_ys, dataset_speeds, \
    dataset_bounce_xs, dataset_bounce_ys


def main(nums_per_image=1,
         file_name="moving_mnist1",
         frame_size=64,
         seq_length=20,
         data_size=14000,
         image_sz=28):

  # 乱数のシードを固定
  np.random.seed(seed=1)
  
  out = generate_data(shape=(frame_size,frame_size),
                      seq_length=seq_length,
                      data_size=data_size,
                      image_sz=image_sz,
                      nums_per_image=nums_per_image)
  
  dat_images, dat_labels, dat_pos_xs, dat_pos_ys, dat_speeds, dat_bounce_xs, dat_bounce_ys = out
  # uint8, uint8, float32, float32, uint8, int8, int8

  # .npzを省いたパス
  file_path = os.path.join(".", file_name)

  # 圧縮して保存
  np.savez_compressed(file_path,
                      images=dat_images,
                      labels=dat_labels,
                      pos_xs=dat_pos_xs,
                      pos_ys=dat_pos_ys,
                      speeds=dat_speeds,
                      bounce_xs=dat_bounce_xs,
                      bounce_ys=dat_bounce_ys)
  

if __name__ == '__main__':
  nums_per_image = 1
  file_name = "moving_mnist{}".format(nums_per_image)
  
  main(nums_per_image=nums_per_image,
       file_name=file_name)
