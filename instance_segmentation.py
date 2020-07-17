import config

import numpy as np
from PIL import Image

import tensorflow as tf
import tarfile
import os
import time
import cv2
import PIL
from skimage.transform import resize, rescale

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import gridspec
from matplotlib import pyplot as plt



class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    self.sess = tf.compat.v1.Session()
    with tf.io.gfile.GFile(tarball_path,'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    self.sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')    


  def run(self, image):
    """Runs inference on a single image.
    Args:
      image: A PIL.Image object, raw input image.
    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    
    width, height = image.size
    resize_ratio = 1.0 * config.MODEL_INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    
    return resized_image, seg_map


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [0, 0, 0]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap

def label_to_color_image(label):

  colormap = create_cityscapes_label_colormap()
  return colormap[label]


if True:
    LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bycycle'])

    FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
    FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""

  plt.figure(figsize=(60, 30))
  grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

  seg_image = label_to_color_image(seg_map).astype(np.uint8)

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[1])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels], fontsize=30)
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  #plt.show()
  plt.draw()
  import io
  buf = io.BytesIO()
  plt.savefig(buf, format='jpg')
  buf.seek(0)
  im = Image.open(buf)
  plt.draw()
  return im


def process_image_detectron2(original_im, model):
  start_time = time.time()
  resized_im, seg_map = model.run(original_im)
  ellapsed_time = time.time() - start_time
  print("Ellapsed time: " + str(ellapsed_time) + "s")

  seg_map2 = resize(seg_map, (original_im.size[1], original_im.size[0]), preserve_range=True, order=1)
  
  temp = vis_segmentation(resized_im, seg_map)

  return seg_map2, temp


def load_detectron_model():
    model_path = config.CITYSCAPES_MODEL_PATH 

    # load model
    model = DeepLabModel(model_path)

    return model


