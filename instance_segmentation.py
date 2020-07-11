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

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
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
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    
    return resized_image, seg_map


def process_image_detectron2(original_im, model):
  start_time = time.time()
  resized_im, seg_map = model.run(original_im)
  ellapsed_time = time.time() - start_time
  print("Ellapsed time: " + str(ellapsed_time) + "s")

  seg_map2 = resize(seg_map, (original_im.size[1], original_im.size[0]), preserve_range=True, order=1)
  
  return seg_map2


def load_detectron_model():
    model_path = config.CITYSCAPES_MODEL_PATH 

    # load model
    model = DeepLabModel(model_path)

    return model


