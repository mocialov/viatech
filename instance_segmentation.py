import numpy as np
from PIL import Image

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import gridspec
from matplotlib import pyplot as plt
import tarfile
import os
import time
import cv2


class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    '''self.graph = tf.Graph()
    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
        break
    tar_file.close()
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')
    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    self.sess = tf.compat.v1.Session(graph=self.graph)'''


    #####

    self.sess = tf.compat.v1.Session()
    if True: #with tf.compat.v1.Session() as sess:
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
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    print(seg_map.shape, resized_image.size)
    seg_map = cv2.resize(seg_map, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
    resized_image = resized_image.resize((width, height), Image.ANTIALIAS)
    print(seg_map.shape, resized_image.size)
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

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape

def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  a_figure = plt.figure(figsize=(60, 30))
  grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 1])

  seg_image = label_to_color_image(seg_map).astype(np.uint8)

  plt.subplot(grid_spec[0])
  #plt.imshow(image)
  #plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[1])
  #plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels], fontsize=30)
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  #plt.show()
  return fig2img ( a_figure )


def process_image_detectron2(original_im):
  # inferences DeepLab model
  start_time = time.time()
  resized_im, seg_map = model.run(original_im)
  ellapsed_time = time.time() - start_time
  print("Ellapsed time: " + str(ellapsed_time) + "s")

  # show inference result
  return vis_segmentation(resized_im, seg_map)

LABEL_NAMES = np.asarray([
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bycycle'])


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

model_path = "/home/ubuntu/flaskapp/train_fine/frozen_inference_graph.pb"

# load model
model = DeepLabModel(model_path)


