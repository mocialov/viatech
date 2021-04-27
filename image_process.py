import config
import torch, torchvision
from PIL import ImageFilter
import detectron2
#from detectron2.utils.logger import setup_logger
#setup_logger()

import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

from os import walk
from PIL import Image
import time
from threading import Thread
import time
import piexif
import PIL.Image

out = None

def load_model():
    print("loading model")
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file(config.DETECTRON2_MODEL ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config.DETECTRON2_MODEL )
    cfg.MODEL.DEVICE='cpu'
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    predictor = DefaultPredictor(cfg)
    return predictor


#detectron2 prediction and blurring mask
def predict(img, predictor, metadata, blurring_parameter=4):
    objects_blurred = 0
    im = np.asarray(img)

    start = time.time()
    outputs = predictor(im) #identify and predict objects
    elapsed = time.time() - start
    print ("prediction time in seconds: %02d" % (elapsed)) #about 9 seconds per image

    out = im.copy()
    blur = PIL.Image.fromarray(im).filter(ImageFilter.GaussianBlur(blurring_parameter))
    blur = np.array(blur)

    keep_masks = []
    all_masks = np.array(torch.Tensor.cpu(outputs["instances"].pred_masks).detach().numpy(), dtype=np.uint8) #all masks
    #keep only wanted labels
    for idx, detected_class in enumerate(torch.Tensor.cpu(outputs["instances"].pred_classes).detach().numpy()):
        if detected_class in list(config.KNOWN_CLASSES_DETECTRON2.values()):
            keep_masks.append(all_masks[idx])
            objects_blurred += 1

    #apply masks
    for a_mask in keep_masks:
        out[a_mask>0] = blur[a_mask>0]

    return PIL.Image.fromarray(out), objects_blurred

#if __name__ == '__main__':
#    load_model()
