import torch, torchvision
from PIL import ImageFilter
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#predictor = None

def load_model():
    #global predictor

    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE='cpu'
    predictor = DefaultPredictor(cfg)
    return predictor


from os import walk
from PIL import Image
import time
from threading import Thread
import time
import piexif
import PIL.Image

threads = 1 #process n images at a time with one image per thread
sleep_time = 3 #3 seconds to wait for threads to finish (could be logarithmic)
keep_classes = [0, 2] #0-person, 2-car
#all categories of COCO dataset https://cocodataset.org/#explore

thread_finished = [False] * threads #when thread finishes, it assigns True to its location in an array

#chunking list into m chunks of <=n size
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#detectron2 prediction and blurring mask
def predict(path, predictor):
    print("predicting")

    img = PIL.Image.open(path)

    #extract metadata
    exif_dict = piexif.load(img.info['exif'])

    im = np.asarray(img)
    #im = cv2.resize(im, (400, 200))

    start = time.time()
    outputs = predictor(im) #identify and predict objects
    elapsed = time.time() - start
    print ("prediction time in seconds: %02d" % (elapsed)) #about 9 seconds per image

    print ("blurring")
    out = im.copy()
    print ("actual blur")
    #blur = cv2.blur(im,(30,30),0) #(30,30) blurring kernel
    #blur = im
    blur = PIL.Image.fromarray(im).filter(ImageFilter.GaussianBlur(50))
    blur = np.array(blur)

    print ("filtering masks")
    keep_masks = []
    all_masks = np.array(torch.Tensor.cpu(outputs["instances"].pred_masks).detach().numpy(), dtype=np.uint8) #all masks
    #keep only wanted labels
    for idx, detected_class in enumerate(torch.Tensor.cpu(outputs["instances"].pred_classes).detach().numpy()):
        print ("mask", str(idx))
        if detected_class in keep_classes:
            keep_masks.append(all_masks[idx])

    print ("applying filtered masks")
    #apply masks
    for a_mask in keep_masks:
        out[a_mask>0] = blur[a_mask>0]

    #show image
    #display(Image.fromarray(out))

    #dump metadata
    print("adding exif")
    exif_bytes = piexif.dump(exif_dict)

    print ("saving")
    #save image
    PIL.Image.fromarray(out).save(path)#, exif=exif_bytes)

    print("done")
    #thread_finished[thread_id] = True

    return True

