import numpy as np
import cv2
import random

# import some common detectron2 utilities
from os import walk
from PIL import Image
import time
from threading import Thread
import time
import piexif
import PIL.Image

import tensorflow as tf

import config

np.set_printoptions(suppress=True)

#PATH_TO_SAVED_MODEL = "models/research/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/training2/saved_model"
#PATH_TO_SAVED_MODEL = "models/research/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8/training2/saved_model"
#PATH_TO_SAVED_MODEL = "models/research/ssd_resnet152_fpn_640x640_learning_rate_0_00035_f1_26/training2/saved_model"
PATH_TO_SAVED_MODEL = "models/research/ssd_resnet152_v1_fpn_640x640_coco17_tpu-8_constLR0.000035_map_0.12_f1_19/training2/saved_model"

def load_crack_detector():
    global crack_detector

    print("loading crack detector model")

    crack_detector = tf.compat.v1.saved_model.load_v2(PATH_TO_SAVED_MODEL)

    return crack_detector

def predict_crack(image_np, crack_detector, initial_image_shape, threshold):

    #image_np = np.array(im)

    input_tensor = tf.convert_to_tensor(image_np)

    input_tensor = input_tensor[tf.newaxis, ...]

    detections = crack_detector(input_tensor)

    #print ("crack detection", detections, flush=True)

    scores = detections["detection_scores"].numpy()[0]
    boxes = detections["detection_boxes"].numpy()[0]

    boxes[:,0] *= 640
    boxes[:,1] *= 640
    boxes[:,2] *= 640
    boxes[:,3] *= 640

    boxes[:,0] *= (initial_image_shape[0]/640)
    boxes[:,1] *= (initial_image_shape[1]/640)
    boxes[:,2] *= (initial_image_shape[0]/640)
    boxes[:,3] *= (initial_image_shape[1]/640)


    #boxes[:,0] = boxes[:,0].astype(int)
    #boxes[:,1] = boxes[:,1].astype(int)
    #boxes[:,2] = boxes[:,2].astype(int)
    #boxes[:,3] = boxes[:,3].astype(int)
    
    #print (boxes, flush=True)


    classes = detections["detection_classes"].numpy()[0]
    classes = [{ v:k for k,v in config.KNOWN_CLASSES_CRACK_DETECTOR.items()}[item] for item in classes]
    #print (scores.shape, boxes.shape, flush=True)

    result = np.column_stack((boxes, scores))
    result = np.column_stack((result, classes))

    result = result[result[:, 4].astype(float) >= threshold]

    #result[:,0:2] = np.int16(result[:,0:2])
    #print(result, flush=True)
    #print (result.shape, flush=True)
    #print (list(map(str,result.flatten())), flush=True)
    #print (result, flush=True)
    result = [str(int(float(item))) if (idx%((idx//6)*6+4)!=0 and idx%((idx//6)*6+5)!=0) or idx==0 else str(item) for idx, item in enumerate(result.flatten())]
    #print(list(map(str,result.flatten())), flush=True)
    #return ','.join(list(map(str,result.flatten())))
    return ','.join(result)

#if __name__ == '__main__':
#    load_model()
