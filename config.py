#https://github.molgen.mpg.de/mohomran/cityscapes/blob/master/scripts/helpers/labels.py
KNOWN_CLASSES_CITYSCAPES={
'road':0,
'sidewalk':1,
'building':2,
'wall':3,
'fence':4,
'pole':5,
'traffic_light':6,
'traffic_sign':7,
'vegetation':8,
'terrain':9,
'sky':10,
'person':11,
'rider':12,
'car':13,
'truck':14,
'bus':15,
'train':16,
'motorcycle':17,
'bicycle':18,
'misc':255
}

#https://code.ihub.org.cn/projects/358/repository/revisions/master/entry/mmdet/datasets/coco.py
KNOWN_CLASSES_DETECTRON2 = {
'person':0,
'car':2,
'bus':5,
'truck':7,
}

KNOWN_CLASSES_CRACK_DETECTOR = {
'crack': 1,
'manhole': 2,
'patch': 3
}

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

SECRET_KEY = "\xbe\xc1\xd5\x16\x03\xf87\xdd\xf5\xe7y\x03\xdf1\xac\xca2\x00\xee\x1d\xdc\xc2M+"

PYTORCH_DOWNLOADS_PATH = "/home/vts/.torch/fvcore_cache/"

CITYSCAPES_MODEL_PATH = "/home/vts/flaskapp/train_fine/frozen_inference_graph.pb"

DETECTRON2_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

MODEL_INPUT_SIZE = 513

PASSWORD = 'y5AHfLgaGr56R&tK'
USERNAME = 'admin'
