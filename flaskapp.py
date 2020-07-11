import config
import os
os.environ['FVCORE_CACHE'] = config.PYTORCH_DOWNLOADS_PATH

import io
from flask import send_file
from flask import Flask, flash, request, redirect, url_for, jsonify, json, make_response
from werkzeug.utils import secure_filename
from image_process import *
import numpy as np
import cv2
from PIL import Image
import piexif
from PIL.ExifTags import TAGS
import base64
from instance_segmentation import *

app = Flask(__name__)
app.secret_key = config.SECRET_KEY


predictor = None
predictor_detectron = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS


@app.route('/segment', methods=['POST'])
def upload_file2():
    global predictor_detectron

    if predictor_detectron == None:
        predictor_detectron = load_detectron_model()
    
    requested_instances = request.args.get('instances', default = None, type = str)
    if requested_instances != None:
        requested_instances = requested_instances.split(",")
    print ("requested instances", requested_instances)

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    print ("reading file")
    file = request.files['file']
    print("done reading file")
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        metadata = piexif.load(PIL.Image.open(file).info["exif"])
        exif_bytes = piexif.dump(metadata)
        processed = process_image_detectron2(Image.open(file), predictor_detectron)
            

        all_items_dict = {}
        for requested_item in requested_instances:
            xs = np.where(processed == config.KNOWN_CLASSES_CITYSCAPES[requested_item])[0]
            ys = np.where(processed == config.KNOWN_CLASSES_CITYSCAPES[requested_item])[1]

            all_items_dict[requested_item] = {'xs': ','.join(map(str, xs)), 'ys': ','.join(map(str, ys))}
        
        response_json = json.dumps(all_items_dict)

        response = app.response_class(
            response=response_json,
            status=200,
            mimetype='application/json'
        )
        return response


@app.route('/blur', methods=['POST'])
def upload_file():
    global predictor

    if predictor == None:
        predictor = load_model()

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        metadata = piexif.load(PIL.Image.open(file).info["exif"])
        exif_bytes = piexif.dump(metadata)
        processed = predict(Image.open(file), predictor, metadata)
        imgByteArr = io.BytesIO()
        exif_bytes = piexif.dump(metadata)
        processed.save(imgByteArr, format='JPEG', exif=exif_bytes)
        imgByteArr = imgByteArr.getvalue()
            
        response = make_response(imgByteArr)
        response.headers.set('Content-Type', 'image/jpeg')
        response.headers.set('Content-Disposition', 'attachment', filename='response.jpg')
        return response

