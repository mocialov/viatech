import os
os.environ['FVCORE_CACHE'] = "/home/ubuntu/.torch/fvcore_cache/"

import io
from flask import send_file
from flask import Flask, flash, request, redirect, url_for, jsonify, json, make_response
from werkzeug.utils import secure_filename
#from process_file import *
from image_process import *
import numpy as np
import cv2
from PIL import Image
import piexif
from PIL.ExifTags import TAGS
import base64
from instance_segmentation import *

UPLOAD_FOLDER = '/home/ubuntu/flaskapp/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.secret_key = "\xbe\xc1\xd5\x16\x03\xf87\xdd\xf5\xe7y\x03\xdf1\xac\xca2\x00\xee\x1d\xdc\xc2M+"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



predictor = None
predictor_detectron = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/segment', methods=['POST'])
def upload_file2():
    global predictor_detectron

    if predictor_detectron == None:
        predictor_detectron = load_detectron_model()

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print ("reading file")
        file = request.files['file'] #request.files['file'] #Image.open(request.files['file']) #cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED) #request.files['file']
        print("done reading file")#, PIL.Image.open(file)._getexif())
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print ("no selected file")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            metadata = piexif.load(PIL.Image.open(file).info["exif"])
            #print(PIL.Image.open(file)._getexif()[item])
            #piexif.load("foo1.jpg")
            exif_bytes = piexif.dump(metadata)
            #Image.open(file).save(os.path.join(app.config['UPLOAD_FOLDER'], filename), exif=exif_bytes)
            print("processing image")
            #processed = predict(Image.open(file), predictor, metadata)
            processed = process_image_detectron2(Image.open(file), predictor_detectron)
            imgByteArr = io.BytesIO()
            exif_bytes = piexif.dump(metadata)
            processed.save(imgByteArr, format='JPEG', exif=exif_bytes)
            imgByteArr = imgByteArr.getvalue()
            
            response = make_response(imgByteArr) #send_file(encoded, mimetype='image/jpg')
            response.headers.set('Content-Type', 'image/jpeg')
            response.headers.set('Content-Disposition', 'attachment', filename='response.jpg')
            return response
        else:
            print ("something else")

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''



@app.route('/blur', methods=['POST'])
def upload_file():
    global predictor

    if predictor == None:
        predictor = load_model()

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        print ("reading file")
        file = request.files['file'] #request.files['file'] #Image.open(request.files['file']) #cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED) #request.files['file']
        print("done reading file")#, PIL.Image.open(file)._getexif())
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print ("no selected file")
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            metadata = piexif.load(PIL.Image.open(file).info["exif"])
            #print(PIL.Image.open(file)._getexif()[item])
            #piexif.load("foo1.jpg")
            exif_bytes = piexif.dump(metadata)
            #Image.open(file).save(os.path.join(app.config['UPLOAD_FOLDER'], filename), exif=exif_bytes)
            print("processing image")
            processed = predict(Image.open(file), predictor, metadata)
            #processed = process_image_detectron2(Image.open(file))
            imgByteArr = io.BytesIO()
            exif_bytes = piexif.dump(metadata)
            processed.save(imgByteArr, format='JPEG', exif=exif_bytes)
            imgByteArr = imgByteArr.getvalue()
            
            response = make_response(imgByteArr) #send_file(encoded, mimetype='image/jpg')
            response.headers.set('Content-Type', 'image/jpeg')
            response.headers.set('Content-Disposition', 'attachment', filename='response.jpg')
            return response
        else:
            print ("something else")

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


from flask import send_from_directory

@app.route('/home/ubuntu/flaskapp/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

'''
if __name__ == "__main__":
    #global predictor

    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    predictor = load_model()
    app.run()'''
