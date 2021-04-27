#production


import config as cfg
import os
#os.environ['FVCORE_CACHE'] = cfg.PYTORCH_DOWNLOADS_PATH
#os.environ['WEB_CONCURRENCY'] = '1'
#os.environ['MKL_DISABLE_FAST_MM'] = '1'
#os.environ['LRU_CACHE_CAPACITY'] = '1'
#os.environ['FLASK_ENV'] = 'development'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"#"1"

import io
from flask import send_file, session
from flask import Flask, flash, request, url_for, jsonify, json, make_response, render_template, session
from werkzeug.utils import secure_filename
from image_process import *
import numpy as np
import cv2
from PIL import Image
import piexif
from PIL.ExifTags import TAGS
import base64
from instance_segmentation import *
from crack_detector import *
from skimage import measure
import base64
from flask import *
import gc
from threading import Lock 
import pandas as pd
from time import gmtime, strftime 
from dateutil import parser
import os.path as path

import tensorflow as tf
import sys
from tensorflow.python.client import device_lib

app = Flask(__name__)
app.secret_key = cfg.SECRET_KEY
app.debug = True
#app.use_reloader = False
#app.with_threads = False
#app.threaded = True

response = None
predictor = None
predictor_detectron = None
crack_detector = None

blurring_activities_data = None
lock = Lock()


@app.before_request
def handle_chunking():
    """
    Sets the "wsgi.input_terminated" environment flag, thus enabling
    Werkzeug to pass chunked requests as streams.  The gunicorn server
    should set this, but it's not yet been implemented.
    """

    transfer_encoding = request.headers.get("Transfer-Encoding", None)
    if transfer_encoding == u"chunked":
        request.environ["wsgi.input_terminated"] = True


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in cfg.ALLOWED_EXTENSIONS


def find_contour(seg_map2, object_class):
    #set border to 'misc class -1'
    seg_map2[:, 0:1] =  254
    seg_map2[0:1, :] =  254
    seg_map2[:, -1:] =  254
    seg_map2[-1:, :] =  254

    seg_map2 = seg_map2 + 1 #free pixel value 0 for finding contours

    data = np.where(seg_map2 != object_class+1, 0, seg_map2)

    contours = measure.find_contours(data, 0)

    return contours

def get_blurring_activities_data():

    if path.exists('blurring_activities.csv'):
        blurring_stats = pd.read_csv('blurring_activities.csv', header=0)
    else:
        columns = ['customer_id', 'image_geo_coordinates', 'image_timestamp', 'blurring_timestamp', 'image_type', 'objects_blurred']
        blurring_stats = pd.DataFrame(columns=columns)

    return blurring_stats


@app.before_first_request
def initialize():

    global predictor_detectron, predictor, crack_detector, blurring_activities_data

    if predictor_detectron == None:
        predictor_detectron = load_detectron_model()

    if predictor == None:
        predictor = load_model()

    if crack_detector == None:
        crack_detector = load_crack_detector()

    blurring_activities_data = get_blurring_activities_data()


def update_blurring_activities_data(blurring_activities_data):
    with lock:
        blurring_activities_data.to_csv('blurring_activities.csv', index=False)
        get_blurring_activities_data()


@app.route('/', methods=['GET'])
def api():
    print ("get get")
    return render_template('api.html')

@app.route('/segment', methods=['POST'])
def upload_file2():
    print ("segmenting")

    global predictor_detectron

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
        
        metadata = None
        exif_bytes = None
        if "exif" in PIL.Image.open(file).info:
            metadata = piexif.load(PIL.Image.open(file).info["exif"])
            exif_bytes = piexif.dump(metadata)
        
        processed, temp = process_image_detectron2(Image.open(file), predictor_detectron)
        
        all_items_dict = {}
        for requested_item in requested_instances:
            contours = find_contour(processed, cfg.KNOWN_CLASSES_CITYSCAPES[requested_item])
            #print (list(contours), flush=True)
            contour_list = []
            contour_list2 = []
            for contour in contours:
                contour_copy = contour[:]
                contour_copy = list(contour_copy)
                deleted = 0
                prev_x = -1
                prev_y = -1
                for idx, x in enumerate(np.array(contour)[:,0]):
                    #print ("x", x, flush=True)
                    if x == prev_x:
                        if idx != len(np.array(contour)[:,0])-1:
                            if np.array(contour_copy)[:,0][idx+1-deleted] == prev_x:
                                del contour_copy[idx-deleted]
                                deleted += 1
                    elif np.array(contour)[:,1][idx] == prev_y:
                        if idx != len(np.array(contour)[:,1])-1:
                            if np.array(contour_copy)[:,1][idx+1-deleted] == prev_y:
                                del contour_copy[idx-deleted]
                                deleted += 1
                    prev_x = x
                    prev_y = np.array(contour)[:,1][idx]
                contour_list2.append(list(zip(list(map(int, np.array(contour)[:, 1].astype(int))), list(map(int, np.array(contour)[:, 0])))))
                contour_list.append(list(zip(list(map(int, np.array(contour_copy)[:, 1].astype(int))), list(map(int, np.array(contour_copy)[:, 0])))))
            all_items_dict[requested_item] = {'contours': contour_list}#, 'contour_original': contour_list2}

        
        response_json = json.dumps(all_items_dict)

        imgByteArr = io.BytesIO()
        temp.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        #response = make_response(imgByteArr)
        #return response
        return { 'Status' : 200, 'polygons': response_json, 'ImageBytes': base64.encodebytes(imgByteArr).decode('ascii')}
        #return response


@app.route('/blur', methods=['POST'])#['POST'])
def upload_file():
    global predictor, response, blurring_activities_data

    if request.method == 'POST':
       
        # check if the post request has the file part
        if 'file' not in request.files:
            print("file not found")
            flash('No file part')
            return 'no file' ##redirect(request.url)
        print("file found")
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print("no selected file")
            flash('No selected file')
            return 'no file' #redirect(request.url)
        if file and allowed_file(file.filename):
            blurring_parameter = request.args.get('blurring_level', default = None, type = int)
            customer_id = request.args.get('customer_id', default='', type=str)

            print("all good")

            image_created = ''
            lat = -1
            lon = -1
            camera_model = ''

            image_ = PIL.Image.open(file)
            filename = secure_filename(file.filename)
            metadata = None
            exif_bytes = None
            if 'exif' in image_.info:
                metadata = piexif.load(image_.info["exif"])
                exif_bytes = piexif.dump(metadata)

                if 'Exif' in metadata:
                    if 36867 in metadata['Exif']:
                        image_created = metadata['Exif'][36867]

                if 'GPS' in metadata:
                    if 2 in metadata['GPS']:
                        lat = metadata['GPS'][2]
                        if 4 in metadata['GPS']:
                            lon = metadata['GPS'][4]

                if '0th' in metadata:
                    if 272 in metadata['0th']:
                        camera_model = metadata['0th'][272]

            if image_created != '':
                image_created = parser.parse(image_created)
                image_created.strftime("%Y-%m-%dT%H:%M:%S")

            if lat != -1 and lon != -1:
                new_lat = ''
                new_lon = ''
                for i, item in enumerate(lat):
                    item = item[0] / item[1]
                    if i != 2:
                        item = str(int(item)) + ';'
                    else:
                        item = str(float(item))
                    new_lat += item
                for i, item in enumerate(lon):
                    item = item[0] / item[1]
                    if i != 2:
                        item = str(int(item)) + ';'
                    else:
                        item = str(float(item))
                    new_lon += item
                lat = new_lat
                lon = new_lon

            camera_model = camera_model.decode('utf-8') if camera_model != '' else ''
            camera_model += '_'+str(image_.size[0]) + 'x' + str(image_.size[1])


            processed, objects_blurred = predict(Image.open(file), predictor, metadata, blurring_parameter=blurring_parameter if blurring_parameter != None else 4)

            
            blurring_activities_data = blurring_activities_data.append({
                'customer_id': customer_id,
                'image_geo_coordinates': str(lat)+','+str(lon),
                'image_timestamp': image_created,
                'blurring_timestamp': strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                'image_type': camera_model,
                'objects_blurred': str(objects_blurred)
                }, ignore_index=True)
            update_blurring_activities_data(blurring_activities_data)


            imgByteArr = io.BytesIO()
            if metadata != None:
                exif_bytes = piexif.dump(metadata)
                processed.save(imgByteArr, format='JPEG', exif=exif_bytes)
            else:
                processed.save(imgByteArr, format='JPEG')
            imgByteArr = imgByteArr.getvalue()
            
            response = make_response(imgByteArr)
            response.headers.set('Content-Type', 'image/jpeg')
            response.headers.set('Content-Disposition', 'attachment', filename='response.jpg')
            

            return response


@app.route('/crack', methods=['POST'])
def upload_file3():
    global crack_detector

    if request.method == 'POST':
        if 'password' in request.form and 'username' in request.form:
            #from werkzeug.security import generate_password_hash, check_password_hash
            #print("password hash", generate_password_hash(request.form['password'], method='pbkdf2:sha256:300', salt_length=14), flush=True)
            if request.form['password'] == cfg.PASSWORD and request.form['username'] == cfg.USERNAME:
                session['logged_in'] = True
                # check if the post request has the file part
                if 'file' not in request.files:
                    print("file not found")
                    flash('No file part')
                    return 'no file' ##redirect(request.url)
                print("file found")
                file = request.files['file']
                # if user does not select file, browser also
                # submit an empty part without filename
                if file.filename == '':
                    print("no selected file")
                    flash('No selected file')
                    return 'no file' #redirect(request.url)
                if file and allowed_file(file.filename):
                    threshold = request.args.get('threshold', default = None, type = float)

                    print("all good from crack detector", flush=True)
                    filename = secure_filename(file.filename)
                    im = PIL.Image.open(file)

                    print ("predicting crack", flush=True)

                    image_np = np.array(im)

                    initial_image_shape = image_np.shape
                    image_np = cv2.resize(image_np, (640,640))

                    detections = predict_crack(image_np, crack_detector, initial_image_shape, threshold)

                    return { 'Status' : 200, 'detections': detections}
            else:
                return 'wrong username or password'
        else:
            return 'wrong username or password'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)


