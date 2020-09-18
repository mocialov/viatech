import config as cfg
import os
os.environ['FVCORE_CACHE'] = cfg.PYTORCH_DOWNLOADS_PATH
os.environ['WEB_CONCURRENCY'] = '1'
os.environ['MKL_DISABLE_FAST_MM'] = '1'
os.environ['LRU_CACHE_CAPACITY'] = '1'
os.environ['FLASK_ENV'] = 'development'

import io
from flask import send_file, session
from flask import Flask, flash, request, url_for, jsonify, json, make_response
from werkzeug.utils import secure_filename
from image_process import *
import numpy as np
import cv2
from PIL import Image
import piexif
from PIL.ExifTags import TAGS
import base64
from instance_segmentation import *
from skimage import measure
import base64
from flask import *
import gc

app = Flask(__name__)
app.secret_key = cfg.SECRET_KEY
app.debug = True
#app.use_reloader = False
#app.with_threads = False
#app.threaded = True

response = None
predictor = None
predictor_detectron = None
temp = 0

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


@app.before_first_request
def initialize():
    print ("initialize")

    global predictor_detectron, predictor
    print ("Called only once, when the first request comes in")

    if predictor_detectron == None:
        predictor_detectron = load_detectron_model()

    if predictor == None:
        predictor = load_model()

@app.route('/', methods=['GET'])
def get_page():
    print ("get get")
    return 'hi'

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
            contour_list = []
            for contour in contours:
                contour_list.append(list(zip(contour[:, 1], contour[:, 0])))
            all_items_dict[requested_item] = {'contours': contour_list}

        print (all_items_dict)
        
        #all_items_dict = {}
        #for requested_item in requested_instances:
        #    xs = np.where(processed == config.KNOWN_CLASSES_CITYSCAPES[requested_item])[0]
        #    ys = np.where(processed == config.KNOWN_CLASSES_CITYSCAPES[requested_item])[1]
        #
        #    all_items_dict[requested_item] = {'xs': ','.join(map(str, xs)), 'ys': ','.join(map(str, ys))}
        
        response_json = json.dumps(all_items_dict)

        #response = app.response_class(
        #    response=response_json,
        #    status=200,
        #    mimetype='application/json'
        #)

        imgByteArr = io.BytesIO()
        temp.save(imgByteArr, format='JPEG')
        imgByteArr = imgByteArr.getvalue()
        #response = make_response(imgByteArr)
        #return response
        return { 'Status' : 200, 'polygons': response_json, 'ImageBytes': base64.encodebytes(imgByteArr).decode('ascii')}
        #return response


@app.route('/blur', methods=['POST'])#['POST'])
def upload_file():
    global predictor, temp, response
    print ("blurring")

    temp += 1
    print('global var', temp)

    print(request)
    print(request.method)
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
            print("all good")

            filename = secure_filename(file.filename)
            metadata = None
            exif_bytes = None
            if 'exif' in PIL.Image.open(file).info:
                metadata = piexif.load(PIL.Image.open(file).info["exif"])
                exif_bytes = piexif.dump(metadata)
            #del metadata
            #del exif_bytes
            #print("returning")
            #return 'hello'
            processed = predict(Image.open(file), predictor, metadata)

            
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
            

            del metadata
            del processed
            del exif_bytes
            #del imgByteArr
            gc.collect()
            return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)


