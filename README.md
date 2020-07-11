#INSTALLATION on AWS EC2

* sudo apt-get update
* sudo apt-get install apache2
* sudo apt-get install libapache2-mod-wsgi-py3
* sudo apt-get install python3.6
* sudo ln -sT /usr/bin/python3 /usr/bin/python
* sudo apt-get install python3-pip
* sudo ln -sT /usr/bin/pip3 /usr/bin/pip
* sudo pip install flask
* sudo apt-get install virtualenv
* mkdir ~/flaskapp
* sudo ln -sT ~/flaskapp /var/www/html/flaskapp
* cd ~/flaskapp
* virtualenv flask --python=python3
* sudo vim /etc/apache2/sites-enabled/000-default.conf:
```
WSGIDaemonProcess flaskapp threads=5
WSGIScriptAlias / /var/www/html/flaskapp/flaskapp.wsgi application-group=%{GLOBAL}

<Directory flaskapp>
        WSGIProcessGroup flaskapp
        WSGIApplicationGroup %{GLOBAL}
        Order deny,allow
        Allow from all
</Directory>
```
* sudo apachectl restart
* sudo pip3 install numpy --no-cache-dir
* sudo pip3 install Pillow --no-cache-dir
* sudo pip3 install tensorflow --no-cache-dir
* sudo apt update && sudo apt install -y libsm6 libxext6 libxrender-dev
* sudo pip3 install opencv-python --no-cache-dir

* sudo pip3 install -U torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
* sudo pip3 install -U cython pyyaml==5.1
* sudo pip3 install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
* sudo pip3 install -U piexif
* sudo pip3 install -U detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.5/index.html
* sudo pip3 install scikit-image
* wget http://download.tensorflow.org/models/deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz
* tar -xzvf deeplab_cityscapes_xception71_trainfine_2018_09_08.tar.gz
* python3 image_process.py
* chmod 777 ...

vim /var/log/apache2/error.log

Blur image:
```
curl -L -X POST -F 'file=@ladybugImageOutput_00000012.jpg' 0.0.0.0/blur --output image.jpg
```
returns blurred image

Get polygons for each class:
```
curl -L -X POST -F 'file=@ladybugImageOutput_00000012.jpg' 0.0.0.0/segment?instances=road,sidewalk,building,wall,fence,pole,traffic_light,traffic_sign,vegetation,terrain,sky,person,rider,car,truck,bus,train,motorcycle,bicycle,misc
```
Example output:
```
{
    "car": {
        "xs": "1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1977,1978...
        "ys": "7883,7884,7885,7886,7887,7888,7889,7890,7891,7892,7893,7894,7895,7896,7897,7898,7899,7900,7901,7902,7903,7904,7905,7906,7907,7908,7909,7910,7911,7912,7913,7914,7915,7916,7917,7918,7919,7920,7921,7922,7923,7924,7925,7926,7927,7928,7929,7883...
     }
}
```
