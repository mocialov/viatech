![image](https://drive.google.com/uc?export=view&id=14Q9ekkDIoz96l5fv0LqA35lwqJLJBhOA)


## INSTALLATION on AWS EC2

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
        ServerAdmin webmaster@localhost
        DocumentRoot /var/www/html

WSGIDaemonProcess flaskapp threads=5
WSGIScriptAlias / /var/www/html/flaskapp/flaskapp.wsgi application-group=%{GLOBAL}

<Directory flaskapp>
        WSGIProcessGroup flaskapp
        WSGIApplicationGroup %{GLOBAL}
        Order deny,allow
        Allow from all
</Directory>

        # Available loglevels: trace8, ..., trace1, debug, info, notice, warn,
        # error, crit, alert, emerg.
        # It is also possible to configure the loglevel for particular
        # modules, e.g.
        #LogLevel info ssl:warn

        ErrorLog ${APACHE_LOG_DIR}/error.log
        CustomLog ${APACHE_LOG_DIR}/access.log combined
```
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

## Watch apache log:
watch tail -20 /var/log/apache2/error.log


## Gunicorn

* sudo pip3 install gunicorn
* #sudo python3 flaskapp.py
* #sudo gunicorn --bind 0.0.0.0:8080 flaskapp:app
* sudo vim /etc/systemd/system/flaskrest.service:
```
[Unit]
Description=Gunicorn instance to serve flask application
After=network.target
[Service]
User=root
Group=www-data
WorkingDirectory=/home/ubuntu/flaskapp/
Environment="PATH=/home/ubuntu/flaskapp/flask/bin"
ExecStart=/usr/local/bin/gunicorn --config gunicorn_config.py wsgi:app
[Install]
WantedBy=multi-user.target
```

* vim gunicorn_config.py:
```
import multiprocessing
bind = "0.0.0.0:8080"
workers = 1
#bind = 'unix:flaskrest.sock'
umask = 0o007
reload = True
#logging
accesslog = '-'
errorlog = '-'
```

* #sudo gunicorn --config gunicorn_config.py flaskapp:app

* sudo systemctl restart flaskrest.service
* sudo systemctl start flaskrest.service
* sudo systemctl enable flaskrest.service
* sudo systemctl status flaskrest.service

* sudo vim /etc/apache2/sites-available/flaskrest.conf:
```
<VirtualHost *:*>

        ErrorLog ${APACHE_LOG_DIR}/flaskrest-error.log
        CustomLog ${APACHE_LOG_DIR}/flaskrest-access.log combined

        <Location />
                ProxyPass unix:/home/ubuntu/flaskapp/flaskrest.sock|http://0.0.0.0:8080
                ProxyPassReverse unix:/home/ubuntu/flaskapp/flaskrest.sock|http://0.0.0.0:8080
        </Location>
</VirtualHost>
```

* sudo rm /etc/apache2/sites-enabled/flaskrest.conf
* sudo ln -s /etc/apache2/sites-available/flaskrest.conf /etc/apache2/sites-enabled/
* sudo /etc/init.d/apache2 start

```
server_ip=0.0.0.0
```

```
curl -i --max-time 60 --connect-timeout 60 $server_ip:8080
```

```
curl --remote-name --remote-header-name --write-out "Downloaded %{filename_effective} file" --max-time 60 --connect-timeout 60 -s -L -X POST -F "file=@some-image.jpg" $server_ip:8080/blur
```

```
curl -i --max-time 60 --connect-timeout 60 -s -L -X POST -F "file=@some-image.jpg" $server_ip:8080/blur
```

```
curl -i --max-time 60 --connect-timeout 60 -s -L -X POST -F "file=@some-image.jpg" $server_ip:8080/segment?instances=road,sidewalk,building,wall,fence,pole,traffic_light,traffic_sign,vegetation,terrain,sky,person,rider,car,truck,bus,train,motorcycle,bicycle,misc
```

```
{
  "ImageBytes": ...
  "polygons": "{\"car\": {\"contours\": [[[7679.0, 2258.0... }, \"sidewalk\": {\"contours\": [.....}}
}
```


1 gunicorn process: RAM (including system)=1.3GB (max 2.5GB)

Uses 100% of all CPUs

No GPU


worker=N specifies how many gunicorn processes are running (pgrep gunicorn), which should be N+1 (1 master process and N workers)
