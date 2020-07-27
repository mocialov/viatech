import multiprocessing

bind = "0.0.0.0:8080"
workers = 1
#bind = 'unix:flaskrest.sock'
umask = 0o007
reload = True

#logging
accesslog = '-'
errorlog = '-'
