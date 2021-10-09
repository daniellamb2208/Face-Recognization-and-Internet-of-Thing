import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

import numpy as np
import json
import requests

import time
import RPi.GPIO as io
import os

io.setmode(io.BOARD)
io.setup(11, io.OUT)

def success():
    io.output(11, True)
    time.sleep(0.5)
    io.output(11, False)

count = 0
try:
    os.mkdir('testing')
except FileExistsError:
    count = len(os.listdir('testing'))

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognize = ''

with PiCamera() as cam:
    cam.resolution = (1296,976)
    cam.framerate = 30
    rawCapture = PiRGBArray(cam, size=(1296,976)) 

    # count = 0
    while True:
        for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array
            crop = image.copy() # for crop without rectangle
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)
            #print(len(faces))
            if len(faces) == 0:
                #print('No face here')
                pass
            else: 
                for (x, y, w, h) in faces:
                    try:    #avoid out of range
                        face = crop[y-20:y+h+20, x-20:x+w+20, :]
                        face = cv2.resize(face, (160,160), interpolation=cv2.INTER_AREA)
                    except:
                        pass
                    print('POST to server')
                    tmp = requests.post('http://daniellamb.duckdns.org:1234/validate', data=face.tobytes(), headers={'Content-Type':'application/octet-stream'})
                    # print('finished')

                    # get result from response
                    print('result')
                    
                    print('\n')
                    result = tmp.content.decode('utf-8')
                    print(result)
                    if recognize == result and result != "unknown":
                    # if not tmp.content == b'unknown':
                        cv2.putText(image, result, (x+w, y+int(h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0),1, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
                        success()
                    else:
                        recognize = result
                        # cv2.putText(image, "S", (x+w, y+int(h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255),1, cv2.LINE_AA)
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
            cv2.imshow('stream', image)
            rawCapture.truncate(0)
            k = cv2.waitKey(1)
            rawCapture.truncate(0)
            
            if k%256==27:
                break
        if k%256 == 27:
            print("Escape hit, closing...")
            break

cv2.destroyAllWindows()
