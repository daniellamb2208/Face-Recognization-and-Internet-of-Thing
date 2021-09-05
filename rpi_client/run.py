import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

import numpy as np
import json
import requests

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


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
                    face = crop[y:y+h, x:x+w, :]
                    face = cv2.resize(face, (160,160), interpolation=cv2.INTER_AREA)

                    print('POST to server')
                    tmp = requests.post('http://192.168.0.102:8081/validate', data=face.tobytes(), headers={'Content-Type':'application/octet-stream'})
                    print('finished')

                    # get result from response
                    print(tmp.content)

                    if not tmp.content == b'Wrong':
                        cv2.putText(image, tmp.content.decode('utf-8'), (x+w, y+int(h/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255),1, cv2.LINE_AA)
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
