import cv2
import os
from picamera import PiCamera
from picamera.array import PiRGBArray
import numpy as np

# get face upfront by entering name and automatically save 25 face wihout resizing

cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
batch = 200

print('Please specify who you are, thanks:')
name = input()
count = 0

try:
    os.mkdir('data/'+name)
except FileExistsError:
    t = os.listdir('data/'+name)
    count = len(t)

with PiCamera() as cam:
    cam.resolution = (1296,976)
    cam.framerate = 30
    rawCapture = PiRGBArray(cam, size=(1296,976)) 

    # count = 0
    while True:
        for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            image = frame.array

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor = 1.11, minNeighbors = 5, minSize=(80,80))
            #print(len(faces))
            if len(faces) == 0:
                #print('No face here')
                pass
            else: 
                (x, y, w, h) = faces[0]
                face = image[y-10:y+h+10, x-10:x+w+10, :]
                # face = cv2.resize(face, (160,160), interpolation=cv2.INTER_AREA)
                #print((x,y,h,w))    
                count = count + 1
                cv2.imwrite('data/' + name + '/' + str(count)+'.jpg', face)
                print(str(count)+'/'+str(((count-1)//batch+1)*batch))

                if count % batch == 0:
                    cv2.destroyAllWindows()
                    print('Collected' + str(batch) + 'pieces of your face, Thanks for your help')
                    exit()

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
