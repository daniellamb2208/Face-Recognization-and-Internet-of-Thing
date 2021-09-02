import cv2
import os

# for collecting face of celebrities from Internet 'google'

cascade = cv2.CascadeClassifier('rpi_client/haarcascade_frontalface_default.xml')

who = 'shine'
path = 'server/raw/' + who
ab_path = os.listdir(path)
#print(ab_path)
#print(len(ab_path))

count = 0
try:
    os.mkdir('server/data_cropped/'+who)
except FileExistsError:
    print('already')

for i in ab_path:

    image = cv2.imread(path+'/'+i)
    print(i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor = 1.11, minNeighbors = 5)

    if len(faces) == 0: 
        print(i, end='')
        print(' is failed')
        pass
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w, :]

        face = cv2.resize(face, (160,160), interpolation=cv2.INTER_AREA)
        count = count + 1
        cv2.imwrite('server/data_cropped/'+who+'/'+str(count)+'.jpg', face)
