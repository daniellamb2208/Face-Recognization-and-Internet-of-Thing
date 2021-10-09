import os
import requests
import cv2
x = os.listdir('lamb')
tmp = cv2.imread('lamb/'+x[0])

p = requests.post('http://daniellamb.duckdns.org:1234/recieve', data={'number':str(len(x)), 'pixel_x':tmp.shape[0], 'pixel_y':tmp.shape[1], 'info':'tired', 'name':'peter'})
print(p.content)
for i in x:
    img = cv2.imread('lamb/'+i)
    r = requests.post('http://daniellamb.duckdns.org:1234/recieve', data=img.tobytes(), headers={'Content-Type':'application/octet-stream'})
    print(r.content)

