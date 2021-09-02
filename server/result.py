from os import name
from flask import request, Flask
import json
import base64
from flask.templating import render_template
import numpy as np
import cv2

from tensorflow import keras
import joblib

from sklearn.preprocessing import Normalizer

# Get POST from rpi, Recognize the face, and Response

app = Flask(__name__)
emb_model = keras.models.load_model('facenet_keras.h5', compile = False)
svc_model = joblib.load('svm.pkl')
name_label = joblib.load('label.pkl')
count = 0
print(name_label)

def preprocessing(face):
    mean = np.mean(face, axis=(0,1,2), keepdims=True)
    std = np.std(face, axis=(0,1,2), keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(face.size))
    face = (face - mean) / std_adj

    return face

@app.route("/", methods=['POST'])
def recv_face():
    
    res = request.data  # take POST method 

    x = np.frombuffer(res, dtype=np.uint8)  # decode bytes to ndarray
    x = x.reshape((160, 160, 3))    # reshape image back to (160, 160, 3)
        # whether to check the data 

    x = preprocessing(x)
    f = emb_model.predict(np.expand_dims(x, axis=0))
    e = Normalizer(norm='l2').transform(f)
    r = svc_model.predict(e)

    g = name_label.inverse_transform(r)
    print(g)    

    print('Handled a request')

    return 'b'
    # reutrn response to client side

if __name__ == "__main__":
    app.run("0.0.0.0", port=8081)  #端口为8081