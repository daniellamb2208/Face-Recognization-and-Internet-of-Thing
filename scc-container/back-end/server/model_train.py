from itertools import count
from tensorflow import keras
#from keras.models import load_model

import os
import cv2
import numpy as np
import pymongo
import joblib
from bson.binary import Binary
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

url = "mongodb://"+os.environ['name']+":"+os.environ['auth']+"@mongo:27017/"
#url = "mongodb://192.168.0.102:27017/"

def preprocessing(x):

    if not (x.ndim == 4 or x.ndim == 3):
        raise ValueError('Dimension error '+ str(x.dim))
    if not x.shape == (160,160,3):
        x = cv2.resize(x, (160,160), interpolation=cv2.INTER_AREA)

    mean = np.mean(x, axis=(0,1,2), keepdims=True)
    std = np.std(x, axis=(0,1,2), keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x):
    return (x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-10)))

def update():
    # Processing extract face from image with new comer
    client = pymongo.MongoClient(url)
    database = client["proj"]   # database of this proj


    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    raw_path = "raw/"

    untrain = database['untrained']
    _new = untrain.find()
    print(_new) # Show who is going to be trained in model
    for who in _new:
        up = 0
        rep = raw_path + who + '/'
        print(who)

        abp = [os.path.join(rep, i) for i in os.listdir(rep)] # absolute path list
        leng = len(abp)
        leng_from_zero = 0
        for j in abp:
            leng_from_zero = leng_from_zero + 1
            print(str(leng_from_zero)+'/'+str(leng))

            image = cv2.imread(j)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 5)

            if len(faces) == 0:
                pass

            (x, y, w, h) = faces[0]
            face = image[y:y+h, x,x+w, :]
            face = cv2.resize(face, (160,160), interpolation=cv2.INTER_AREA)
            up = up + 1
            cv2.imwrite("data_cropped/"+who+'/'+str(up)+'.jpg', face)

def train():

    model = keras.models.load_model('facenet_keras.h5', compile = False)
    print("Facenet model loaded!")
    client = pymongo.MongoClient(url)
    database = client["proj"]   # database of this proj
    collection = database['dataset']

    collection.delete_many({})  # avoid duplicate data

    dataset_path = 'data_cropped/'
    names = [i for i in os.listdir(dataset_path)]
    print("Who are already in dataset ",end='')
    print(names)

    _name = []
    _feature = []   # for training

    for name in names:
        print("processing "+name)
        base_path = dataset_path + name + '/'
        face_path = [os.path.join(base_path, f) for f in os.listdir(base_path)]   # data_cropped/lamb/[1.jpg 2.jpg ...]
        
        # face_path = sample(face_path, 10)
        # print(face_path)

        number = 0
        _embs = []  # for collect embs of ONE person 
        for i in face_path:
            _name.append(name)  # tag with each face

            face = cv2.imread(i)
            number = number + 1
            print("Read "+str(number)+"/"+str(len(face_path)))
            emb = l2_normalize(model.predict(np.expand_dims(preprocessing(face), axis=0)))
            _embs.append(emb)
            _feature.append(emb)    # facenet extract 128dim v
        collection.insert_one({'name':name, 'embs':Binary(pickle.dumps(_embs, protocol=-1), subtype=128)})
        
        print(name + " embs Inserted")
    
    _feature = np.concatenate(_feature)

    # encode name to be label
    encoder = LabelEncoder()
    en = encoder.fit(_name)
    input_label = en.transform(_name)
    print('Label is ready')

    input_data = Normalizer(norm='l2').transform(_feature)
    print('_feature is ready')

    

    # Training

    svm_model = SVC(kernel = 'linear')
    svm_model.fit(input_data, input_label)
    print('svm is trained')

    joblib.dump(en, 'label.pkl')
    print("Name label saved")

    joblib.dump(svm_model, 'svm.pkl')

    print("Recognization model saved")
    print('Done')
'''
def train(init_flag = False):
    
    if init_flag:
        init()
        return
    update()
    dataset_path = 'data_cropped/'

    client = pymongo.MongoClient(url)
    database = client["proj"]   # database of this proj

    model = keras.models.load_model('facenet_keras.h5', compile = False)
    print("Facenet model loaded!")
    names = [f for f in os.listdir(dataset_path)]
    print("Who is in the dataset:", end='')
    print(names)

    collection = database['dataset']
    # untrain = database['untrained']

    tbt = untrain.find()
    for i in tbt:   # to be trained
        tp = dataset_path + i + '/' # this path # data_cropped/name/
        absp = [os.path.join(tp, j) for j in os.listdir(tp)] # abs path # data_cropped/name/1.jpg
        print(i+" is now procssing")
        _number = 0
        embs = []
        for k in absp:
            img = preprocessing(cv2.imread(k))
            _number = _number + 1
            print(str(_number)+'/'+str(len(absp)))
            embs.append(l2_normalize(model.predict(np.expand_dims(img, axis=0))))

        collection.insert_one({'name':i, 'embs':Binary(pickle.dumps(embs, protocol=-1), subtype=128)})

    _name = []        
    _feature = []    # concatenate all feature to a list along with _name to feed SVM

    all = collection.find({})
    for a in all:
        n = a['name']
        e = a['embs']
        print(n)
        print(e)
        e = pickle.loads(e)
        for b in e:
            _name.append(n)
            _feature.append(b)
    
    _feature = np.concatenate(_feature)

    # encode name to be label
    encoder = LabelEncoder()
    en = encoder.fit(_name)
    print('Label is ready')
    input_label = en.transform(_name)


    print('---------debug-------')
    print('Label ', end='')
    print(names)
    print('_name')
    print(_name)
    print('input_data')
    print(input_label)
    print('encoder', end='')
    print(encoder)
    print(en)
    print('---------------------')
    

    input_data = Normalizer(norm='l2').transform(_feature)

    print('_feature is ready')

    # training

    svm_model = SVC(kernel = 'linear')
    svm_model.fit(input_data, input_label)
    print('svm is trained')

    joblib.dump(en, 'label.pkl')
    print("Name label saved")

    joblib.dump(svm_model, 'svm.pkl')

    print("Recognization model saved")
    print('Done')
'''