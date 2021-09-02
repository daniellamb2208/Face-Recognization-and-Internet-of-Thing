from tensorflow import keras
import cv2
import os
from random import sample
import numpy as np
import joblib
from itertools import combinations
from scipy.spatial import distance

# for thinking how to tell stranger

def preprocessing(x):

    if not (x.ndim == 4 or x.ndim == 3):
        raise ValueError('Dimension error '+ str(x.dim))
    if not (x.shape == (160,160,3)):
        x = cv2.resize(x, (160,160), interpolation=cv2.INTER_AREA)

    mean = np.mean(x, axis=(0,1,2), keepdims=True)
    std = np.std(x, axis=(0,1,2), keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y

emb_model = keras.models.load_model('facenet_keras.h5', compile = False)
svm_model = joblib.load('svm.pkl')
rel_model = joblib.load('label.pkl')

names = os.listdir('data_cropped')

for name in names:
    print(name)
    path = os.listdir('data_cropped/'+name)
    path_sampled = sample(path, 10)
    print(path_sampled)
    all_path = ['data_cropped/'+name+'/'+i for i in path_sampled]
    
    combination = list(combinations(all_path, 2))

    for (x, y) in combination:
        i = preprocessing(cv2.imread(x))
        j = preprocessing(cv2.imread(y))

        emb1 = emb_model.predict(np.expand_dims(i, axis=0))
        emb2 = emb_model.predict(np.expand_dims(j, axis=0))

        print(distance.euclidean(emb1, emb2))
        # np.linalg.norm(a-b)
    
        #print(emb)
        #result = svm_model.predict(emb)
        #print(rel_model.inverse_transform(result))

