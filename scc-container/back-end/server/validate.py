from itertools import combinations
import os
from re import T
import cv2
import numpy as np
from numpy.lib.function_base import append, select
from tensorflow import keras
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from scipy.spatial import distance

#this is for getting testing data

def preprocessing(face):
    if not (face.ndim == 4 or face.ndim == 3):
        raise ValueError('Dimension error '+ str(face.dim))
    if not face.shape == (160,160,3):
        face = cv2.resize(face, (160,160), interpolation=cv2.INTER_AREA)
    
    mean = np.mean(face, axis=(0,1,2), keepdims=True)
    std = np.std(face, axis=(0,1,2), keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(face.size))
    face = (face - mean) / std_adj

    return face

def l2_normalize(x):
        return (x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-10)))

def train():
    model = keras.models.load_model('facenet_keras.h5', compile = False)

    path = "data_cropped/"
    names = [i for i in os.listdir(path)]
    print(names)

    f = open("emb.txt", "w")
    dic = {}

    for name in names:
        _feature = []
        print("processing "+name)
        base_path = path + name + '/'
        face_path = [os.path.join(base_path, f) for f in os.listdir(base_path)]   # data_cropped/lamb/[1.jpg 2.jpg ...]
        a = 0    
        for i in face_path:
            face = cv2.imread(i)

            _feature.append(l2_normalize(model.predict(np.expand_dims(preprocessing(face), axis=0))))    # facenet extract 128dim v

        from itertools import combinations
        from random import sample

        # you = sample(_feature, 3)
        you = list(combinations(you, 2))
        distance_intra = []
        for (x, y) in you:
            distance_intra.append(distance.euclidean(x,y))
        
        n = len(distance_intra)
        f.write(str(n))
        f.write("\n")
        for i in distance_intra:
            f.write(i)
            f.write('\n')
        
        #dic[name] = distance_intra
    f.close()
    return dic

def compare():
    from random import sample
    model = keras.models.load_model('facenet_keras.h5', compile = False)
    t_path = "data_test/"
    c_path = "data_cropped/"
    i_path = "inpic/"

    base_path = [os.path.join(c_path, f) for f in os.listdir(c_path)]
    test_path = [os.path.join(t_path, g) for g in os.listdir(t_path)]
    in_path = [os.path.join(i_path, k) for k in os.listdir(i_path)]

    print(base_path)
    print(test_path)
    print(in_path)
    '''
    base_embs = []
    for i in base_path:
        chosen = sample(os.listdir(i), 1)
        chosen = str(i + '/' + chosen[0])
        face = cv2.imread(chosen)
        base_embs.append(l2_normalize(model.predict(np.expand_dims(preprocessing(face), axis=0))))

    test_embs = []
    for k in test_path:
        face = cv2.imread(k)
        test_embs.append(l2_normalize(model.predict(np.expand_dims(preprocessing(face), axis=0))))

    distance_inter = []
    for m in base_embs:
        for l in test_embs:
            distance_inter.append(distance.euclidean(l, m))
    '''
    #print(len(distance_inter))
    #print(distance_inter)

    output = []
    for i in base_path:
        the = [os.path.join(i, j) for j in os.listdir(i)]
        the = sample(the, 5)
        

        emb_test = []
        for k in the:
            f = cv2.imread(k)
            emb_test.append(l2_normalize(model.predict(np.expand_dims(preprocessing(f), axis=0))))

        comb = list(combinations(emb_test, 2))
        dis = []

        for (x, y) in comb:
            dis.append(distance.euclidean(x, y))

        output.append(dis)

    print(output)
    print(len(output))