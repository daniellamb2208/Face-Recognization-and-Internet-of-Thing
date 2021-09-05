from tensorflow import keras
import cv2
import os
from random import sample
import numpy as np
import joblib
from itertools import combinations
from scipy.spatial import distance
import pymongo
from bson.binary import Binary
import pickle

# for thinking how to tell stranger

def reconstruct_distance():

    client = pymongo.MongoClient("mongodb://192.168.0.102:27017/")
    database = client["proj"]   # database of this proj
    data = database['dataset']  # data collection
    th = database['q3']
    print('Distance')
    names = data.find({})
    print(names)
    for name in names:
        print(name['name'])
        print(name['embs'])
        
        embs = name['embs']
        embs = pickle.loads(embs)
        selected = sample(embs, 10)

        combination = list(combinations(selected, 2))

        distance_intra = []
        for (x, y) in combination:
            distance_intra.append(distance.euclidean(x, y))
            # np.linalg.norm(a-b)

        print(name)
        # name
        # print(distance_intra)
        # intra_distance
        q3 = np.percentile(distance_intra,[75], interpolation='midpoint')
        print(q3)
        th.insert_one({'name':name['name'], 'q3':q3[0]})
        print("Done")
