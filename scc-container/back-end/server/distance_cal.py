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
import matplotlib.pyplot as plt

# for thinking how to tell stranger


url = "mongodb://"+os.environ['name']+":"+os.environ['auth']+"@mongo:27017/"
#url = "mongodb://192.168.0.102:27017/"

def reconstruct_distance():

    client = pymongo.MongoClient(url)
    database = client["proj"]   # database of this proj
    data = database['dataset']  # data collection
    th = database['q3']
    print('Distance')
    names = data.find({})
    print(names)

    _box = []
    _name = []

    for name in names:

        
        print(name['name'])
        print(name['embs'])
        _name.append(name['name'])

        embs = name['embs']
        embs = pickle.loads(embs)
        #selected = sample(embs, 10)

        combination = list(combinations(embs, 2)) #

        distance_intra = []
        for (x, y) in combination:
            distance_intra.append(distance.euclidean(x, y))
            # np.linalg.norm(a-b)
        _box.append(distance_intra)
        print(name)
        # name
        # print(distance_intra)
        # intra_distance
        q3 = np.percentile(distance_intra, 75, interpolation='midpoint')
        q6 = np.percentile(distance_intra, 62.5, interpolation='midpoint')
        q2 = np.percentile(distance_intra, 50, interpolation='midpoint')

        th.insert_one({'name':name['name'], 'q3':q3, 'q60':q6, 'q2':q2})
        print(name['name']+" Done")
    
    plt.boxplot(_box, labels=_name)
    plt.savefig("data/box.png")