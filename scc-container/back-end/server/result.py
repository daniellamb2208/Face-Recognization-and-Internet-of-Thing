import os
from time import thread_time
from flask import request, Flask, jsonify, after_this_request
import json
import base64
from flask.templating import render_template
import numpy as np
import cv2
import pymongo
from tensorflow import keras
import joblib
from bson.binary import Binary
import pickle
import time

from sklearn.preprocessing import Normalizer

import model_train
import distance_cal
# Get POST from rpi, Recognize the face, and Response

app = Flask(__name__)
emb_model = keras.models.load_model('facenet_keras.h5', compile = False)

def reload():
    global svc_model, name_label
    svc_model = joblib.load('svm.pkl')
    name_label = joblib.load('label.pkl')
reload()
count = 0
Header_count = 0    # the first header about new one
number, pixel_x, pixel_y, name, information = (0,0,0,0,None)

url = "mongodb://"+os.environ['name']+":"+os.environ['auth']+"@mongo:27017/"
print('----------------+'+url+'---------------------------')
#url = "mongodb://192.168.0.102:27017/"

client = pymongo.MongoClient(url)
database = client["proj"]   # database of this proj

def preprocessing(face):
    mean = np.mean(face, axis=(0,1,2), keepdims=True)
    std = np.std(face, axis=(0,1,2), keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(face.size))
    face = (face - mean) / std_adj

    return face

@app.route("/")
def home_page():
    return render_template("Home.html")

@app.route("/init")
def welcome():
    return "Welcome to this page without anything"

@app.route("/init/<name>/<word>/<nonce>")
def initialization(name=None, word=None, nonce=None):
    if name == 'lamb' and word == 'guess' and nonce == '31':
        database['untrained'].delete_many({})
        database['dataset'].delete_many({})
        database['q3'].delete_many({})
        database["log"].delete_many({})
        model_train.train(True)
        print('Trained')
        distance_cal.reconstruct_distance()
        print('Q3 is ready')
    else:
        return 'gg'
    return 'Initialized'

@app.route("/recieve", methods = ['POST'])
def recv_data():
    # Recieve the data of new one

    global number, pixel_x, pixel_y, name, information, Header_count

    if Header_count == 0:
        try:
            number = int(request.form["number"])
            pixel_x = request.form["pixel_x"] # supposed pixels is the same, in same bundle of data
            pixel_y = request.form["pixel_y"]
            name = request.form["name"]

            untrained = database["untrained"]
            exist = untrained.find_one({'name':name})
            if not exist:
                untrained.insert_one({'name':name})

        except:
            # error request, without above four element
            number, pixel_x, pixel_y, name, information, Header_count = (0,0,0,0,None,0)
            return '400'
        try:
            information = request.form["info"]
            print(information)
        except:
            print("No more information about this person")
            pass
        try:
            os.mkdir("raw/"+name)
        except FileExistsError:
            path = os.listdir("raw/"+name)
            amount = len(list(path))
            Header_count = Header_count + amount
            number = number + amount
        Header_count = Header_count + 1
        print('Name:  ', end='')
        print(name)
        print('amount:', end='')
        print(number)
        print('shape: ('+str(pixel_x)+', '+str(pixel_y)+')')
        
        return 'Ready for recieving pic'
    else:
        try:
            res = request.data
            x = np.frombuffer(res, dtype=np.uint8)
            x = x.reshape((int(pixel_x), int(pixel_y), 3))
            cv2.imwrite('raw/'+name+'/'+str(Header_count)+'.jpg', x)
            Header_count = Header_count + 1
            if Header_count > number:
                # successful tx all
                number, pixel_x, pixel_y, name, information, Header_count = (0,0,0,0,None,0)
                return 'Done'

            return "Recieved "+str(Header_count-1)
        except:
            number, pixel_x, pixel_y, name, information, Header_count = (0,0,0,0,None,0)
            # if failed within tranfer session, need restart to process
            return 'Transfer error'

@app.route("/validate", methods=['POST'])
def tell():
    # Get 160,160 from rpi collect in real-time face detect

    _log = database["log"]
    def timing():
        localtime = time.localtime()
        now = time.strftime("%Y-%m-%d %I:%M:%S %p", localtime)
        return now

    res = request.data  # take POST method 

    try:
        x = np.frombuffer(res, dtype=np.uint8)  # Decode bytes to ndarray
        x = x.reshape((160, 160, 3))    # Reshape image back to (160, 160, 3)
        # How to check the data 
    except:
        return '400'
    x = preprocessing(x)
    f = emb_model.predict(np.expand_dims(x, axis=0))
    e = Normalizer(norm='l2').transform(f)
    r = svc_model.predict(e)
    g = name_label.inverse_transform(r)
    bf = Binary(pickle.dumps(f, protocol=-1), subtype=128)
    print('-------------------')
    print(g)
    '''if g[0] == 'alin':
        _log.insert_one({"Time":timing(), "Who":"", "Stranger":True, "Emb":bf})
        return "Wrong"'''
#-----------------------------------
    q3 = database['q3']
    ppl = database['dataset']

    from random import sample
    from scipy.spatial import distance
    you = ppl.find_one({'name':g[0]})
    you = pickle.loads(you['embs'])
    take = sample(you, 10)

    dis = []
    for t in take:
        dis.append(distance.euclidean(t, e))
    min_inter = min(dis)
    threshold = q3.find_one({'name':g[0]})

    if min_inter > threshold['q3']:
        print('Handled a request')
        _log.insert_one({"Time":timing(), "Who":"", "Stranger":True, "Emb":bf})
        return "Wrong"
        # tell it's not the person
    # Checking the person if in the dataset
#-----------------------------------
    _log.insert_one({"Time":timing(), "Who":g[0], "Stranger":False, "Emb":bf})
    return g[0]
    # reutrn response to client side

@app.route("/show", methods=['GET'])
def history():

    @after_this_request
    def add_header(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    _log = database["log"]
    if _log.find_one() == None:
        return 'No data in the system'
    
    dictt = {}
    x = _log.find()
    nn = 0
    for i in x:
        dic = {}
        dic['name'] = i['Who']
        dic['time'] = i["Time"]
        if i['Stranger']:
            dic['Stranger'] = True
        #dic['emb'] = pickle.loads(i['Emb'])
        dictt[str(nn)] = dic
        nn = nn + 1
    return jsonify(dictt)

@app.route("/temp")
def temp():
    import requests
    return requests.post("http://192.168.0.105").content

@app.route("/train")
def train():
    model_train.train()
    distance_cal.reconstruct_distance()
    reload()
    return 'A'

@app.route("/test")
def x():
    import validate
    return validate.train()

if __name__ == "__main__":
    app.run("0.0.0.0", port=8081, debug=True)  #端口为8081