from tensorflow import keras
#from keras.models import load_model

import os
import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC


def preprocessing(x):

    if not (x.ndim == 4 or x.ndim == 3):
        raise ValueError('Dimension error '+ str(x.dim))

    mean = np.mean(x, axis=(0,1,2), keepdims=True)
    std = np.std(x, axis=(0,1,2), keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y


model = keras.models.load_model('facenet_keras.h5')
print("Facenet model loaded!")
names = [f for f in os.listdir('crop')]
print("Who is in the dataset", end='')
print(names)

_name = []        
_feature = []    # concatenate all feature to a list along with _name to feed SVM

for name in names:
    print("processing "+name)
    base_path = 'crop/' + name + '/'
    face_path = [os.path.join(base_path, f) for f in os.listdir(base_path)]   # formed/lamb/[1.jpg 2.jpg ...]
    
    number = 0
    for i in face_path:
        _name.append(name)  # tag with each face

        face = cv2.imread(i)
        number = number + 1
        print("Read "+str(number)+"/"+str(len(face_path)))
        _feature.append(model.predict(np.expand_dims(preprocessing(face), axis=0)))    # facenet extract 128dim v

_feature = np.concatenate(_feature)

# encode name to be label
encoder = LabelEncoder()
encoder.fit(_name)
print('label is ready')
input_label = encoder.transform(_name)

print('---------debug-------')
print('Label ', end='')
print(_name)
print(input_label)
print('---------------------')

input_data = Normalizer(norm='l2').transform(_feature)

print('_feature is ready')

# training

svm_model = SVC(kernel = 'linear')
svm_model.fit(input_data, input_label)
print('svm is trained')

import joblib

#joblib.dump(svm_model, 'svm_recog_model')

print("recognization model saved")
print('done')

'''
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

t = []
for i in _feature:
    t.append(i)
pca = PCA(n_components=9).fit(t)

x_k = []
x_l = []
x_b = []
x_t = []
x_y = []
x_k.append(_feature[0])
x_k.append(_feature[1])
x_k.append(_feature[2])
x_k.append(_feature[3])
x_k.append(_feature[4])
x_l.append(_feature[5])
x_l.append(_feature[6])
x_l.append(_feature[7])
x_l.append(_feature[8])
x_l.append(_feature[9])
x_b.append(_feature[10])
x_b.append(_feature[11])
x_b.append(_feature[12])
x_b.append(_feature[13])
x_b.append(_feature[14])
x_b.append(_feature[15])
x_b.append(_feature[16])
x_b.append(_feature[17])
x_b.append(_feature[18])
x_t.append(_feature[19])
x_t.append(_feature[20])
x_t.append(_feature[21])
x_t.append(_feature[22])
x_t.append(_feature[23])
x_t.append(_feature[24])
x_y.append(_feature[25])

x_k = pca.transform(x_k)
x_l = pca.transform(x_l)
x_b = pca.transform(x_b)
x_t = pca.transform(x_t)
x_y = pca.transform(x_y)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(x_k[:,0], x_k[:,1],x_k[:,2],x_k[:,3], x_k[:,4], 'o', markersize=8, color='green', alpha=0.5, label='k')
ax.plot(x_l[:,0], x_l[:,1],x_l[:,2],x_l[:,3], x_l[:,4], 'o', markersize=8, color='blue', alpha=0.5, label='l')
ax.plot(x_b[:,0], x_b[:,1],x_b[:,2],x_b[:,3], x_b[:,4],x_b[:,5],x_b[:,6], x_b[:,7],x_b[:,8], 'o', markersize=8, color='red', alpha=0.5, label='b')
ax.plot(x_t[:,0], x_t[:,1],x_t[:,2],x_t[:,3], x_t[:,4],x_t[:,5], 'o', markersize=8, color='black', alpha=0.5, label='t')
#ax.plot(x_y[:,0], 'o', markersize=8, color='yellow', alpha=0.5, label='y')
plt.title("embs")
ax.legend(loc='upper right')

plt.show()
print('good')
'''