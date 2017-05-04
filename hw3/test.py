
import numpy as np
from numpy import genfromtxt
from numpy import loadtxt
import sys
import keras
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,BatchNormalization
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model = load_model('mymodel.h5')
#print (model.summary())
test = genfromtxt(sys.argv[1],delimiter=',',dtype=None)  
test = np.delete(test,0,0)                                  
num = test.shape[0]
test = [ [int(x) for x in test[r,1].split()] for r in range (num)]
test = np.array(test)
test = test.reshape((num,48,48,1))
test = test/255
#datagen.fit(test)
output = model.predict(test)
cnt = []
for x in range(test.shape[0]):
    m = 0.0
    tmp = -1
    for y in range(7):
        if output[x][y] > m:
           m = output[x][y]
           tmp = y
    cnt.append(tmp)
# print (output)
with open(sys.argv[2],"w+") as fd:
    print("id,label",file=fd) 
    for i in range(0,test.shape[0]):
        print(str(i)+","+str(cnt[i]),file=fd)

