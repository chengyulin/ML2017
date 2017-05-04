
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

data = genfromtxt(sys.argv[1],delimiter=',',dtype=None)  
data = np.delete(data,0,0)
label = [int(x) for x in data[:,0]]
label = np.array(label)                                   
num = data.shape[0]
feature = [ [int(x) for x in data[r,1].split()] for r in range (num)]
feature = np.array(feature)
# print (np.amax(label))
feature = feature.reshape((28709,48,48))
feature = feature/255
val = feature[:5500,:,:]
feature = feature[5500:,:,:]
f2 = np.zeros((13000*2,48,48))
f2[:13000,:,:] = feature[:13000,:,:]
f2[13000:26000,:,:] = feature[:13000,:,::-1]

unl = np.zeros((10209*2,48,48))
unl[:10209,:,:] =  feature[13000:,:,:]
unl[10209:,:,:] =  feature[13000:,:,::-1]
unl = unl.reshape((20418,48*48))
 
f2 = f2.reshape((26000,48*48))
val = val.reshape((5500,48*48))
label2 = np.zeros((28709,7))
l2 = np.zeros((13000*2,7))

for x in range(28709):
    a = label[x]
    label2[x][a] = 1
    #label2[x+28709][a] = 1
vlab = label2[:5500,:]
label2 = label2[5500:,:]
l2[:13000,:] =label2[:13000]
l2[13000:26000,:] =label2[:13000]
#l2[23209*2:23209*3,:] =label2
model = Sequential()
model.add(Dense(input_dim=48*48,units=1024))
model.add(Activation('sigmoid'))
model.add(Dense(units=512))
model.add(Activation('sigmoid'))
model.add(Dense(units=256))
model.add(Activation('sigmoid'))
model.add(Dense(units=128))
model.add(Activation('sigmoid'))
model.add(Dense(units=7))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

ma = 0.0
loc = 0
v = open("val","w+")
t = open("tra","w+")
for a in range(50):
    print (a)
    model.fit(x=f2,y=l2,batch_size=32,epochs=1)
    score = model.evaluate(x=val,y=vlab)
    print ('\nTest loss:', score[0])
    print (a, score[1],file = v)
    score2 = model.evaluate(x=f2,y=l2)
    print (a,score2[1],file = t)
    if score[1] > ma:
        ma = score[1]
        tmp = model
        loc = a
        model.save('udnnmodel.h5')    

uout = model.predict(unl)
a1 = np.zeros((23209*2,48*48))
a1[:26000,:] =f2
a1[26000:,:] = unl
l1 = np.zeros((23209*2,7))
l1[:26000,:] =l2
l1[26000:,:] = uout
for a in range(50):
    print (a)
    model.fit(x=a1,y=l1,batch_size=32,epochs=1)
    score = model.evaluate(x=val,y=vlab)
    print ('\nTest loss:', score[0])
    print (50+a, score[1],file = v)
    score2 = model.evaluate(x=a1,y=l1)
    print (50+a,score2[1],file = t)
    if score[1] > ma:
        ma = score[1]
        tmp = model
        loc = a
        model.save('udnnmodel.h5')  

f2 = open(sys.argv[2])
test = genfromtxt(sys.argv[2],delimiter=',',dtype=None)  
test = np.delete(test,0,0)                                  
num = test.shape[0]
test = [ [int(x) for x in test[r,1].split()] for r in range (num)]
test = np.array(test)
test = test.reshape((num,48*48))
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
with open(sys.argv[3],"w+") as fd:
    print("id,label",file=fd) 
    for i in range(0,test.shape[0]):
        print(str(i)+","+str(cnt[i]),file=fd)

