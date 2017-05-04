
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
# print (label.shape)
feature = feature.reshape((28709,48,48,1))
feature = feature/255
val = feature[:5500,:,:,:]
feature = feature[5500:,:,:,:]
f2 = np.zeros((23209*3,48,48,1))
f2[:23209,:,:,:] = feature
tmp = feature
feature = feature[:,:,::-1,:]
#f2[28709:28709*2,:,:,:] = feature
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)

f2[23209:23209*2,:,:,:] = feature
datagen.fit(tmp)
f2[23209*2:23209*3,:,:,:] = tmp

label2 = np.zeros((28709,7))
l2 = np.zeros((23209*3,7))
for x in range(28709):
    a = label[x]
    label2[x][a] = 1
    #label2[x+28709][a] = 1
vlab = label2[:5500,:]
label2 = label2[5500:,:]
l2[:23209,:] =label2
l2[23209:23209*2,:] =label2
l2[23209*2:23209*3,:] =label2
model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same',input_shape=(48, 48,1)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))#24
model.add(Dropout(0.375))

model.add(Conv2D(128, (5, 5), padding = 'same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))#12
model.add(Dropout(0.375))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))#6
model.add(Dropout(0.25))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
#model.add(Dense(512))
#model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
#model.add(Dropout(0.5))
#model.add(Dense(24))
#model.add(Activation('relu'))
model.add(Dense(units=7))
model.add(Activation('softmax'))
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

ma = 0.0
loc = 0
for a in range(100):
    print (a)
    model.fit(x=f2,y=l2,batch_size=32,epochs=1)
    score = model.evaluate(x=val,y=vlab)
    print ('\nTest loss:', score[0])
    print ('Test Acc:', score[1])

    if score[1] > ma:
        ma = score[1]
        tmp = model
        loc = a
        model.save('mymodel.h5')
    # else:
    #     break
    # if score[1]>0.6:
    #     break
print (ma,loc)
# model = tmp
# f2 = open(sys.argv[2])
# test = genfromtxt(sys.argv[2],delimiter=',',dtype=None)  
# test = np.delete(test,0,0)                                  
# num = test.shape[0]
# test = [ [int(x) for x in test[r,1].split()] for r in range (num)]
# test = np.array(test)
# test = test.reshape((num,48,48,1))
# test = test/255
# #datagen.fit(test)
# output = model.predict(test)
# cnt = []
# for x in range(test.shape[0]):
#     m = 0.0
#     tmp = -1
#     for y in range(7):
#         if output[x][y] > m:
#            m = output[x][y]
#            tmp = y
#     cnt.append(tmp)
# # print (output)
# with open(sys.argv[3],"w+") as fd:
#     print("id,label",file=fd) 
#     for i in range(0,test.shape[0]):
#         print(str(i)+","+str(cnt[i]),file=fd)

