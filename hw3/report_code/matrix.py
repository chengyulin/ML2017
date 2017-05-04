#!/usr/bin/env python
# -- coding: utf-8 --
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
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    """
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    #model_path = os.path.join('mymodel.h5')
emotion_classifier = load_model('mymodel.h5')
#np.set_printoptions(precision=2)
data = genfromtxt(sys.argv[1],delimiter=',',dtype=None)  
data = np.delete(data,0,0)
label = [int(x) for x in data[:,0]]
label = np.array(label)                                   
num = data.shape[0]
print ('aaa')
feature = [ [int(x) for x in data[r,1].split()] for r in range (num)]
feature = np.array(feature)
feature = feature.reshape((28709,48,48,1))
feature = feature/255
val = feature[:5500,:,:,:]
label2 = np.zeros((28709,7))
for x in range(28709):
    a = label[x]
    label2[x][a] = 1
    #label2[x+28709][a] = 1
vlab = label[:5500]
#print (val)
dev_feats = val
predictions = emotion_classifier.predict_classes(dev_feats)
#print (predictions.shape())
te_labels = vlab
conf_mat = confusion_matrix(te_labels,predictions)
np.set_printoptions(precision=2)
print (conf_mat)
plt.figure()
plot_confusion_matrix(conf_mat, classes=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"])
plt.show()