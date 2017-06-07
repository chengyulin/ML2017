import numpy as np
from numpy import genfromtxt
from numpy import loadtxt,savetxt
import sys
import keras
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout
from keras.layers import Input,dot,add,concatenate
from keras.layers.embeddings import Embedding
from keras.layers.merge import Add,Dot
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,BatchNormalization
import pickle
from keras.utils import plot_model
from keras.regularizers import l2
from keras.models import load_model
import sys
def rmse(y_true,y_pred):
	#y_pred = K.round(y_pred)
	a =((y_pred - y_true)**2)
	a = K.mean(a)
	a = K.sqrt(a)
	return a

model=load_model('best.hdf5',custom_objects={'rmse':rmse})
test = genfromtxt(sys.argv[1],dtype=None,delimiter=',')
test = test[1:,:]
test = test[:,1:]
test = test.astype('float')
test_user = test[:,0]

test_movie = test[:,1]
print (test_movie)
out = model.predict([test_user,test_movie])

print (out)
f1  = open(sys.argv[2],'w+')
print ('TestDataID,Rating',file=f1)
for x in range(100336):
	if out[x] > 5:
		out[x] = 5
	if out[x] < 1:
		out[x] = 1
	a = out[x]
	a = float(a)
	print(str(x+1)+','+str(a),file=f1)

	