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
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,BatchNormalization
import pickle
from keras.utils import plot_model
from keras.regularizers import l2

def rmse(y_true,y_pred):
	#y_pred = K.round(y_pred)
	a =((y_pred - y_true)**2)
	a = K.mean(a)
	a = K.sqrt(a)
	return a

def MF(umax,mmax,dim,lmbda):
	x_u = Input(shape=(1,))
	x_m = Input(shape=(1,))
	embedding_u = Embedding(umax+1,dim)(x_u)
	embedding_m = Embedding(mmax+1,dim)(x_m)#,embeddings_regularizer= l2(lmbda)
	f_u = Flatten()(embedding_u)
	f_m = Flatten()(embedding_m)
	f_u = Dropout(0.4)(f_u)
	f_m = Dropout(0.4)(f_m)
	p = dot([f_u,f_m],axes=1)
	model = Model(inputs=[x_u,x_m],outputs=[p])
	#adam = Adam(lr = 0.0005)
	model.compile(optimizer='adam',loss='mse')
	model.summary()
	return model


data = genfromtxt('train.csv',dtype=None,delimiter=',')
data = data[1:,:]
data = data[:,1:]
data = data.astype('float')

print (data.shape)
umax = 0
mmax = 0 
for x in range(899873):
	if data[x][0] > umax:
		umax = data[x][0]
	if data[x][1] > mmax:
		mmax = data[x][1]
umax = int(umax);mmax = int(mmax)
print (umax,mmax)
# indices = np.arange(data.shape[0])  
# np.random.shuffle(indices) 
# data = data[indices]
user = data[:,0]
movie = data[:,1]
rank = data[:,2]
model = MF(umax,mmax,100,1e-7)
earlystopping = EarlyStopping(monitor='val_loss', patience = 10, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath='best.hdf5',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=False,
                             monitor='val_loss',
                             mode='min')
model.fit([user,movie],rank,epochs=30, batch_size=1024)

test = genfromtxt('test.csv',dtype=None,delimiter=',')
test = test[1:,:]
test = test[:,1:]
test = test.astype('float')
test_user = test[:,0]
#print (test_user)
test_movie = test[:,1]
print (test_movie)
out = model.predict([test_user,test_movie])
#print (out)
#out = np.round(out)
print (out)
f1  = open('out.csv','w+')
print ('TestDataID,Rating',file=f1)
for x in range(100336):
	if out[x] > 5:
		out[x] = 5
	if out[x] < 1:
		out[x] = 1
	a = out[x]
	a = float(a)
	print(str(x+1)+','+str(a),file=f1)

	