import numpy as np
from numpy import genfromtxt
from numpy import loadtxt,savetxt
import sys
import keras
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,BatchNormalization
import pickle


def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

f1 = open(sys.argv[1],"r")
f2 = open(sys.argv[2],"w+")
f3 = open('tag.txt')
tags =[]
for line in f3.readlines():
	a = len(line)
	line = line[:a-1]  
	tags.append(line)
#print (tags)
model = Sequential()
    
model.add(Dense(512,activation='elu',input_dim=40587))
model.add(Dropout(0.5))
model.add(Dense(512,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='elu'))
model.add(Dropout(0.5))
model.add(Dense(512,activation='elu'))
model.add(Dropout(0.5))
# model.add(Dense(128,activation='elu'))
# model.add(Dropout(0.5))
model.add(Dense(38,activation='sigmoid'))
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=[f1_score])
model.load_weights('best.hdf5')
#model.save('bag.h5')
data =[]
for line in f1.readlines():  
    data.append(line)
data = data[1:]
tokenizer1 = pickle.load(open('bagtk','rb'))
#tokenizer1.fit_on_texts(data)
ptra = tokenizer1.texts_to_matrix(data,mode='tfidf')
#ptra = ptra[:,:51867]
#ptra = pad_sequences(sequences,maxlen=206)
output = model.predict(ptra)
output = np.array(output)
for x in range(1234):
	for y in range(38):
		if output[x][y] > 0.4:
			output[x][y] = 1
		else:
			output[x][y] = 0
print (output[0])
print ("\"id\",\"tags\"",file=f2)
for x in range(1234):
	cnt = 0
	tmp=""
	for y in range(38):
		if (output[x][y] == 1) & (cnt == 0):
			tmp += tags[y]
			cnt +=1
		elif(output[x][y] == 1) & (cnt != 0):
			tmp += " "
			tmp += tags[y]
	print("\""+str(x)+"\",\""+tmp+"\"",file = f2)

