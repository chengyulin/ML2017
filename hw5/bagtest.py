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

def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

(_, X_test,_) = read_data(sys.argv[1],False)
tokenizer = pickle.load(open('bagtk','rb'))
test_sequences = tokenizer.texts_to_matrix(X_test,mode='tfidf')
#test_sequences = pad_sequences(test_sequences,maxlen=306)
su = np.zeros((1234,38))
for x in range(20):
    print(x)
    tmp = str(x)+'.hdf5'
    model = load_model(tmp,custom_objects={'f1_score':f1_score})
    output = model.predict(test_sequences)
    output = np.array(output)
    for x in range(1234):
        for y in range(38):
            if output[x][y] > 0.4:
                su[x][y]+=1
f2 = open(sys.argv[2],'w+')
tags = loadtxt('tag.txt',dtype=str)
#tags = pickle.load(open('tag.txt','r'))
output = np.zeros((1234,38))
for x in range(1234):
    for y in range(38):
        if su[x][y]>=15:
            output[x][y]=1
print (tags)
print ("\"id\",\"tags\"",file=f2)
for x in range(1234):
	cnt = 0
	tmp=""
	for y in range(38):
		if (output[x][y] == 1) & (cnt == 0):
			tmp += str(tags[y][2:len(tags[y])-1])
			cnt +=1
		elif(output[x][y] == 1) & (cnt != 0):
			tmp += " "
			tmp += str(tags[y][2:len(tags[y])-1])
	print("\""+str(x)+"\",\""+str(tmp)+"\"",file = f2)

