import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,BatchNormalization
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import random
train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]

#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    stopword = stopwords.words('english')
    lmtzr = WordNetLemmatizer()
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
            
            article = re.sub('[^a-zA-Z]', ' ', article)
            article = text_to_word_sequence(article, lower=True, split=' ')
            article = [ w for w in article if w not in stopword ]
            article = [ lmtzr.lemmatize(w) for w in article ]
            article = ' '.join(article)
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0]) 
    random.seed(42) 
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)


def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################
def main():

    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_corpus)
    word_index = tokenizer.word_index
    pickle.dump(tokenizer,open('tk','wb'))
    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_matrix = tokenizer.texts_to_matrix(X_data,mode='tfidf')
    test_matrix = tokenizer.texts_to_matrix(X_test,mode='tfidf')

    ### padding to equal length
    #print ('Padding sequences.')
    #train_sequences = pad_sequences(train_sequences)
    #max_article_length = train_sequences.shape[1]
    #test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_matrix,train_tag,split_ratio)
    #X_train = X_train.reshape((X_train.shape[0],1,X_train.shape[1]))
    print (Y_train.shape)
    print (X_train.shape)
    ### get mebedding matrix from glove
    # print ('Get embedding dict from glove.')
    # embedding_dict = get_embedding_dict('glove/glove.6B.%dd.txt'%embedding_dim)
    # print ('Found %s word vectors.' % len(embedding_dict))
    # num_words = len(word_index) + 1
    # print ('Create embedding matrix.')
    # embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    print ('Building model.')
    
    
    for x in range(20):
        model = Sequential()
        print (x)
        model.add(Dense(512,activation='elu',input_dim=40587))
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='elu'))
        model.add(Dropout(0.5))
        model.add(Dense(512,activation='elu'))
        model.add(Dropout(0.5))
        # model.add(Dense(512,activation='elu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(128,activation='elu'))
        # model.add(Dropout(0.5))
        model.add(Dense(38,activation='sigmoid'))
        model.summary()

        adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
        tmp = str(x)+'.hdf5'
        model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])
   
        earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(filepath=tmp,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_f1_score',
                                     mode='max')
       
        hist = model.fit(X_train, Y_train, 
                         validation_data=(X_val, Y_val),
                         epochs=1000, 
                         batch_size=batch_size,
                         callbacks=[earlystopping,checkpoint])

    Y_pred = model.predict(test_matrix)
    thresh = 0.4
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()