from gensim.models import Word2Vec
import word2vec
import numpy as np
from argparse import ArgumentParser
import os
import sys
import multiprocessing

word2vec.word2phrase('all.txt', 'phrases.txt', verbose=True)
#word2vec.word2vec('phrases.txt' , 'my_model.bin',size=100, verbose=True)
# parser = ArgumentParser()
# parser.add_argument('--train', action='store_true',
#                     help='Set this flag to train word2vec model')
# parser.add_argument('--corpus-path', type=str, default='all.txt',
#                     help='Text file for training')
# parser.add_argument('--model-path', type=str, default='my_model.bin',
#                     help='Path to save word2vec model')
# parser.add_argument('--plot-num', type=int, default=600,
#                     help='Number of words to perform dimensionality reduction')
# args = parser.parse_args()
MIN_COUNT = 6
WORDVEC_DIM = 300
WINDOW = 5
NEGATIVE_SAMPLES = 10
ITERATIONS = 5
MODEL = 1
LEARNING_RATE = 1e-3
#CPU_COUNT = multiprocessing.cpu_count()
# train model
word2vec.word2vec(
    train='phrases.txt',
    output='my_model.bin',
    cbow=MODEL,
    size=WORDVEC_DIM,
    min_count=MIN_COUNT,
    window=WINDOW,
    negative=NEGATIVE_SAMPLES,
    threads = 4,
    iter_=ITERATIONS,
    alpha=LEARNING_RATE,
    verbose=True,)

model = word2vec.load('my_model.bin')

vocabs = []                 
vecs = []                   
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
print (len(vecs))
vecs = np.array(vecs)[:800]
#print (vecs,method)
# for x in vecs:
# 	if (np.isnan(x) == True):
# 		x = 0
vocabs = vocabs[:800]

'''
Dimensionality Reduction
'''
# from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,method='exact')
reduced = tsne.fit_transform(vecs)


'''
Plotting
'''
import matplotlib.pyplot as plt
from adjustText import adjust_text
import nltk
# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
puncts = ["'", '.', ':', ";", ',', "?", "!", u"’","Page","-","_"]


plt.figure()
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and all(c not in label for c in puncts)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))#0.5
#test_model('my_model.bin')
#print ((model['Ron']- model['Weasley']) - (model['Harry']- model['Potter']))
plt.savefig('hp.png', dpi=600)
plt.show()
#word2vec.word2clusters('my_model', 'clusters.txt', 100, verbose=True)