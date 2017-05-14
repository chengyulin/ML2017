import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA 
from random import randint
import math
from numpy import genfromtxt,loadtxt 
import sys
rdata = np.load(sys.argv[1])
cof = genfromtxt("train2.txt")
cof = cof/1.5
print (cof)
f1 = open(sys.argv[2],"w+")
print("SetId,LogDim",file=f1) 
for i in range(200):
	print (i)	
	data = rdata[str(i)]
	data = np.array(data)
	pca = sklearn.decomposition.PCA(100)
	data2 = pca.fit_transform(X=data)
	pca2 = sklearn.decomposition.PCA(80)
	data2 = pca2.fit_transform(X=data2)
	pca2 = sklearn.decomposition.PCA(60)
	data2 = pca2.fit_transform(X=data2,y=None)
	cov = np.cov(data2.T)
	a,b = np.linalg.eig(cov)
	su = 0
	mx = np.sum(a)
	cnt = 0
	for j in range(60):
		su = su + a[j]
		cnt+=1
		tmp = mx * cof[j+1]
		if su > tmp:
			break
	cnt = math.log(cnt)
	print(str(i)+","+str(cnt),file=f1)


