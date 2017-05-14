import numpy as np
import sklearn
from sklearn import decomposition
from sklearn.decomposition import PCA 
from random import randint
import math
from numpy import genfromtxt,loadtxt 
from PIL import Image
import os
from scipy.misc import imread ,imsave,imshow
import matplotlib.pyplot as plt
#dr = "faceExpressionDatabase"
data =[]
for x in os.listdir("faceExpressionDatabase"):
	#print(x)
	if ((x[0]>='A')& (x[0]<='J') & (x[1]=='0') & (x[2] >='0') & (x[2] <='9')):
		st = "./faceExpressionDatabase/" 
		x = st + x
		data.append(imread(x))
#print (len(data))
data = np.array(data)
data = data.reshape((100,4096))
data_mean = data.mean(axis=0, keepdims=True)
avr = data_mean.reshape(64,64)
imsave('avr.png',avr)
data_ctr = data - data_mean
u, s, v = np.linalg.svd(data_ctr)
imgdata =[]
fig = plt.figure()
for x in range(9):
	tmp = v[x].reshape((64,64))
	imgdata.append(tmp)
imgdata = np.array(imgdata)
#print (len(imgdata))
imgdata = imgdata.reshape((9,64,64))
#print (imgdata.shape())
# print (imgdata[0].shape)
for i in range(9):
	tmp = fig.add_subplot(3,3,i+1)
	tmp.imshow(imgdata[i],cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
plt.show()
eg = []
for x in range(5):
	eg.append(v[x])
eg = np.array(eg)
ans = np.zeros((100,4096))
ans = ans+ data_mean
for x in range(100):
	for y in range(5):
		tmp = np.dot(data_ctr[x].T , eg[y])
		tmp = np.dot(tmp,eg[y])
		ans[x]+=tmp
ans = ans.reshape((100,64,64))
fig2 = plt.figure()
#print (ans)
for i in range(100):
	tmp = fig2.add_subplot(10,10,i+1)
	tmp.imshow(ans[i],cmap='gray')
	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
#imsave(fig,"1.jpg")
plt.show()
ans = np.zeros((100,4096))
ans = ans+ data_mean
for x in range(100):
	S = np.diag(s)
	recon = np.dot(u[:,:x],np.dot(S[:x,:x],v[:x,:]))
	rmse = np.sqrt(np.mean((recon-data_ctr)**2))
	rmse = rmse/255
	print (rmse)
	if rmse < 0.01:
		print ("smallest",x)
		break