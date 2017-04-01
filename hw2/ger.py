import numpy as np
import pandas as pd
import math as math
import random as rnd
from numpy import genfromtxt
from numpy import loadtxt
import sys
import os
import pickle

def sigmoid(z):
  return 1/(1+np.exp(-z))

data = genfromtxt(sys.argv[3],delimiter=',',dtype=float)
dflag = genfromtxt(sys.argv[4],delimiter='\n',dtype=float)
test = genfromtxt(sys.argv[5],delimiter=',',dtype=float)

for x in range(data.shape[0]):
	data[x][1]=data[x][0]**2
	data [x][65] = data [x][102]
	data [x][66] = data [x][3]**2
	data [x][67] = data [x][4]**2
	data [x][68] = data [x][5]**2
	data [x][69] = data [x][3]**3
	data [x][70] = data [x][4]**3
	data [x][71] = data [x][5]**3
for x in range(test.shape[0]):
	test [x][1] = test[x][0]**2
	test [x][65] = test [x][102]
	test [x][66] = test [x][3]**2
	test [x][67] = test [x][4]**2
	test [x][68] = test [x][5]**2
	test [x][69] = test [x][3]**3
	test [x][70] = test [x][4]**3
	test [x][71] = test [x][5]**3

data = data[1:data.shape[0],:72]
test = test[1:test.shape[0],:72]
vdata = np.var(data,axis =0 )
#vtest = np.var(test,axis=0)
vdata = np.sqrt(vdata)  
test = (test - np.mean(data,axis=0)) / vdata
data = (data - np.mean(data,axis=0))/vdata

dataf = np.zeros((data.shape[0],data.shape[1]+1))
dataf[:,:-1] = data
for x in range(data.shape[0]):
	dataf[x][data.shape[1]] = dflag[x]
#print (dataf)
flag0 = (dataf[:,data.shape[1]]==0)
flag1 = (dataf[:,data.shape[1]]==1)
data0 = data[flag0,:]
data1 = data[flag1,:]
#print (data0.shape)
#print (data1.shape)
mu0 = np.mean(data0,axis=0)
mu1 = np.mean(data1,axis=0)
#print (mu0.shape)
var0 = np.zeros((72,72))
for x in range(data0.shape[0]):
	a = data0[x,:]
	a1 = a - mu0
	t1 = np.zeros((72,1))
	t2 = np.zeros((1,72))
	for x in range(72):
		t1[x][0] = a1[x]
		t2[0][x] = a1[x]
	#print (np.dot(t1,t2))
	var0 += np.dot(t1,t2)
var0 =var0 / data0.shape[0]
#print (var1)
var1 = np.zeros((72,72))
for x in range(data1.shape[0]):
	a = data1[x,:]
	a1 = a - mu1
	t1 = np.zeros((72,1))
	t2 = np.zeros((1,72))
	for x in range(72):
		t1[x][0] = a1[x]
		t2[0][x] = a1[x]
	#print (np.dot(t1,t2))
	var1 += np.dot(t1,t2)
var1 =var1/ data1.shape[0]
#print (var0)
sig =  (data1.shape[0]*var1 +data0.shape[0]*var0)/(data0.shape[0]+data1.shape[0])
print (sig)
inverse = np.linalg.inv(sig)
cof = np.dot((mu1-mu0).T,inverse)
print (cof.shape)
b =0.0
tmp = np.dot(mu1.T,inverse)
tmp = np.dot(tmp,mu1)
b+=(-1/2)*tmp
tmp = np.dot(mu0.T,inverse)
tmp = np.dot(tmp,mu0)
b+=(1/2)*tmp
b+=math.log(data1.shape[0]/data0.shape[0])

print (b)
tmp2 = np.dot(test,cof)	
tmp2 +=  b
cnt =[]
for x in range(tmp2.shape[0]):
	if (tmp2[x])>0.5:
		cnt.append(1)
	else:
		cnt.append(0)
with open(sys.argv[6],"w+") as fd:
    print("id,label",file=fd) 
    for i in range(0,tmp2.shape[0]):
        print(str(i+1)+","+str(cnt[i]),file=fd)


