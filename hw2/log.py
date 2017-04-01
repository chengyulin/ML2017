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
  '''
  truncated sigmoid to avoid overflows
  '''
  # if z >= 100:
  #   return 1
  # if z <= -100:
  #   return 0  
  return 1/(1+np.exp(-z))

data = genfromtxt(sys.argv[3],delimiter=',',dtype=float)
dflag = genfromtxt(sys.argv[4],delimiter='\n',dtype=float)
test = genfromtxt(sys.argv[5],delimiter=',',dtype=float)
for x in range(data.shape[0]):
	data[x][1]=data[x][0]**2
	data[x][63] = data[x][0]**3
	data [x][65] = data [x][102]
	data [x][66] = data [x][3]**2
	data [x][67] = data [x][4]**2
	data [x][68] = data [x][5]**2
	data [x][69] = data [x][3]**3
	data [x][70] = data [x][4]**3
	data [x][71] = data [x][5]**3
for x in range(test.shape[0]):
	test [x][1] = test[x][0]**2
	test[x][63] = test[x][0]**3
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
#vtest = np.sqrt(vtest) 
test = (test - np.mean(data,axis=0)) / vdata
data = (data - np.mean(data,axis =0 ))/vdata

data2 = data[0:6000 ,:(data.shape[1])]
dflag2 = dflag[0:6000 ,]
data = data[:data.shape[0],:(data.shape[1])]
dflag = dflag[:dflag.shape[0],]
#print (dflag)
#print (data.shape)
cof = np.zeros(72)
lr = 1
b_grad = 0
b = 0
b_lr = 0.0
w_lr = [0.0]*72
mi = float(1e7)
fin1 = np.zeros(72)
fin2 = np.zeros(72)
fin = np.zeros(72)
lam = 0
for x in range(int(9e4)):
	b_grad = 0.0
	w_grad = [0.0]*72
	tmp = np.dot(data,cof)
	tmp += b
	tmp = sigmoid(tmp)
	# for i in range(tmp.shape[0]):
	# 	tmp[i] = sigmoid(tmp[i])
	tmp2 = np.dot(data2,cof)
	tmp2 += b
	tmp2 = sigmoid(tmp2)
	entropy = -np.sum(dflag2 * np.log(tmp2+1e-200) + (1-dflag2) * (np.log(1-tmp2+1e-200)) )
	#tmp2 = sigmoid(tmp2)
	# for i in range(tmp2.shape[0]):
	# 	tmp2[i] = sigmoid(tmp2[i])
	delta = dflag - tmp
	#delta2 = dflag2 - tmp2
	if x%1000==0:
		print (x , entropy)
	if (entropy < mi):
		mi = entropy
		fin1 = cof
	if ((entropy > mi) & (x > int(5000))):
		print (x,mi)
		break
	b_grad = -(np.sum(delta))# + b	
	w_grad = -(np.dot(data.T,delta)) + lam * cof
	#print (w_grad.shape)
	b_lr = b_lr + b_grad**2
	w_lr = w_lr + w_grad**2
	b = b -(lr/np.sqrt(b_lr))* b_grad #1 * b_grad# 1.5e-4
	cof = cof -(lr/np.sqrt(w_lr))* w_grad# * w_grad 

print (fin1)
fin = fin1
tmp2 = np.dot(test,fin)	
tmp2 +=  b
cnt =[]
for x in range(tmp2.shape[0]):
	if sigmoid(tmp2[x])>0.5:
		cnt.append(1)
	else:
		cnt.append(0)
with open(sys.argv[6],"w+") as fd:
    print("id,label",file=fd) 
    for i in range(0,tmp2.shape[0]):
        print(str(i+1)+","+str(cnt[i]),file=fd)


