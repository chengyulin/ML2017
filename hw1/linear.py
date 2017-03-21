#
import sys
from numpy import genfromtxt
import numpy as np
import math
import random
#loadTrainingData
data = genfromtxt(sys.argv[1],delimiter=',',dtype=float)
test = genfromtxt(sys.argv[2],delimiter=',',dtype=float)
for i in range(data.shape[0]):
	for j in range(data.shape[1]):
		if math.isnan(data[i][j]) == True:
			data[i][j] = 0.0
for i in range(test.shape[0]):
	for j in range(test.shape[1]):
		if math.isnan(test[i][j]) == True:
			test[i][j] = 0.0
for i in range(1,data.shape[0],1):#把不用的刪掉
	for j in range(0,data.shape[1],1):
		(data[i-1][j]) = (data[i][j])
for i in range(0,data.shape[0],1):#把不用的刪掉
	for j in range(3,data.shape[1],1):
		(data[i][j-3]) = (data[i][j])
for i in range(0,test.shape[0],1):#把不用的刪掉
	for j in range(2,test.shape[1],1):
		(test[i][j-2]) = (test[i][j])

data = data[:data.shape[0]-1,:(data.shape[1]-3)]
test = test[:test.shape[0],:(test.shape[1]-2)]#load testdata

connect = []
for w in range(0,4320,360):#接起來
	ttmp =[]
	for x in range(9,360,18):
		for y in range(data.shape[1]):
			ttmp.append(data[w+x][y])
	connect.append(ttmp)

connect2 = []	#rainfall
for w in range(0,4320,360):#接起來
	ttmp =[]
	for x in range(10,360,18):
		for y in range(data.shape[1]):
			ttmp.append(data[w+x][y]**2)
	connect2.append(ttmp)

no =[]
for w in range(0,4320,360):#接起來
	ttmp =[]
	for x in range(9,360,18):
		for y in range(data.shape[1]):
			ttmp.append(data[w+x][y]**2)
	no.append(ttmp)
o3 =[]
for w in range(0,4320,360):#接起來
	ttmp =[]
	for x in range(7,360,18):
		for y in range(data.shape[1]):
			ttmp.append(data[w+x][y])
	o3.append(ttmp)

tra = np.zeros((36,2826))
right = np.zeros((1,2826))
tes = np.zeros((36,2826))
tesright = np.zeros((1,2826))
i = 0
for w in range(6):
	for x in range(471):
		for z in range(9):
			tra[z][i] = connect[w][x+z]
			tra[z+9][i] = connect2[w][x+z]
			tra[z+18][i] = no[w][x+z]
			tra[z+27][i] = o3[w][x+z]
		right[0][i] = connect[w][x+9]
		i+=1
i = 0
for w in range(6,12,1):
	for x in range(471):
		for z in range(9):
			tes[z][i] = connect[w][x+z]
			tes[z+9][i] = connect2[w][x+z]
			tes[z+18][i] = no[w][x+z]
			tes[z+27][i] = o3[w][x+z]
		tesright[0][i] = connect[w][x+9]
		i+=1

lr = 1
b_grad = 0
b = 0
b_lr = 0.0
w_lr = [0.0]*36
lam = 1e5
start = np.zeros((1,36))
for x in range(36):
	start[0][x]= 1.0
mini = float(1e13)
fin = np.zeros((1,36))
for v in range(int(9e5)):
	b_grad = 0.0
	w_grad = [0.0]*36
	tmp = np.dot(start,tra)
	delta = right - b - tmp
	#delta += lam * start * 2
	tmp2 = np.dot(start,tes)
	delta2 = tesright - b - tmp2
	b_grad = -2*(np.sum(delta))
	if v%1000 ==0 :
		print (v,np.sum(delta2**2))
	if ((np.sum(delta2**2) > mini)):
		print(v , mini)
		break
	else:
		mini = np.sum(delta2**2)
		fin = start
	w_grad = -2*(np.dot(delta,tra.T))
	#w_grad += lam * start * 2
	b_lr = b_lr + b_grad**2
	w_lr = w_lr + w_grad**2
	b = b - (lr/np.sqrt(b_lr)) * b_grad 
	start = start - (lr/np.sqrt(w_lr)) * w_grad
start = fin
print (start)
cnt = []
for w in range(9,test.shape[0],18):
	count = 0
	for z in range(9):
		if z < 9:
			count += test[w][z] * start[0][z]
			count += (test[w+1][z]) * start[0][z+9]
			count += (test[w][z]**2) * start[0][z+18]
			count += test[w-2][z] * start[0][z+27]
	count +=  b
	#print (count)
	cnt.append(count)
for x in range(len(cnt)):
	cnt[x] = round(cnt[x])
	if cnt[x] < 0:
		cnt[x]=0

with open(sys.argv[3],"w+") as fd:
    print("id,value",file=fd) 
    for i in range(0,240):
        print("id_"+str(i)+","+str(cnt[i]),file=fd)