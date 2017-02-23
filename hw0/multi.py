import sys
from numpy import loadtxt,savetxt,zeros
import numpy as np
arr1 = np.loadtxt(sys.argv[1],dtype = int ,delimiter=',')
arr2 = np.loadtxt(sys.argv[2],dtype = int ,delimiter=',')
#print arr1
#print arr2
arr3 = np.dot(arr1,arr2)
#print arr3
arr3 = np.reshape(arr3,arr3.size)
arr3.sort();
f1 = open('ans_one.txt', 'w',)
np.savetxt(f1,arr3,fmt= "%d")
	
