#Se importa el paquete de numpy y matplotlib

import numpy as np 
import matplotlib.pyplot as plt 
from numpy.linalg import *

data = np.genfromtxt("WDBC.dat",delimiter=",", usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

def covarianza(x): 
	size = np.shape(x)[1]
	p = np.shape(x)[0] 
	cov = np.zeros([d,d]) 
	for i in range(size):
	    for j in range(size):
	        meani = np.mean(x[:,i])
		meanj = np.mean(x[:,j])
		cov[i,j] = (np.sum((x[:,i]-meani)*(x[:,j]-meanj)))/(p-1)
	return cov 
	
matriz = covarianza(data)


