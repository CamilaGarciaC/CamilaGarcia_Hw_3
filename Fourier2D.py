#Se importan los paquetes de datos 
import numpy as np
import matplotlib.pyplot as plt 
import math 
from scipy.fftpack import fft2, fftshift, ifftshift, ifft2
from scipy.signal import convolve2d

#Se alamacena la imagen en un arreglo de numpy
img=plt.imread('arbol.png')
x, y = img.shape 

#Usando los paquetes, se hace la transformada de la imagen
transf=fft2(img)
transf_s=fftshift(transf)
transf_s_abs=abs(transf_s)

#Se le saca el logaritmo para que se vea con mayor claridad
log_transf=np.log(transf_s_abs)

#Se grafica la transformada y se guarda
plt.imshow(log_transf)
plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia")
plt.ylabel("y(t)")
plt.savefig("GarciaCamila_FT2D.pdf")
