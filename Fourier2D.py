#Se importan los paquetes
import matplotlib.pylab as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft, fftfreq, ifft, fft2, ifft2
from scipy import misc
from skimage import io
from matplotlib.colors import LogNorm

#Se almacenan los datos en un arreglo nupy
data =plt.imread("arbol.png")
arr=np.array(data)

imagen1=fft2(arr)
imagen2=imagen1.copy()	

plt.figure()
plt.plot(abs(imagen1), norm=LogNorm(vmin=1))
plt.savefig("GarciaCamila_FT2D.pdf")
