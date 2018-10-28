import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq,ifft
from scipy import interpolate

#Se carga y se almacena los datos de signal.dat
signal = np.genfromtxt("signal.dat",delimiter=",")
xsig=signal[:,0]
ysig=signal[:,1]

#Se carga y almacena los datos de incompletos.dat
incompletos = np.genfromtxt("incompletos.dat",delimiter=",")
xinc=incompletos[:,0]
yinc=incompletos[:,1]

#Grafica de datos de signal.dat y se guarda sin mostrarla 
plt.figure()
plt.plot(xsig,ysig,label="Signal")
plt.xlabel("Tiempo")
plt.ylabel("y(x)")
plt.title("Signal")
plt.savefig("GarciaCamila_signal.pdf")
