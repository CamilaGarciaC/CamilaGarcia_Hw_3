import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq,ifft
from scipy import interpolate

#Se carga y se almacena los datos de signal.dat
signal = np.genfromtxt("signal.dat",delimiter=",")
x_sig=signal[:,0]
y_sig=signal[:,1]
n=len(signal)
nx=len(x_sig)

#Se carga y almacena los datos de incompletos.dat
incompletos = np.genfromtxt("incompletos.dat",delimiter=",")
x_inc=incompletos[:,0]
y_inc=incompletos[:,1]

#Grafica de datos de signal.dat y se guarda sin mostrarla 
plt.figure()
plt.plot(x_sig,y_sig,label="Signal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Signal")
plt.savefig("GarciaCamila_signal.pdf")


#Se declara el sample spacing y rating
spacing = xsen[1]-xsen[0]
rating = 1/spacing

#Se define la funcion de transformada
def T(x_sig,y_sig):
	contador=np.linspace(0,0, nx)
	for i in range(nx):
		for j in range(nx):
			contador[i]=contador[i]+(np.exp(-1j*2*np.pi*j*i/nx)*y_sig[j])
	contador=contador/nx
	return contador
contador=abs(T(x_sig,y_sig))


