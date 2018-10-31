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

#Se hace la definicion de transformada de fourierpara los datos de la señal.
def DFT(x):
    N = len(x)
    n= np.arange(len(x))
    F_k=[]
    for k in range(N):
        omega = 2*np.pi*n*k/N
        F_real = x*np.cos(2*np.pi*n*k/N)
        F_im = x*np.sin(2*np.pi*n*k/N)
        F_r = np.sum(F_real)
        F_i = np.sum(F_im)
        Fk= F_r -1j*F_i
        F_k.append(Fk)
    return F_k

#Se declara la función de las frecuencias utilizando el paquete fftfreq
n_sig=len(x_sig)
dx_sig=x_sig[1]-x_sig[0]
f_sig=fftfreq(n_sig, dx_sig)
print("Se uso el paquete de fftfreq para la grafica de transformada")

#Se grafica la transformada de Fourier y se guarda
fourier_DFT=DFT(y_sig)
plt.figure()
plt.plot(f_sig, np.abs(fourier_DFT))
plt.title("Transformada de Fourier de signal")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada de Fourier")
plt.savefig("GarciaCamila_TF.pdf")

#Se imprime un mensaje que indica las principales frecuencias de la señal.
princip=[]
z = len (f_sig)
for i in range (z):
	if (fourier_DFT[i]>0.5):
		princip.append(f_sig[i])

print("Las frecuencias principales de la transformada de Fourier son", princip)

#Se hace un filtro pasa bajos con frecuencia fc = 1000Hz
filtro1=1000.0 
fourier_DFT1=DFT(y_sig)

#Se realiza la transformada inversa
F_filtrado=[]
for i in range(len(f_sig)):
	if abs(f_sig[i]) > filtro1:F_filtrado.append(0)
	else:
		F_filtrado.append(fourier_DFT1[i])	 
inversa1=ifft(np.array(F_filtrado))

#Grafica de la señal filtrada
plt.figure()
plt.plot(x_sig, inversa1.real)
plt.title("Funcion filtrada con fc=1000")
plt.xlabel("tiempo")
plt.ylabel("f filtrada")
plt.savefig("GarciaCamila_filtrada.pdf")
