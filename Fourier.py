import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.io.wavfile as wav
from scipy.fftpack import fft, fftfreq,ifft
from scipy.interpolate import interp1d

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

#Se imprime un mensaje en la terminal explicando por que no se puede hacer la transformada de incompletos.dat
print("La cantidad de datos en incompletos.dat es mucho menor a la de signal.dat, ademas su delta no es el mismo para todos, la derivada de la funcion no es continua, los datos no estan espaciados con la misma distancia, por lo que la transformada de Fourier tampoco podra ser continua.")

#Se hace una interpolacion cuadratica y cubica de los datos incompletos.dat con 512 puntos
fcuad=interp1d(x_inc, y_inc, kind='quadratic') 
fcubi=interp1d(x_inc, y_inc, kind='cubic') 
xn=np.linspace(min(x_inc), max(x_inc), 512)

fun_cuad=fcuad(xn)
fun_cubi=fcubi(xn)

#Se hace la transformada de fourier de cada una de las series de datos con la funcion creada anteriormente.
DFT_orig=DFT(y_inc)	
DFT_cuad=DFT(fun_cuad)	
DFT_cubi=DFT(fun_cubi)

#Se halla el numero de puntos y el delta para cada situacion.
nint=len(xn)
dxint=xn[1]-xn[0]
fint=fftfreq(nint, dxint) 

ninc=len(x_inc)
dxinc=x_inc[1]-x_inc[0]
finc=fftfreq(ninc, dxinc) 

#Se hace la grafica con los 3 subplots de las 3 transformadas de Fourier
f,(ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(f_sig, np.abs(fourier_DFT1), c='b')
ax2.plot(fint, np.abs(DFT_cuad), c='orangered')
ax3.plot(fint, np.abs(DFT_cubi), c='g')
ax1.set_title("Transformada de los datos originales")
ax2.set_title("Transformada de la interpolacion cuadratica")
ax3.set_title("Transformada de la interpolacion cubica")
plt.savefig('GarciaCamila_TF_interpola.pdf')

#Se imprime un mensaje donde describa las diferencias encontradas entre la transformada de Fourier de la se˜nal original y las de las interpolaciones.
#print("")

#Se aplica el filtro pasabajos con una frecuencia de corte fc=1000 hz para la frecuencia original 

filtro1000 = 1000
filtro500 = 500
trans1 = np.copy(fourier_DFT)
for i in range (len (fourier_DFT)):
	if (abs(f_sig[i])<=filtro1000):
		trans1[i]=trans1[i]
	else: 	
		trans1[i]=0

#y con una frecuencia de corte de fc=500hz

for i in range (len (fourier_DFT)):
	if (abs(f_sig[i])<=filtro500):
		fourier_DFT[i]=fourier_DFT[i]
	else: 	
		fourier_DFT[i]=0

#Se aplica el filtro pasabajos con una frecuencia de corte fc=1000 hz para la frecuencia cuadratica

trans2 = np.copy(DFT_cuad)
for i in range (len (DFT_cuad)):
	if (abs(fint[i])<=filtro1000):
		trans2[i]=trans2[i]
	else: 	
		trans2[i]=0

#y con una frecuencia de corte de fc=500hz

for i in range (len (DFT_cuad)):
	if (abs(fint[i])<=filtro500):
		DFT_cuad[i]=DFT_cuad[i]
	else: 	
		DFT_cuad[i]=0


#Se aplica el filtro pasabajos con una frecuencia de corte fc=1000 hz para la frecuencia cubica

trans3 = np.copy(DFT_cubi)
for i in range (len (DFT_cubi)):
	if (abs(fint[i])<=filtro1000):
		trans3[i]=trans3[i]
	else: 	
		trans3[i]=0


#y con una frecuencia de corte de fc=500hz

for i in range (len (DFT_cubi)):
	if (abs(fint[i])<=filtro500):
		DFT_cubi[i]=DFT_cubi[i]
	else: 	
		DFT_cubi[i]=0

#Se hace la grafica con los dos subplots de las 3 señales y se guarda.
plt.figure()
plt.subplot(3,2,1)
plt.plot(f_sig, trans1 )
plt.title("Transformada de Fourier-1000")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada")
plt.xlim(-1500,1500)
plt.subplot(3,2,2)
plt.plot(f_sig, fourier_DFT )
plt.title("Transformada de Fourier-500")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada")
plt.xlim(-1500,1500)
plt.subplot(3,2,3)
plt.plot(fint, trans2, color="g")
plt.title("Transformada Cuadratica-1000")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada")
plt.xlim(-1500,1500)
plt.subplot(3,2,4)
plt.plot(fint, DFT_cuad, color="g")
plt.title("Transformada Cuadratica filtro 500")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada")
plt.xlim(-1500,1500)
plt.subplot(3,2,5)
plt.plot(fint, trans3, color="r")
plt.title("Transformada Cubica filtro 1000")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada")
plt.xlim(-1500,1500)
plt.subplot(3,2,6)
plt.plot(fint, DFT_cubi, color="r")
plt.title("Transformada Cubica filtro 500")
plt.xlabel("Frecuencia")
plt.ylabel("Transformada")
plt.xlim(-1500,1500)

plt.savefig("GarciaCamila_2Filtros.pdf")

