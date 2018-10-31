#Se importan los paquetes de datos 
import numpy as np
import matplotlib.pyplot as plt 
import math 
from matplotlib.colors import LogNorm
from scipy.fftpack import fft2, fftshift, ifftshift, ifft2
from scipy.signal import convolve2d

#Se alamacena la imagen en un arreglo de numpy
img=plt.imread('arbol.png')
 
#Usando los paquetes, se hace la transformada de la imagen
transf=fft2(img)
transf_s=fftshift(transf)
transf_s_abs=abs(transf_s)

#Se le saca el logaritmo para que se vea con mayor claridad
log_transf=np.log(transf_s_abs)

#Se grafica la transformada y se guarda
plt.imshow(log_transf,plt.cm.gray)
plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia")
plt.ylabel("y(t)")
plt.savefig("GarciaCamila_FT2D.pdf")

#Se hace una funcion como filtro que elimina el ruido de la imagen 
def filtrar(transf):
    for i in range(len(transf)):
        for j in range(len(transf)):
            if (abs(transf[i,j])>4000.0 and abs(transf[i,j])<5000.0 ):
                transf[i,j]=0
            else:
                transf[i,j]=transf[i,j]
    return transf
filtrarTransf=filtrar(transf_s)

#Se grafica la transformada despues del proceso de filtrado usando lognorm y se guarda
plt.figure()
plt.imshow(np.abs(filtrarTransf),norm=LogNorm(vmin=5),cmap='gray')
plt.title('Transformada de Fourier usando filtro')
plt.savefig("GarciaCamila_FT2D_Filtrada.pdf")

#Se hace la transformada de Fourier inversa
arbolfinal=np.real(ifft2(ifftshift(filtrarTransf)))

#Se grafica la imagen filtrada
plt.figure()
plt.title('Transformada inversa de Fourier filtrada')
plt.imshow(np.abs(arbolfinal),plt.cm.gray)
plt.savefig("GarciaCamila_Imagen_filtrada.pdf")

