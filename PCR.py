#Se importa el paquete de numpy y matplotlib
import matplotlib.pylab as plt
import numpy as np
from numpy.linalg import *

#Se cargan los datos y se especifica que datos usar
data=np.genfromtxt("WDBC.dat",delimiter=",",dtype='U16')
filas=np.shape(data)[0]
colum=np.shape(data)[1]
dat=np.zeros((filas,colum-2))
M=np.zeros((filas,colum))
mal = 212
ben = 357
a=0
b=0

#Se declara la matriz de covarianza basandome en la formula hallada en https://stattrek.com/matrix-algebra/covariance-matrix.aspx

def covarianza(x): 
	size = np.shape(x)[1]
	p = np.shape(x)[0] 
	cov = np.zeros([size,size]) 
	for i in range(size):
		for j in range(size):
			meani = np.mean(x[:,i])
			meanj = np.mean(x[:,j])
			cov[i,j] = (np.sum((x[:,i]-meani)*(x[:,j]-meanj)))/(p-1)
	return cov 

#Se especifica el numero de compenentes en malignos y benignos
benig=np.zeros((ben,np.shape(dat)[1]))
malig=np.zeros((mal,np.shape(dat)[1]))

#Se especifica un for que recorra las columnas separando los valores malignos y los benignos

for i in range(np.shape(data)[0]):
	for k in range(np.shape(data)[1]): 
		if (k==0):
			M[i,k]=float(data[i,k])
		elif (k==1):
			if (data[i,k]=='B'):
				M[i,k]=1
				benig[a,:]=data[i,k+1:]
				a=a+1
			else:
				M[i,k]=0
				malig[b,:]=data[i,k+1:]
				b=b+1
		else:
			M[i,k]=float(data[i,k])
			dat[i,k-2]=float(data[i,k])

#Se halla la matriz de covarianza y se imprime
mat=covarianza(dat)
print ("La matriz de covarianza es", mat)

#Se hallan los auto valores y auto vectores
sol=np.linalg.eig(mat)
evalores=sol[0]
evectores=sol[1]

#Se hace un for que imprima todos los valores propios con su vector propio.
for i in range(len(evalores)):
	print("Valor propio",i,":",evalores[i],"Vector propio:", evectores[i])

#Imprime los parametros mas importantes en base a las componentes de los autovectores
print("El parametro principal es", evalores[0], "y su vector propio", evectores[:,0], "el parámetro secundario principal es", evalores[1], "con su vector propio:", evectores[:,1])

#Multiplica los vectores para graficas PCA
plot_B=np.dot(benig,evectores)
plot_M=np.dot(malig,evectores)

#Se grafican los datos y se guardan
plt.figure()
plt.title("PCA")
plt.scatter(plot_B[:,0],plot_B[:,1],label="Benignos", color="yellow")
plt.scatter(plot_M[:,0],plot_M[:,1],label="Malignos", color="red")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.savefig("GarciaCamila_PCA.pdf")
plt.figure()

#Se imprime un mensaje diciendo si el metodo de PCA es util para hacer esta clasificacion

print("Para este caso el método de PCA es util para diagnosticar pacientes con cancer maligno ya que estos pueden presentar valores muy alto en las variables, asi como se puede ver en la grafica que no existen valores benignos en esa zona, mientras que en general los pacientes de cáncer benigno poseen valores mas bajos. El método de PCA permite reducir el tamaño de las variables que son estudiadas y nos ayuda a  proyectarlas en un sistema 2D.")

