# %% Modulos y datos

import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 

# Datos 
from sklearn.datasets import load_digits
digits = load_digits()
#en digits se guarda un conjutno de imagenes en forma de matrices de numeros con sus respectivos labels

# %% Tipo de informacion que almacena digits

type(digits)


# Lista de metodos que puedo aplicar a digits
dir(digits)
# ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']


# Primer metodo 
print(digits.DESCR)

# Analicemos el metodo images
type(digits.images)
digits.images.shape

# Visualicemos una de estas 1797 matrices de tamaño 8x8
observacion = 1000
print(digits.images[observacion])

# Usemos matplotlib
plt.figure(num = 1)
plt.imshow(digits.images[observacion], cmap = "binary")
plt.title("Imagen : " + str(digits.target[observacion]))


# Mostremos varios digitos 
fig,ax = plt.subplots(4,4, figsize = (5,5))
#la funcion de arriba crea 16 plots y guarda en fig los indices de cada plot y en ax el sobplot
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap = "binary")
    axi.set(xticks=[])#elimina las etiquetas horizontales de cada subplot

hola=digits.target
# Variable dependiente : target 
pd.DataFrame(digits.target).value_counts().sort_index().plot(kind = "bar")


# %% Construyamos un modelo de regresion logistica 
# La variable dependiente tiene 10 posibles valores
# Modelo de regresion multinomial 

# Particionado de los datos 
x_train, x_test, y_train, y_test = train_test_split(digits.data,
                                                    digits.target,
                                                    test_size = 0.2,
                                                    stratify = digits.target)

# Instanciar la clase a modelar
ModelBaseLog = LogisticRegression(multi_class = "multinomial",
                                  max_iter = 5000,
                                  verbose = 2)

# Ajustamos el modelo con los datos de entrenamiento : fit
ModelBaseLog.fit(x_train, y_train)

# Mostremos el score para entrenamiento y testeo
print("""
      Score (R^2):
              train : %f
              test : %f
      
      """ %(ModelBaseLog.score(x_train, y_train),
      ModelBaseLog.score(x_test, y_test)))

# %% Construccion de pronosticos 

# Seleccionemos un elemento delataset de testeo 
elemento = 1
DigitoPrueba = x_test[elemento].reshape(1, -1)
etiqueta = y_test[elemento]
# 
# Pronostico del modelo para DigitoPrueba
ModelBaseLog.predict(DigitoPrueba)

if ModelBaseLog.predict(DigitoPrueba) == etiqueta :
    print("El modelo no se equivoco")
else:
    print("El modelo se equuivoco")


# Construyamos pronosticos de varios elementos del dataset de testeo
# NUmero de elementos a testear
NumElementosTest = y_test.shape[0]

# Generemos 10 indices aleatorios 
MuestraIdx = np.random.randint(0, NumElementosTest+1, 10)


# Con estos indices , recuperemos los digitos para predecir 
DigitosPrueba1 = x_test[MuestraIdx]

# Construyamos los pronosticos para estos 10 digitos de prueba 
ForecastPrueba1 = ModelBaseLog.predict(DigitosPrueba1)
y_test[MuestraIdx]


# %% Evaluemos el performance del modelo 
# matriz de confusion
# accuracy
# precision
# recall
# Score f1
# Curva ROC.

Forecast = ModelBaseLog.predict(x_test)

# Calculo de la matriz de confusion
metrics.confusion_matrix(y_test ,Forecast)#filas valor real columnas valor predecido

# Generemos un grafico d esta matriz de confusion 
# Usemos unu heat map de seaborn 
sns.heatmap(metrics.confusion_matrix(y_test ,Forecast),
            fmt = "2d",
            annot = True,
            cmap = "gray_r",
            cbar = False)
plt.xlabel("Pronostico")
plt.ylabel("Valor Real (test)")
plt.title("Score " + str(ModelBaseLog.score(x_train,y_train)))
plt.show()














































