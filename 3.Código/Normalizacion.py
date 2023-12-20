"""
Normalización y preprocesameinto de datos mediante la librería scikitLearn
"""
import matplotlib.pyplot as plt#, plot
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import preprocessing


########## PREPARAR LA DATA ##########
dataset = pd.read_excel(r'/home/cristina/Documentos/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx',
                        sheet_name='normalizedCorrelation'
                        )
dataset = dataset[dataset['CSF_Result'].notna()] # quito las filas 'NA'. train_test_split --> tiene que tener mismo tam

df = pd.DataFrame(dataset) # Como se van a cambiar los datos realizo una copia para que no se modifiquen en la fuente original


datasetSin = dataset.drop(['CSF_Result',
                           'CSF_AB42_Cutoff_LUD/Sant_pau',
                           'CSF_TTAU_Cutoff_LUD/Sant_pau',
                           'CSF_PTAU_Cutoff_LUD/Sant_pau'
                           ], axis=1)
y = dataset['CSF_Result']                                           # la clase
dataset.drop('CSF_Result', axis=1, inplace=True)
# las 12 columnas de las variables seleccionadas
X = datasetSin.iloc[:, [7, 19, 131, 148, 161, 128, 2, 3, 5, 6, 8, 11]]

print()
print("Antes de la normalización")
print("Media: ", X.mean())
print("Desviación estandar", X.std())
print("Max: ", X.max())
print("Min: ", X.min())

#____________NORMALIZACIÓN - PREPROCESAMIENTO DE DATOS mediante SciKitLearn_________________________________________________

df_scaler = preprocessing.StandardScaler().fit_transform(df)    #estandarizado = (X - media) /std

# Comparación de métodos
# convierte vectores de numpy a DataFrames para graficarlos
df_scaler = pd.DataFrame(df_scaler)

# crea una figura con 4 subfiguras para comparar los métodos
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1, 4, 1)
ax2 = fig.add_subplot(1, 4, 2)
ax3 = fig.add_subplot(1, 4, 3)
ax4 = fig.add_subplot(1, 4, 4)


ax1.set_title("Datos originales")
ax1.plot(df)
ax2.set_title(" Datos normalizados")
ax2.plot(df_scaler)

ax3.set_title("HIST:    Datos originales")
ax3.hist(df)

ax4.set_title("Datos normalizados")
ax4.hist(df_scaler)

#df_scaler.to_excel('DatosNormalizados.xlsx')



#Para la X
X_scaler = preprocessing.StandardScaler().fit_transform(X)  # estandarizado = (X - media) /std
X_scaler = pd.DataFrame(X_scaler)


print()
print("Después de la normalización")
print("Media: ", X_scaler.mean())
print("Desviación estandar", X_scaler.std())
print("Max: ", X_scaler.max())
print("Min: ", X_scaler.min())


fig = plt.figure(figsize=(15, 5))

ax5 = fig.add_subplot(1, 4, 1)
ax6 = fig.add_subplot(1, 4, 2)
ax7 = fig.add_subplot(1, 4, 3)
ax8 = fig.add_subplot(1, 4, 4)

ax5.set_title("Datos originales de X")
ax5.plot(X)
ax6.set_title("Datos normalizados de X")
ax6.plot(X_scaler)
ax7.set_title("HIST:    Datos originales de X")
ax7.hist(X)
ax8.set_title("Datos normalizados de X ")
ax8.hist(X_scaler)
#plt.show()
#df_scaler.to_excel('DatosNormalizadosDeX.xlsx') 