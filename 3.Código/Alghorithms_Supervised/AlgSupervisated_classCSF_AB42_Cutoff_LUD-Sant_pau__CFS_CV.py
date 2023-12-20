"""
Naive Bayes
"""
########## LIBRERÍAS A UTILIZAR ##########
#Se importan la librerias a utilizar
import pydotplus
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from matplotlib.colors import ListedColormap
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn import tree, show_versions
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, precision_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt#, plot
from matplotlib.gridspec import GridSpec
from sklearn import datasets, metrics, preprocessing


########## PREPARAR LA DATA ##########
#Importamos los datos de la misma librería de scikit-learn
dataset = pd.read_excel(r'/home/cristina/Documentos/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx',
                        sheet_name='normalizedCorrelation'
                        )
dataset = dataset[dataset['CSF_AB42_Cutoff_LUD/Sant_pau'].isin([0,1])] # quito las filas '9999'

# Elimino las columas C D E F medainte .drop
datasetSin = dataset.drop(['CSF_Result',
                        'CSF_AB42_Cutoff_LUD/Sant_pau',
                        'CSF_TTAU_Cutoff_LUD/Sant_pau',
                        'CSF_PTAU_Cutoff_LUD/Sant_pau'
                        ], axis=1)  

#print(dataset)

########## ENTENDIMIENTO DE LA DATA ##########
#Verifico la información contenida en el dataset, nombre de las columnas
print('Cabeceras del dataset:')
print(dataset.keys())
print()

#Detalles estadísticos como percentil, media, estándar... de cada columna
print('Características del dataset:')
print(dataset.describe())
#print(datasetSin.describe())

#Seleccionamos todas las columnas y defino los datos correspondientes a las etiquetas
y = dataset['CSF_AB42_Cutoff_LUD/Sant_pau']# la clase
#dataset.drop('CSF_AB42_Cutoff_LUD/Sant_pau', axis=1, inplace=True)
X = datasetSin.iloc[:, [7, 19, 131, 148, 128, 161, 2, 3, 5, 6]]  # las 10 columnas de las variables seleccionadas


X = StandardScaler().fit_transform(X)  # escala = (X - media) /std. NORMALIZACIÓN - PREPROCESAMIENTO DE DATOS mediante SciKitLearn_

print()
print("____________________NAIVE BAYES_____________________________\n")

algoritmoNB = GaussianNB()

# Ajustar el modelo en los datos de entrenamiento (conjunto de entrenamiento, TRAIN)
algoritmoNB.fit(X, y)

print("Precision del modelo en Train",  algoritmoNB.score(X, y))


# Cross-Validation usando K-folds con 10 splits (Lo divide en 10 pliegues, entrena 9 y luego prueba con el restante)
k_fold = KFold(n_splits=10)

scoreNB_Acc = cross_val_score(
    algoritmoNB, X, y, cv=k_fold, scoring="accuracy")
scoreNB_Pre = cross_val_score(
    algoritmoNB, X, y, cv=k_fold, scoring="precision")
scoreNB_Rec = cross_val_score(
    algoritmoNB, X, y, cv=k_fold, scoring="recall")
scoreNB_f1 = cross_val_score(
    algoritmoNB, X, y, cv=k_fold, scoring="f1_macro")

# Genera la estimacion con cross-validation para cada punto de datos de entrada
print("Accuracy Score Test: ", scoreNB_Acc)
print("Precision Score Test: ", scoreNB_Pre)
print("Recall Score Test: ", scoreNB_Rec)
print("F1 Score Test: ", scoreNB_f1, "\n")


# Puntuación media y desviación estándar
print("Accuracy  -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreNB_Acc.mean(), scoreNB_Acc.std()))
print("Precision -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreNB_Pre.mean(), scoreNB_Pre.std()))
print("Recall    -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreNB_Rec.mean(), scoreNB_Rec.std()))
print("F1        -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f \n" %
      (scoreNB_f1.mean(), scoreNB_f1.std()))


print()
print("____________________KNN_____________________________________\n")

algoritmoKNN = KNeighborsClassifier(5)

algoritmoKNN.fit(X, y)                   # Ajustar el modelo en los datos de entrenamiento (conjunto de entrenamiento, TRAIN)


print("Precision del modelo en Train",  algoritmoKNN.score(X, y)    )


k_fold = KFold(n_splits=10) # Cross-Validation usando K-folds con 10 splits (Lo divide en 10 pliegues, entrena 9 y luego prueba con el restante)

scoreKNN_Acc= cross_val_score (algoritmoKNN, X, y, cv=k_fold, scoring="accuracy")
scoreKNN_Pre = cross_val_score(algoritmoKNN, X, y, cv=k_fold, scoring="precision")
scoreKNN_Rec = cross_val_score(algoritmoKNN, X, y, cv=k_fold, scoring="recall")
scoreKNN_f1 = cross_val_score(algoritmoKNN, X, y, cv=k_fold, scoring="f1_macro")

# Genera la estimacion con cross-validation para cada punto de datos de entrada
print("Accuracy Score Test: ", scoreKNN_Acc)
print("Precision Score Test: ", scoreKNN_Pre)
print("Recall Score Test: ", scoreKNN_Rec)
print("F1 Score Test: ", scoreKNN_f1, "\n")


# Puntuación media y desviación estándar
print("Accuracy  -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " % (scoreKNN_Acc.mean(), scoreKNN_Acc.std()))  
print("Precision -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " % (scoreKNN_Pre.mean(), scoreKNN_Pre.std()))
print("Recall    -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " % (scoreKNN_Rec.mean(), scoreKNN_Rec.std()))
print("F1        -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f \n" % (scoreKNN_f1.mean(), scoreKNN_f1.std()))


print()
print("____________________SVM_____________________________________\n")

algoritmoSVM = svm.SVC()

# Ajustar el modelo en los datos de entrenamiento (conjunto de entrenamiento, TRAIN)
algoritmoSVM.fit(X, y)

print("Precision del modelo en Train",  algoritmoSVM.score(X, y))


# Cross-Validation usando K-folds con 10 splits (Lo divide en 10 pliegues, entrena 9 y luego prueba con el restante)
k_fold = KFold(n_splits=10)

scoreSVM_Acc = cross_val_score(
    algoritmoSVM, X, y, cv=k_fold, scoring="accuracy")
scoreSVM_Pre = cross_val_score(
    algoritmoSVM, X, y, cv=k_fold, scoring="precision")
scoreSVM_Rec = cross_val_score(
    algoritmoSVM, X, y, cv=k_fold, scoring="recall")
scoreSVM_f1 = cross_val_score(
    algoritmoSVM, X, y, cv=k_fold, scoring="f1_macro")

# Genera la estimacion con cross-validation para cada punto de datos de entrada
print("Accuracy Score Test: ", scoreSVM_Acc)
print("Precision Score Test: ", scoreSVM_Pre)
print("Recall Score Test: ", scoreSVM_Rec)
print("F1 Score Test: ", scoreSVM_f1, "\n")


# Puntuación media y desviación estándar
print("Accuracy  -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreSVM_Acc.mean(), scoreSVM_Acc.std()))
print("Precision -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreSVM_Pre.mean(), scoreSVM_Pre.std()))
print("Recall    -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreSVM_Rec.mean(), scoreSVM_Rec.std()))
print("F1        -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f \n" %
      (scoreSVM_f1.mean(), scoreSVM_f1.std()))


print()
print("____________________Multi-layer Perceptron__________________\n")

algoritmoMLP = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)

# Ajustar el modelo en los datos de entrenamiento (conjunto de entrenamiento, TRAIN)
algoritmoMLP.fit(X, y)

print("Precision del modelo en Train",  algoritmoMLP.score(X, y))


# Cross-Validation usando K-folds con 10 splits (Lo divide en 10 pliegues, entrena 9 y luego prueba con el restante)
k_fold = KFold(n_splits=10)

scoreMLP_Acc = cross_val_score(
    algoritmoMLP, X, y, cv=k_fold, scoring="accuracy")
scoreMLP_Pre = cross_val_score(
    algoritmoMLP, X, y, cv=k_fold, scoring="precision")
scoreMLP_Rec = cross_val_score(
    algoritmoMLP, X, y, cv=k_fold, scoring="recall")
scoreMLP_f1 = cross_val_score(
    algoritmoMLP, X, y, cv=k_fold, scoring="f1_macro")

# Genera la estimacion con cross-validation para cada punto de datos de entrada
print("Accuracy Score Test: ", scoreMLP_Acc)
print("Precision Score Test: ", scoreMLP_Pre)
print("Recall Score Test: ", scoreMLP_Rec)
print("F1 Score Test: ", scoreMLP_f1, "\n")


# Puntuación media y desviación estándar
print("Accuracy  -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreMLP_Acc.mean(), scoreMLP_Acc.std()))
print("Precision -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreMLP_Pre.mean(), scoreMLP_Pre.std()))
print("Recall    -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreMLP_Rec.mean(), scoreMLP_Rec.std()))
print("F1        -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f \n" %
      (scoreMLP_f1.mean(), scoreMLP_f1.std()))


print()
print("____________________RANDOM FOREST___________________________\n")

algoritmoRF = RandomForestClassifier(random_state=0)

# Ajustar el modelo en los datos de entrenamiento (conjunto de entrenamiento, TRAIN)
algoritmoRF.fit(X, y)

print("Precision del modelo en Train",  algoritmoRF.score(X, y))


# Cross-Validation usando K-folds con 10 splits (Lo divide en 10 pliegues, entrena 9 y luego prueba con el restante)
k_fold = KFold(n_splits=10)

scoreRF_Acc = cross_val_score(
    algoritmoRF, X, y, cv=k_fold, scoring="accuracy")
scoreRF_Pre = cross_val_score(
    algoritmoRF, X, y, cv=k_fold, scoring="precision")
scoreRF_Rec = cross_val_score(
    algoritmoRF, X, y, cv=k_fold, scoring="recall")
scoreRF_f1 = cross_val_score(
    algoritmoRF, X, y, cv=k_fold, scoring="f1_macro")

# Genera la estimacion con cross-validation para cada punto de datos de entrada
print("Accuracy Score Test: ", scoreRF_Acc)
print("Precision Score Test: ", scoreRF_Pre)
print("Recall Score Test: ", scoreRF_Rec)
print("F1 Score Test: ", scoreRF_f1, "\n")


# Puntuación media y desviación estándar
print("Accuracy  -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreRF_Acc.mean(), scoreRF_Acc.std()))
print("Precision -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreRF_Pre.mean(), scoreRF_Pre.std()))
print("Recall    -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreRF_Rec.mean(), scoreRF_Rec.std()))
print("F1        -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f \n" %
      (scoreRF_f1.mean(), scoreRF_f1.std()))


print()
print("____________________DECISION TREE___________________________\n")

algoritmoDT = DecisionTreeClassifier(random_state=0)

# Ajustar el modelo en los datos de entrenamiento (conjunto de entrenamiento, TRAIN)
algoritmoDT.fit(X, y)

print("Precision del modelo en Train",  algoritmoDT.score(X, y))


# Cross-Validation usando K-folds con 10 splits (Lo divide en 10 pliegues, entrena 9 y luego prueba con el restante)
k_fold = KFold(n_splits=10)

scoreDT_Acc = cross_val_score(
    algoritmoDT, X, y, cv=k_fold, scoring="accuracy")
scoreDT_Pre = cross_val_score(
    algoritmoDT, X, y, cv=k_fold, scoring="precision")
scoreDT_Rec = cross_val_score(
    algoritmoDT, X, y, cv=k_fold, scoring="recall")
scoreDT_f1 = cross_val_score(
    algoritmoDT, X, y, cv=k_fold, scoring="f1_macro")

# Genera la estimacion con cross-validation para cada punto de datos de entrada
print("Accuracy Score Test: ", scoreDT_Acc)
print("Precision Score Test: ", scoreDT_Pre)
print("Recall Score Test: ", scoreDT_Rec)
print("F1 Score Test: ", scoreDT_f1, "\n")


# Puntuación media y desviación estándar
print("Accuracy  -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreDT_Acc.mean(), scoreDT_Acc.std()))
print("Precision -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreDT_Pre.mean(), scoreDT_Pre.std()))
print("Recall    -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f " %
      (scoreDT_Rec.mean(), scoreDT_Rec.std()))
print("F1        -->  Puntuacion media %0.4f , con una desviacion estandar de %0.4f \n" %
      (scoreDT_f1.mean(), scoreDT_f1.std()))


