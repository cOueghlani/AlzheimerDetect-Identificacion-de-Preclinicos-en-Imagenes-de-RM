"""
Naive Bayes
"""
########## LIBRERÍAS A UTILIZAR ##########
#Se importan la librerias a utilizar
from scipy import stats
from operator import index
from sklearn.metrics import f1_score
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
import matplotlib.pyplot as plt  # , plot
from matplotlib.gridspec import GridSpec
from sklearn import datasets, metrics, preprocessing
from sklearn.semi_supervised import LabelPropagation, SelfTrainingClassifier


########## PREPARAR LA DATA ##########
#Importamos los datos de la misma librería de scikit-learn
dataset = pd.read_excel(r'/home/cristina/Documentos/TFG/ALL_neurocloud_estudio_ALDAPA.xlsx',
                        sheet_name='normalizedCorrelation'
                        )



#_______________________Sin Normalizar_________________________________________

datasetEtiquetado = dataset[dataset['CSF_Result'].notna()] # Grupo etiquetado (tienen 0 o 1). Quito las filas 'NA'


# Grupo NO etiquetado (clase -1)
dataset['CSF_Result'] = dataset['CSF_Result']. fillna(-1)       # Modifico de la columna 'CSF_Result' para sustiruir los valores NaN por el valor -1
datasetNoEtiquetado = dataset[dataset.CSF_Result.isin([-1])]    # Solamente dejo las filas con la clase -1 (177 filas)


#_______________________Normalizando_________________________________________

# escala = (X - media) /std. NORMALIZACIÓN - PREPROCESAMIENTO DE DATOS mediante SciKitLearn_
datasetEtiquetadoNormalizado = StandardScaler().fit_transform(datasetEtiquetado)
datasetNoEtiquetadoNormalizado = StandardScaler().fit_transform(datasetNoEtiquetado)

# Vuelvo a crear el dataser sin las variables continuas y con los valores normalizados
datasetEtiquetadoNormalizado = pd.DataFrame(datasetEtiquetadoNormalizado)
datasetNoEtiquetadoNormalizado = pd.DataFrame(datasetNoEtiquetadoNormalizado)


print("Ejemplos usados para dataset Etiquetado : ", len(datasetEtiquetado))
print("Ejemplos usados para dataset No Etiquetado : ", len(datasetNoEtiquetado))
print("Ejemplos usados para dataset Etiquetado Normalizado: ", len(datasetEtiquetadoNormalizado))
print("ESTOO Ejemplos usados para dataset No Etiquetado Normalizado: ", len(datasetNoEtiquetadoNormalizado))
# print(datasetEtiquetado)
# print(datasetNoEtiquetado)
# print(datasetEtiquetadoNormalizado)
# print(datasetNoEtiquetadoNormalizado)
#print("Cabeceras dataset No Etiquetado Normalizado: ", datasetNoEtiquetadoNormalizado.columns.values)
# print("Cabeceras dataset No Etiquetado Normalizado: ", list(datasetEtiquetadoNormalizado.columns))
# print("Cabeceras dataset No Etiquetado Normalizado: ", list(datasetNoEtiquetadoNormalizado.columns))

 
datosAccTrain = [] 
datosAccTest = []
datosPrecisionTest = []
datosRecallTest = []
datosF1Test = []


 
#print("__________instancia del clasificador --> debe implementar la función predict_proba _____________________\n")
      
     
gaussian = GaussianNB()
knn = KNeighborsClassifier(5)
svc = SVC(probability=True)
#mlp = MLPClassifier()
algoritmoRF = RandomForestClassifier(random_state=0)
dtc = DecisionTreeClassifier(random_state=0)
 
for i in range (1,31):
      
      print("Iteración ", i, " --> \n")

      # Dividir de manera aleatoria el dataset --> 70% train y 30% test
      train, test = train_test_split(datasetEtiquetado, test_size=0.30)
      #print("Ejemplos usados para entrenar: ", len(train))      # 252 ejemplos usados para entrenar
      #print("Ejemplos usados para test:     ", len(test))       # 108 ejemplos usados para entrenar

      # Dividir de manera aleatoria el dataset normalizado --> 70% train y 30% test
      trainNormalizado, testNormalizado = train_test_split(datasetEtiquetadoNormalizado, test_size=0.30)
      #print(" Ejemplos usados para entrenar: ", len(trainNormalizado))      # 252 ejemplos usados para entrenar
      # print("Ejemplos usados para test:     ", len(testNormalizado))       # 108 ejemplos usados para entrenar



      # Juntar los dataset: el 70% del grupo etiquetado (clase 0 y 1) y el dataset no etiquetado (clase -1)
      framesConcatenated = train.merge(datasetNoEtiquetado, how='outer')
      # framesConcatenated = pd.concat([train, datasetNoEtiquetado], axis=1) #otra manera de lograr framesConcatenated

      # Juntar los dataset normalizado: el 70% del grupo etiquetado y normalizado (clase 0 y 1) y el dataset no etiquetado pero sí normalizado(clase -1)
      framesConcatenatedNormalizado = pd.concat([trainNormalizado, datasetNoEtiquetadoNormalizado],ignore_index=True)
      
      # print("Dataframe concatenado: ", len(framesConcatenated))      
      # print(" Dataframe normalizado concatenado: ", len(framesConcatenatedNormalizado))
      #print(framesConcatenatedNormalizado)



      # Elimino las columas C D E F medainte .drop
      #datasetSinNormalizado = framesConcatenatedNormalizado.drop(['2', '3', '4', '5'], axis=1)
      datasetSinNormalizado = framesConcatenatedNormalizado.drop((framesConcatenatedNormalizado.iloc[:,[2,3,4,5]]), axis=1)



      #Seleccionamos todas las columnas y defino los datos correspondientes a las etiquetas
      # la clase
      y = framesConcatenated['CSF_Result']
      framesConcatenated.drop('CSF_Result', axis=1, inplace=True)
      # las 12 columnas de las variables seleccionadas
      X = datasetSinNormalizado.iloc[:, [7, 19, 131, 148, 161, 128, 2, 3, 5, 6, 8, 11]]

      # escala = (X - media) /std. NORMALIZACIÓN - PREPROCESAMIENTO DE DATOS mediante SciKitLearn_
      #X = stats.zscore(X)

      # print("X es ", len(X))
      # print("y es ", len(y))


     

      #______________________________________________Gaussian__________________________________________________\n")
      #print("____________________sklearn.semi_supervised.SelfTrainingClassifier en TRAIN______________________\n")
      # definir modelo
      algoritmoNB = SelfTrainingClassifier(dtc)

      # Ajuste un modelo de propagación de etiquetas semisupervisado a X (conjunto de entrenamiento, TRAIN)
      algoritmoNB.fit(X, y)

      # Return the mean accuracy on the given test data and labels:
      print('Accuracy mean on train % .4f \n' % algoritmoNB.score(X, y))
      datosAccTrain.append(algoritmoNB.score(X, y))

      #print("Precisión mean on train: ",  algoritmoNB.predict(X),  "\n")

      #____________________________sklearn.semi_supervised.SelfTrainingClassifier en TEST _______________________________)

      # Elimino las columas C D E F medainte .drop
      datasetSinTest = testNormalizado.drop((testNormalizado.iloc[:, [2, 3, 4, 5]]), axis=1)

      #Sobre el 30% (test)
      y_test = test['CSF_Result']
      test.drop('CSF_Result', axis=1, inplace=True)  # la clase

      # las 12 columnas de las variables seleccionadas
      X_test = datasetSinTest.iloc[:, [7, 19, 131, 148, 161, 128, 2, 3, 5, 6, 8, 11]]

      # Transformo la matriz en array
      arrayX_test = algoritmoNB.predict(X_test)

      print('Accuracy mean on test: %.4f' %
            accuracy_score(arrayX_test, y_test))
      print('Precision mean on test: %.4f' %
            precision_score(arrayX_test, y_test))
      print('Recall mean on test: %.4f' % recall_score(arrayX_test, y_test))
      print('F1 mean on test: % .4f \n' % f1_score(arrayX_test, y_test))

      #Guardo los datos en las listas
      datosAccTest.append(accuracy_score(arrayX_test, y_test))
      datosPrecisionTest.append(precision_score(arrayX_test, y_test))
      datosRecallTest.append(recall_score(arrayX_test, y_test))
      datosF1Test.append(f1_score(arrayX_test, y_test))

hoja1 = pd.DataFrame([datosAccTrain, datosAccTest, datosPrecisionTest, datosRecallTest, datosF1Test],
                    index=['ACC mean on train', 'ACC mean on test', 'Precision mean on test', 'Recall mean on test', 'F1 mean on test'])

# Calculo la media y de la desviación estandar por cada fila
data = {'Media': [np.mean(datosAccTrain), np.mean(datosAccTest), np.mean(datosPrecisionTest), np.mean(datosRecallTest), np.mean(datosF1Test)],
        'DesviacionEstandar': [np.std(datosAccTrain), np.std(datosAccTest), np.std(datosPrecisionTest), np.std(datosRecallTest), np.std(datosF1Test)]
        }
hoja2 = pd.DataFrame(data, columns=['Media', 'DesviacionEstandar'])



# Write each dataframe to a different worksheet.
with pd.ExcelWriter('Resultados.xlsx') as writer:
      hoja1.to_excel(writer, sheet_name='Sheet_name_1')
      hoja2.to_excel(writer, sheet_name='Sheet_name_2')
      
      
      

print('Sales record successfully exported into Excel File')